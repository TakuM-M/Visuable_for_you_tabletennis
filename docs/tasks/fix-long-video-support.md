# fix: 長時間動画（30分級）のアップロード・処理対応

**Status: In Progress**

重要度: 高

## 背景・目的

30分級の動画をアップロードした際の挙動を「アップロード → ML 解析 → 書き出し」の
全経路で評価し、破綻する箇所を潰す。評価時点では受け入れ上限がどこにも定義されておらず、
何分の動画まで面倒を見るのかがコードからもUIからも読み取れない状態だった。

前提となる規模感（1080p/30fps・30分）:

| 項目 | 概算 |
| --- | --- |
| ファイルサイズ | 2〜4GB（チャンク 40〜80 本） |
| 総フレーム数 | 約 54,000 |
| GPU 推論時間 | 20〜40分 |
| 検出されるプレー区間 | 100〜200 本 |
| 書き出し（FFmpeg 再エンコード）| 出力尺の 1〜2 倍の時間 |

## 評価結果（対応前の問題点）

### A. 受け入れ上限が存在しない — 対応済み

サイズ・長さの検証がどこにも無く、UI の「最大 5GB」は表示だけだった。3時間の動画でも
そのまま R2 に保存され GPU に流れる。長い動画ほど GPU 課金と待ち時間が線形に伸びるため、
入口で止まらないことが一番効く欠陥だった。

### B. RunPod の実行時間上限が未指定 — 対応済み

`/run` に `policy` を渡しておらず、エンドポイント既定の実行時間上限が効く。30分動画の
推論はそれを超えうるため、途中で `TIMED_OUT` になる。backend からは「理由不明の異常終了」
にしか見えず、しかも自動リトライで同じ失敗をもう一度 GPU 課金つきで繰り返す。

あわせて元動画 presigned URL の期限が 2 時間固定だった。ワーカー枯渇でキューに積まれた
まま期限が切れると、GPU が起動した直後に 403 で落ちる。

### C. ディスクのピーク使用量が動画サイズの 2 倍 — 対応済み

`process_chunk_upload` はチャンク群を残したまま結合ファイルを作っていた（3GB の動画で
6GB を一時占有）。さらに失敗時は中間ファイルが `tmp_retention_hours`（既定 24時間）まで
残るため、失敗が続くとディスクが先に尽きる。本番 backend にボリュームが無く、これらが
コンテナの書き込みレイヤーに書かれていた点も合わせて修正した。

### D. FFmpeg 書き出しのタイムアウト・並列度・同時実行制御が無い — 対応済み

- `subprocess.run` に timeout が無く、進まなくなった FFmpeg を無期限に待つ
- `capture_output=True` のまま `CalledProcessError` を投げるので stderr がどこにも出ず、
  失敗原因（壊れた入力・ディスク不足）が追えない
- プレー区間は 1 本数秒と短く、逐次実行では起動・シークのオーバーヘッドが支配的
- 同時実行制御が無く、複数ユーザーが書き出すと VPS の CPU を食い尽くす

### E. コールバック 1 回勝負 — 対応済み

ML 側のコールバックは timeout 10 秒・リトライ無し。ここで落ちると backend からは
「RunPod は COMPLETED なのにコールバック未達」となり、20〜40分の推論をやり直すことになる。

### F. 推論が全フレーム処理になっていた — 対応済み（config JSON の反映が残り）

`VideoProcessingConfig.target_fps` の既定が 30.0 で、30fps 動画では `frame_step=1`、
つまり全 54,000 フレームに姿勢推定が走っていた。**30分動画の待ち時間と GPU 課金の
主因はここ**。学習データ側（`02_export_player_pose.ipynb`）と評価（`05_evaluate_*`）は
15fps 間引き前提で、推論だけが 30fps だった。

### F-2. 15fps に切り替えると顕在化するフレーム番号のズレ — 対応済み

`target_fps` を変えるだけでは済まなかった。`PlaySceneDetector._predict` は
シーケンス内 i 番目の実フレーム番号を `metadata["start_frame"] + i` として数えていたが、
`start_frame` は**実フレーム番号**、`i` は**サンプルのインデックス**である。間引きが無い
（`frame_step=1`）間は両者が一致するので問題が出ないが、15fps 間引き（`frame_step=2`）では
実フレーム番号が 2 刻みになるためズレる。

検証（`sequence_length=30`・実フレーム 1000〜1120 の区間が全てプレーと判定された場合）:

| フレーム番号の求め方 | 出力される区間 | 尺 |
| --- | --- | --- |
| 真の区間 | 1000〜1120 | 4.00秒 |
| 現行 `start_frame + i` | 1000〜1091 | 3.03秒（末尾 29 フレーム欠落）|
| 修正後 `dataset.frames[start_idx + i]` | 1000〜1120 | 4.00秒 |

欠落量は `sequence_length - 1` フレームで固定なので、15fps では**全区間の末尾が
一律 1 秒近く切れる**（ラリーの決め球が落ちる）。正規化に失敗したフレームは
`get_pose_data_for_dataset` でスキップされるため、欠測がある区間ではさらにズレる。

評価ノート（05）はこれを避けて `start_idx` から `dataset.frames` を引いており
（「複数トラックの行やフレーム欠落があっても壊れない」というコメントがある）、
本番推論だけが古い数え方のまま残っていた。つまり**評価指標は正しく、本番だけがずれる**
状態だったため、30fps のままでは誰も気付けなかった。

修正は本番も評価ノートと同じ方式に揃えた。

**残作業**: RunPod の Network Volume 上の `/workspace/configs/runpod_config.json`
（リポジトリ管理外。`.gitignore` の `configs/` で除外）の `video_processing.target_fps`
を 15 にする。リポジトリ側のフォールバックは 15.0 に変更済みなので、JSON からキーごと
消す対応でもよい。学習データ生成に使った `crip_app_config.json` と
`tracking_export` / `player_classification` の値が揃っているかも合わせて確認すること
（`min_consecutive_frames` や `recent_frames_window` はサンプル数で数えるため、
fps が変わると実時間の意味が変わる）。

判断材料として、`runpod_handler.py` に処理フレーム数と推論所要時間のログを追加済み。
15fps 化で推論時間は概ね半減する見込みだが、`runpod_execution_timeout_ratio`（3.0）は
安全側の値なのでそのままでよい。

### G. 書き出しの進捗がユーザーに見えない — 未対応

30分動画の書き出しは数十分かかりうるが、画面上は `processing` のまま変化しない。
進捗率を返すにはジョブ状態の持ち方（clips 単位の進捗カラム or 別テーブル）から
設計が必要なため、このタスクの範囲外とした。

## 対応内容

### 受け入れ上限（`backend/app/core/config.py`）

| 設定 | 既定値 | 意味 |
| --- | --- | --- |
| `max_upload_bytes` | 5GB | アップロード全体のサイズ上限 |
| `max_video_duration_seconds` | 3600（60分） | 解析を受け付ける動画長の上限 |
| `max_chunk_bytes` | 55MB | 1 チャンクの上限（nginx の 60M の内側） |

三段構えで検査する。

1. **フロント**: 選択時にサイズと再生時間（`<video>` のメタデータ）を見て送信前に弾く
2. **init**: 申告された `total_bytes` で、1 バイトも受け取る前に判定
3. **チャンク受信 / 結合後**: 実際の受信量を累積で検査し、結合後に ffprobe で長さを確認

上限超過は HTTP 413 で返す。フロントの `isTransient` は 413 をリトライ対象にしないため、
無駄な再送も起きない。

### GPU ディスパッチ（`video_service.call_ml_service`）

- `policy.executionTimeout` を動画長から算出して明示
  （`base 900秒 + 3.0 × 動画長`、上限 3時間。30分動画なら 105分）
- presigned URL の期限を 6 時間に延長（キュー待ち + ダウンロードを賄う）

### ディスク（`video_service.process_chunk_upload`）

- チャンクは結合しながら 1 本ずつ削除し、ピークを動画 1 本分に抑える
- 成功・失敗によらず `finally` で中間ファイルを消す
- 長さの検証を R2 アップロードより前に移動（上限超過の動画を保存してから消す往復を廃止）
- 本番 compose の backend に `backend-tmp` ボリュームを追加

### 書き出し（`video_clip_service` / `video_service.process_export`）

- FFmpeg に timeout を設定し、失敗時は stderr 末尾をエラーメッセージに含める
- セグメント切り出しを並列化（既定 2、CPU コア数とセグメント数で頭打ち）
- セマフォで同時書き出しを 1 件に制限。順番待ちが 30分を超えたら `ready` に戻す

### ML 側

- `target_fps` の既定を 15.0 に（`src/pipelines/config.py` と `runpod_handler.py` の
  フォールバック。姿勢推定の実行回数が半分になる）
- `PlaySceneDetector._predict` の実フレーム番号を `dataset.frames` から引くよう修正
- コールバックを 3 回までリトライ（推論のやり直しを避ける）
- ダウンロードサイズ・所要時間・処理フレーム数・推論時間をログに出力

なお ml 側にはテストディレクトリも venv も無いため、F-2 の検証は
`MemoryPoseSequenceDataset` 相当のフレーム配列（2 刻み）に対して両方式の被覆範囲を
計算する形で行った。テスト基盤を作る際はこのケースを回帰テストにすること。

## 受け入れ条件

- [x] サイズ・長さの上限がコードと UI の両方に明示されている
- [x] 上限超過が「送信前」「init」「受信中」の各段階で止まる
- [x] RunPod の実行時間上限が動画長から決まる
- [x] チャンク結合時のディスクピークが動画 1 本分に収まる
- [x] 中間ファイルが失敗時にも残らない
- [x] FFmpeg がハングしても書き出しスロットを占有し続けない
- [x] backend テストが通る（282 件）
- [x] 推論が学習時と同じ 15fps 間引きになる（リポジトリ側の既定値）
- [x] fps 間引き時にシーン区間の末尾が欠落しない
- [ ] RunPod の `runpod_config.json` に 15fps を反映する（Network Volume 上）
- [ ] 実機で 30分動画を通してエンドツーエンドの所要時間を計測する
- [ ] 15fps 化による検出精度の変化を 05 のノートブックで確認する

## 関連ファイル

- `backend/app/core/config.py`（上限・タイムアウトの設定を集約）
- `backend/app/services/video_service.py`
- `backend/app/services/video_clip_service.py`
- `backend/app/routers/videos.py` / `backend/app/schemas/video.py`
- `frontend/src/lib/chunkedUpload.ts` / `frontend/src/pages/VideoUploadPage.tsx`
- `ml/runpod_handler.py` / `ml/src/pipelines/config.py`
- `ml/src/pipelines/play_scene_detector.py`（フレーム番号の対応づけ）
- `nginx/nginx.conf` / `docker-compose.yml`

## 進捗ログ

- 2026-07-28: 全経路を評価し、A〜E を修正。F・G は判断待ち・範囲外として記録
- 2026-07-28: 推論も 15fps が本来の想定と判明したため F を対応。
  切り替えで顕在化するフレーム番号のズレ（F-2）を発見・修正。
  RunPod 上の config JSON への反映が残り
