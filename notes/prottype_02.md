# 卓球のプレー中のみを切り抜くアプリ開発
- プレー映像から必要な部分，つまりプレー中のみを抽出しプレー間のいらない時間をカットする

## Prottypse01
- 撮影する画角の固定
- 高画質撮影の想定
- 基本的なプレイスタイルを想定する
- ユーザがあらかじめ卓球台の位置を指定する(四角)

## Task01
- 卓球台の四角の座標から，プレーエリアを特定する

## もっと簡単にする
- まずは画面中央に卓球台がある想定で作成
- プレイヤーは画面中央にいる想定

## もっともっと簡単にする
- 検出するのは手前にいる選手だけ
- 手前選手がサーブ, レシーブの動作からプレー中, プレー外を判定する

## モデルを学習させるステップ
1. 選手の骨格データを獲得する
    - yolopose11を利用して骨格データを取得
    - バウンディングボックスについては画面中央で固定
既に以下が実装済み:
✅ YOLOv11-poseによる骨格検出 (yolo_tracker.py)
✅ 動画からのフレーム読み込み (video_loader.py)
✅ 画面中央の選手検出 (data_collection/player_detector.py)
✅ CSVへの骨格データ出力 (同上)
- 獲得する骨格データの質を上げる
    - 座標の「正規化」と「相対化」
    - 絶対座標ではなく相対座標にする
    - スケール（大きさ）を統一する
    - ノイズデータの除去（Visibilityの活用）
    - Savitzky-Golay（サビツキー・ゴーレイ）フィルタ

# 正規化データを出力（デフォルト）
python ./src/data_collection/player_detector.py \
    -i data/raw/sample_video_01_short_version.MOV \
    --csv output/pose_normalized.csv \
    --conf 0.3 \
    --center-ratio 0.60

# 可視化
python src/preprocessing/visualize_normalized_pose.py \
    -i output/pose_normalized.csv \
    -n 6 \
    --track-id 4

2. プレイ中/プレイ外のラベル付け
- ウィンドウスイライドングで時系列データを分割

3. モデル学習
    - ラベル付きデータを用いてモデルを学習させる
    - LSTMやTransformerなどの時系列モデルを検討