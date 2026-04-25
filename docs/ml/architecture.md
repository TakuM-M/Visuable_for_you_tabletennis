    ┌─────────────────────────────────────────────────────────────────┐
    │  ユーザー (ブラウザ)                                              │
    │  動画をアップロード                                               │
    └──────────────┬──────────────────────────────────────────────────┘
                ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Backend (FastAPI)                                               │
    │  video_service.py                                                │
    │                                                                  │
    │  1. 動画を /app/uploads/videos/ に保存                            │
    │  2. Video + Job レコードをDBに作成                                 │
    │  3. BackgroundTask で call_ml_service() を実行                    │
    │     ├─ USE_RUNPOD=true  → RunPod API に POST                    │
    │     └─ USE_RUNPOD=false → ml-mock に POST（開発用）              │
    └──────────────┬──────────────────────────────────────────────────┘
                ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  RunPod Serverless GPU Worker                                    │
    │  runpod_handler.py                                               │
    │                                                                  │
    │  起動時（コールドスタート）:                                        │
    │    ・JSON設定読み込み (runpod_config.json)                         │
    │    ・3つのモデルをGPUにロード                                       │
    │    ・InferencePipeline インスタンスをグローバルに保持                 │
    │                                                                  │
    │  リクエスト受信時:                                                 │
    │    1. Backend から動画をHTTPダウンロード                             │
    │    2. _pipeline.process_video() を実行                            │
    │    3. 結果(clips)を Backend にコールバックPOST                      │
    └──────────────┬──────────────────────────────────────────────────┘
                ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │  Backend (コールバック受信)                                        │
    │  job_service.py → complete_job()                                 │
    │                                                                  │
    │  1. clips の start_time/end_time で FFmpeg クリッピング             │
    │  2. Clip レコードをDBに保存                                        │
    │  3. Job/Video ステータスを completed に更新                         │
    │  4. 完了メールを送信                                               │
    └──────────────────────────────────────────────────────────────────┘

-----------------------------------------------------------------------

    InferencePipeline.process_video()          ← inference_pipeline.py
    │
    ├── Task1: 骨格データ抽出 ──────────────────────────────────────
    │   │
    │   │  PlayerPoseExporter.process_video()  ← player_pose_exporter.py
    │   │
    │   │  Step A: 卓球台検出
    │   │  ├── cap.read() でフレームをサンプリング
    │   │  └── TableDetector.detect_table_frame()  ← table_detector.py
    │   │      └── Custom YOLO で "Ping Pong Table" を検出
    │   │          画面中央に最も近い台を選択
    │   │          ※1回検出すれば500フレームキャッシュ
    │   │
    │   │  Step B: 全フレームループ ★最大のボトルネック
    │   │  ├── cap.read() で1フレーム読み込み
    │   │  ├── frame_step でスキップ判定
    │   │  │   （target_fps=30, 動画30fps → frame_step=1 → 全フレーム処理）
    │   │  │
    │   │  ├── YOLOPose_Tracker.track_frame_with_table_filter()
    │   │  │   └── yolopose_tracker.py
    │   │  │       ├── model.track(frame)  ← YOLO11l-pose で1フレーム推論
    │   │  │       │   17キーポイント × 検出人数 を返す
    │   │  │       └── 卓球台から遠い人物をフィルタリング
    │   │  │
    │   │  ├── PlayerClassifier.update() + classify_players()
    │   │  │   └── player_classifier.py
    │   │  │       運動量 + 台への近接率 でスコア計算
    │   │  │       上位2名をプレイヤーとして選定
    │   │  │
    │   │  └── TrackingExporter.add_frame()   ← tracking_exporter.py
    │   │      プレイヤーのキーポイントをメモリに蓄積
    │   │
    │   │  Step C: 後処理
    │   │  ├── TrackingExporter.filter_by_consecutive_frames()
    │   │  │   断片的な出現区間を除去（30フレーム未満の区間を削除）
    │   │  └── TrackingExporter.normalize_poses()
    │   │      腰中心を原点、腰幅でスケール正規化 → (17×2=34次元)
    │   │
    ├── Task2: プレーシーン検出 ──────────────────────────────────────
    │   │
    │   │  PlaySceneDetector.detect_from_exporter()  ← play_scene_detector.py
    │   │
    │   │  Step D: データ準備
    │   │  ├── TrackingExporter.get_pose_data_for_dataset()
    │   │  │   正規化済み骨格データ (N, 34) を取得
    │   │  └── MemoryPoseSequenceDataset を作成
    │   │      30フレームのスライディングウィンドウ
    │   │
    │   │  Step E: LSTM推論
    │   │  ├── PlayClassifierLSTM に各シーケンスを入力
    │   │  │   Bi-LSTM (2層, hidden=128) + Attention
    │   │  │   入力: (1, 30, 34) → 出力: (1, 30, 1) 各フレームの確率
    │   │  └── フレームごとに確率を平均化
    │   │
    │   │  Step F: シーン抽出
    │   │  ├── 閾値(0.3)でプレー/非プレーを二値化
    │   │  └── 最小シーン長(10フレーム)でフィルタリング
    │   │      → scenes: [(start_frame, end_frame), ...]
    │   │
    ├── 結果返却 ──────────────────────────────────────────────────
    │   │
    │   └── runpod_handler.py に戻る
    │       scenes のフレーム番号を秒に変換
    │       → clips: [{"start_time": 0.5, "end_time": 3.2}, ...]
