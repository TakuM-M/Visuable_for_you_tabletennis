# コンポーネントの名前
Yolopose_Tracker

## 概要
YOLOv11-poseを用いて動画内の人物を検出・追跡する．

## 機能要件
- YOLOv11-poseで人物の骨格データと姿勢を検出する
- ByteTrackを用いて人物を追跡し、tracking IDを付与する
- PlayerTableAnalyzerを使用してプレイヤーと卓球台の位置関係を分析する
- 運動量を計算してプレイヤー候補を絞り込む（審判などの静止人物を除外）
- tracking継続性と卓球台との関係から最終的なプレイヤーを確定する

## 入力
- フレーム画像（numpy.ndarray）
- frame_idx: フレーム番号

## 出力
- 画面内にいる全人間のバウンディングボックス情報のリスト（PersonTrack）

## 処理フロー

### detect_and_track() - フレーム単位の検出
1. フレーム画像と卓球台情報を受け取る
2. YOLOv11-pose + ByteTrackで人物検出・追跡
3. PersonTrackリストを生成して返す


## レビュー指摘事項
- []

## その他
- ソースコード: src/detection/Yolopose_tracker.py
- データクラス: src/detection/data_classes.py (PersonTrack)
