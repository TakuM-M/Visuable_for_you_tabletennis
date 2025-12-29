# Legacy MediaPipe-based Implementation

このディレクトリには、MediaPipeを使用した旧実装が保存されています。

## 概要
- **姿勢検出**: MediaPipe Holisticによる姿勢推定
- **サービス検出**: ルールベースのサービス姿勢判定
- **複数人物対応**: エッジ検出ベースの人物領域検出

## ファイル構成
- `pose_detector.py`: MediaPipe姿勢検出
- `service_detector.py`: サービス姿勢判定ロジック
- `multi_person_detector.py`: 複数人物検出と選手選別
- `motion_detector.py`: 動き検出

## 新実装への移行
新しい実装ではYOLOv11-Poseを使用したトラッキングベースのアプローチに変更されています。
このコードは参考・比較用に保存されています。

## 移行日
2025-12-29
