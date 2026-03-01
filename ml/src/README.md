# src/ ディレクトリ構造

卓球動画解析・モデル作成用データ収集プロジェクトのソースコード

## ディレクトリ構成

```
src/
├── data/                     # データ収集・処理モジュール
│   ├── collectors/           # データ収集器
│   │   └── player_detector.py    # プレイヤー検出・トラッキング
│   ├── processors/           # データ処理
│   │   ├── pose_normalizer.py    # 骨格データ正規化
│   │   └── filter_pose_data.py   # トラッキングIDフィルター
│   └── exporters/            # データエクスポート（今後拡張予定）
│
├── detection/                # 検出・トラッキング
│   ├── yolo_tracker.py           # YOLOv11 Poseトラッカー
│   ├── player_filter.py          # 選手フィルタリング
│   ├── table_detector.py         # 卓球台検出
│   └── tracking_exporter.py      # トラッキング結果CSV出力
│
├── visualization/            # 可視化
│   └── visualize_normalized_pose.py  # 正規化骨格データ可視化
│
├── utils/                    # 共通ユーティリティ
│   └── video_loader.py           # 動画読み込み
│
├── scripts/                  # 実行スクリプト（今後拡張予定）
│
└── main.py                   # メインスクリプト

```

## モジュール概要

### 1. data/ - データ収集・処理

#### collectors/
動画から骨格データを収集するモジュール

- **player_detector.py**: 画面中央のプレイヤーを検出・トラッキングし、姿勢データを抽出

```bash
# 実行例
python ./src/data/collectors/player_detector.py \
    -i data/raw/sample_video.MOV \
    -o output/result_video.mp4 \
    --csv output/pose_data.csv \
    --conf 0.3 \
    --center-ratio 0.60
```

#### processors/
収集したデータを処理・変換するモジュール

- **pose_normalizer.py**: 骨格データを正規化（腰中心を原点、体幹長でスケール正規化）
- **filter_pose_data.py**: 特定のトラッキングIDのデータを抽出

```bash
# 統計情報を表示
python src/data/processors/filter_pose_data.py -i output/pose_data.csv --stats

# 特定IDのみ抽出
python src/data/processors/filter_pose_data.py -i output/pose_data.csv -o output/player3.csv --ids 3
```

### 2. detection/ - 検出・トラッキング

YOLOベースの人物検出・トラッキング機能

- **yolo_tracker.py**: YOLOv11-Poseを使った姿勢推定とトラッキング
- **player_filter.py**: 手前選手・相手選手の分類
- **table_detector.py**: 卓球台領域の検出
- **tracking_exporter.py**: トラッキング結果のCSV出力

### 3. visualization/ - 可視化

データ解析・検証用の可視化ツール

- **visualize_normalized_pose.py**: 正規化された骨格データを可視化

```bash
python src/visualization/visualize_normalized_pose.py \
    -i output/pose_data.csv \
    -n 5 \
    --track-id 3
```

### 4. utils/ - ユーティリティ

共通の補助機能

- **video_loader.py**: 動画ファイル読み込み用のラッパークラス

## データフロー

```
1. 動画入力
   ↓
2. data/collectors/player_detector.py
   → プレイヤー検出・トラッキング
   → 骨格データ抽出（正規化オプション付き）
   ↓
3. CSV出力
   ↓
4. data/processors/filter_pose_data.py
   → 必要なトラッキングIDのみ抽出
   ↓
5. visualization/visualize_normalized_pose.py
   → データの可視化・検証
   ↓
6. 機械学習モデルの訓練データとして使用（今後）
```

## インポート方法

新しいディレクトリ構造では、`src.`プレフィックスを使用します：

```python
# 検出モジュール
from src.detection.yolo_tracker import YOLOPoseTracker, PersonTrack

# データ処理モジュール
from src.data.processors.pose_normalizer import PoseNormalizer
from src.data.collectors.player_detector import CenterPlayerDetector

# 可視化モジュール
from src.visualization.visualize_normalized_pose import visualize_normalized_poses

# ユーティリティ
from src.utils.video_loader import VideoLoader
```

## 今後の拡張予定

- `scripts/`: 実行スクリプトの追加
- `data/exporters/`: 多様なデータフォーマット出力（JSON, Parquet等）
- `models/`: 機械学習モデル定義
- `features/`: 特徴量エンジニアリング
- `training/`: モデル訓練スクリプト

## 設計思想

### 責務の明確化
- **data/**: データ収集と処理に特化
- **detection/**: 検出・トラッキングのコアロジック
- **visualization/**: 可視化専用
- **utils/**: 汎用的な補助機能

### スケーラビリティ
- モジュール単位での拡張が容易
- 新しい処理器やコレクターの追加が簡単
- 将来的な機械学習パイプラインへの統合を想定

### 保守性
- 明確なディレクトリ構造
- `__init__.py`による適切なモジュール公開
- ドキュメント化された実行例
