# データ拡張パイプライン使用ガイド

## 概要

`PoseAugmentationPipeline` は、正規化された骨格データ（CSV形式）に対してデータ拡張を適用するパイプラインです。

## 主な機能

### データ拡張手法

1. **左右反転 (Horizontal Flip)** - 右利き/左利きの変換
2. **ガウシアンノイズ (Gaussian Noise)** - 検出誤差のシミュレーション
3. **回転 (Rotation)** - 体の向きの変化
4. **スケーリング (Scaling)** - 体の大きさの変化
5. **関節ドロップアウト (Keypoint Dropout)** - オクルージョンのシミュレーション
6. **時間的ジッター (Temporal Jitter)** - 微小な時間変動（時系列データ用）
7. **時間スケーリング (Temporal Scaling)** - 動作速度の変更（時系列データ用）

## 基本的な使い方

### 1. デフォルト設定での使用

```python
from src.pipelines import PoseAugmentationPipeline

# デフォルト設定でパイプラインを作成
pipeline = PoseAugmentationPipeline.create_default(
    augmentation_factor=5,  # 元データの5倍に拡張
    random_seed=42          # 再現性のためのシード
)

# データ拡張を実行
results = pipeline.augment_csv(
    input_csv="output/player_pose_data.csv",
    output_csv="output/augmented_pose_data.csv"
)

print(f"元データ: {results['original_samples']} サンプル")
print(f"拡張後: {results['augmented_samples']} サンプル")
```

### 2. カスタム設定での使用

```python
from src.pipelines import (
    PoseAugmentationPipeline,
    AugmentationPipelineConfig,
    AugmentationConfig
)

# カスタム拡張設定を作成
augmentation_config = AugmentationConfig(
    # 左右反転
    horizontal_flip=True,
    horizontal_flip_prob=0.5,  # 50%の確率で反転

    # ガウシアンノイズ
    add_noise=True,
    noise_std=0.03,  # ノイズの標準偏差

    # 回転
    rotation=True,
    rotation_range=20.0,  # ±20度の範囲で回転

    # スケーリング
    scaling=True,
    scale_range=(0.9, 1.1),  # 0.9~1.1倍にスケーリング

    # 関節ドロップアウト
    keypoint_dropout=True,
    dropout_prob=0.15,  # 15%の確率で各関節をドロップ

    # ランダムシード
    random_seed=42
)

# パイプライン設定を作成
pipeline_config = AugmentationPipelineConfig(
    augmentation=augmentation_config,
    augmentation_factor=10,  # 10倍に拡張
    preserve_original=True,  # 元データも保持
    save_metadata=True,      # メタデータを保存
    show_progress=True       # プログレスバー表示
)

# パイプラインを作成
pipeline = PoseAugmentationPipeline(pipeline_config)

# 実行
results = pipeline.augment_csv(
    input_csv="input.csv",
    output_csv="output.csv",
    output_metadata="output_metadata.json"  # オプション
)
```

### 3. 軽量な拡張（高速処理用）

```python
from src.pipelines import (
    PoseAugmentationPipeline,
    AugmentationPipelineConfig,
    AugmentationConfig
)

# 軽量な拡張設定
augmentation_config = AugmentationConfig(
    horizontal_flip=True,
    horizontal_flip_prob=0.5,
    add_noise=True,
    noise_std=0.01,  # ノイズを小さく
    random_seed=42
)

pipeline_config = AugmentationPipelineConfig(
    augmentation=augmentation_config,
    augmentation_factor=3,  # 3倍に留める
    preserve_original=True
)

pipeline = PoseAugmentationPipeline(pipeline_config)
results = pipeline.augment_csv("input.csv", "output.csv")
```

### 4. 強力な拡張（多様なデータ生成用）

```python
# すべての拡張手法を有効化
augmentation_config = AugmentationConfig(
    horizontal_flip=True,
    horizontal_flip_prob=0.5,
    add_noise=True,
    noise_std=0.03,
    rotation=True,
    rotation_range=25.0,
    scaling=True,
    scale_range=(0.85, 1.15),
    keypoint_dropout=True,
    dropout_prob=0.2,
    random_seed=42
)

pipeline_config = AugmentationPipelineConfig(
    augmentation=augmentation_config,
    augmentation_factor=15,  # 15倍に拡張
    preserve_original=True
)

pipeline = PoseAugmentationPipeline(pipeline_config)
results = pipeline.augment_csv("input.csv", "output.csv")
```

## 入力データフォーマット

入力CSVは以下のカラムを持つ必要があります:

### 必須カラム
- `frame`: フレーム番号
- `timestamp`: タイムスタンプ
- `track_id`: トラッキングID

### キーポイントカラム（COCO形式の17キーポイント）

各キーポイントについて `{keypoint_name}_norm_x` と `{keypoint_name}_norm_y` が必要:

```
nose_norm_x, nose_norm_y
left_eye_norm_x, left_eye_norm_y
right_eye_norm_x, right_eye_norm_y
left_ear_norm_x, left_ear_norm_y
right_ear_norm_x, right_ear_norm_y
left_shoulder_norm_x, left_shoulder_norm_y
right_shoulder_norm_x, right_shoulder_norm_y
left_elbow_norm_x, left_elbow_norm_y
right_elbow_norm_x, right_elbow_norm_y
left_wrist_norm_x, left_wrist_norm_y
right_wrist_norm_x, right_wrist_norm_y
left_hip_norm_x, left_hip_norm_y
right_hip_norm_x, right_hip_norm_y
left_knee_norm_x, left_knee_norm_y
right_knee_norm_x, right_knee_norm_y
left_ankle_norm_x, left_ankle_norm_y
right_ankle_norm_x, right_ankle_norm_y
```

## 出力データフォーマット

出力CSVには元のカラムに加えて以下が追加されます:

- `augmentation_id`: 拡張ID（0は元データ、1以降は拡張データ）

## メタデータ

`save_metadata=True` の場合、以下の情報を含むJSONファイルが生成されます:

```json
{
  "config": {
    "augmentation_factor": 5,
    "preserve_original": true,
    "augmentation": {
      "horizontal_flip": true,
      "noise_std": 0.02,
      "rotation_range": 15.0,
      ...
    }
  },
  "statistics": {
    "original_samples": 1000,
    "augmented_samples": 5000,
    "augmentation_factor_actual": 5.0
  }
}
```

## エラーハンドリング

```python
from src.pipelines import (
    PoseAugmentationPipeline,
    DataInputError,
    AugmentationError,
    ExportError
)

pipeline = PoseAugmentationPipeline.create_default(augmentation_factor=5)

try:
    results = pipeline.augment_csv("input.csv", "output.csv")
except DataInputError as e:
    print(f"データ読み込みエラー: {e.input_path}")
    print(f"理由: {e.reason}")
except AugmentationError as e:
    print(f"拡張処理エラー: {e}")
    if e.sample_index >= 0:
        print(f"エラー発生サンプル: {e.sample_index}")
except ExportError as e:
    print(f"出力エラー: {e.output_path}")
    print(f"理由: {e.reason}")
```

## 設定パラメータ詳細

### AugmentationConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `horizontal_flip` | bool | False | 左右反転を有効化 |
| `horizontal_flip_prob` | float | 0.5 | 反転確率 (0.0-1.0) |
| `add_noise` | bool | False | ノイズ付加を有効化 |
| `noise_std` | float | 0.02 | ノイズの標準偏差 |
| `rotation` | bool | False | 回転を有効化 |
| `rotation_range` | float | 15.0 | 回転角度範囲（度） |
| `scaling` | bool | False | スケーリングを有効化 |
| `scale_range` | tuple | (0.9, 1.1) | スケール範囲 |
| `keypoint_dropout` | bool | False | ドロップアウトを有効化 |
| `dropout_prob` | float | 0.1 | ドロップアウト確率 |
| `random_seed` | int\|None | None | ランダムシード |

### AugmentationPipelineConfig

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `augmentation` | AugmentationConfig | - | 拡張設定 |
| `augmentation_factor` | int | 5 | 拡張倍率 |
| `preserve_original` | bool | True | 元データを保持 |
| `is_sequence` | bool | False | 時系列データとして処理 |
| `sequence_length` | int\|None | None | シーケンス長 |
| `save_metadata` | bool | True | メタデータを保存 |
| `show_progress` | bool | True | プログレスバー表示 |

## 統合例: PlayerPoseExporter → PoseAugmentationPipeline

```python
from src.pipelines import PlayerPoseExporter, PoseAugmentationPipeline

# 1. 動画から姿勢データを抽出
exporter = PlayerPoseExporter.create_default(
    table_model_path="models/table_detection/best.pt",
    pose_model_path="models/pose/yolov8n-pose.pt"
)

exporter.process_video(
    input_video="match.mp4",
    output_video="output_match.mp4",
    csv_output="pose_data.csv"
)

# 2. 抽出したデータを拡張
pipeline = PoseAugmentationPipeline.create_default(
    augmentation_factor=5,
    random_seed=42
)

results = pipeline.augment_csv(
    input_csv="pose_data.csv",
    output_csv="augmented_pose_data.csv"
)

print(f"学習用データ準備完了: {results['augmented_samples']} サンプル")
```

## ベストプラクティス

### 1. 再現性の確保
```python
# 常にrandom_seedを設定
pipeline = PoseAugmentationPipeline.create_default(random_seed=42)
```

### 2. 段階的な拡張
```python
# 最初は軽い拡張でテスト
augmentation_config = AugmentationConfig(
    horizontal_flip=True,
    add_noise=True,
    noise_std=0.01,
    random_seed=42
)
```

### 3. メタデータの保存
```python
# 拡張設定を追跡可能にする
pipeline_config = AugmentationPipelineConfig(
    augmentation=augmentation_config,
    save_metadata=True  # 必ず有効化
)
```

### 4. 適切な拡張倍率の選択
- **小規模データセット（< 1000サンプル）**: 10-15倍
- **中規模データセット（1000-10000サンプル）**: 5-10倍
- **大規模データセット（> 10000サンプル）**: 3-5倍

## トラブルシューティング

### Q: メモリ不足エラーが発生する
A: `augmentation_factor` を小さくするか、データを分割して処理してください。

### Q: 拡張後のデータが不自然
A: `noise_std`、`rotation_range`、`scale_range` を小さくしてください。

### Q: 処理が遅い
A: `show_progress=False` にするか、拡張手法を減らしてください。

## 次のステップ

データ拡張後は、拡張されたデータを使ってLSTMモデルを学習します:

```bash
python src/training/train_play_classifier.py \
  --train-csv augmented_pose_data.csv \
  --train-labels labels.csv \
  --epochs 50 \
  --batch-size 32
```

詳細は `TrainingPipeline` のドキュメントを参照してください。
