# 骨格データ正規化設計

## 概要

骨格データは**生座標と正規化座標の両方を保持**する設計とします。

## データフロー

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: データ獲得（PlayerPoseExporter）                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  動画 → YOLOPose → PersonTrack                                   │
│                     ├── keypoints (生座標 17×3)                 │
│                     └── (正規化座標は未設定)                     │
│                                                                  │
│  TrackingExporter.normalize_poses() 実行                         │
│                     ↓                                            │
│  PersonTrack に追加:                                             │
│    ├── normalized_keypoints (17×2) ← 新規作成                  │
│    ├── hip_center (x, y)                                        │
│    ├── scale_factor (腰幅)                                       │
│    └── is_normalized_valid (True/False)                         │
│                                                                  │
│  TrackingExporter.export_csv()                                   │
│                     ↓                                            │
│  CSV出力: 生座標 + 正規化座標 + メタデータ                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: データ拡張（PoseAugmentationPipeline）                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CSV読み込み                                                     │
│   ├── 生座標カラム (*_x, *_y, *_conf)                          │
│   ├── 正規化座標カラム (*_norm_x, *_norm_y)                     │
│   └── メタデータ (hip_center_*, scale_factor)                   │
│                                                                  │
│  データ拡張処理                                                  │
│   ├── 正規化座標のみを拡張 (*_norm_x, *_norm_y)                │
│   └── 生座標は変更しない (*_x, *_y は保持)                     │
│                                                                  │
│  CSV出力: 生座標 + 拡張済み正規化座標 + メタデータ              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: 学習・推論（LSTM）                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  学習時:                                                         │
│    CSV読み込み → 正規化座標のみ使用 (*_norm_x, *_norm_y)       │
│                                                                  │
│  推論時（End-to-End）:                                           │
│    動画 → PlayerPoseExporter.process_video()                     │
│         → TrackingExporter.normalize_poses()                     │
│         → メモリ上の正規化座標を直接取得                        │
│         → LSTM推論                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## CSV構造

### 完全なCSV構造（normalize_poses実行後）

```csv
# 基本情報
track_id,frame,timestamp,role,confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2,

# 生座標（17キーポイント × 3カラム = 51カラム）
nose_x,nose_y,nose_conf,
left_eye_x,left_eye_y,left_eye_conf,
...,

# 正規化座標（17キーポイント × 2カラム = 34カラム）
nose_norm_x,nose_norm_y,
left_eye_norm_x,left_eye_norm_y,
...,

# 正規化メタデータ（4カラム）
hip_center_x,hip_center_y,scale_factor,is_normalized_valid
```

**総カラム数:**
- 基本情報: 9カラム
- 生座標: 51カラム (17 × 3)
- 正規化座標: 34カラム (17 × 2)
- メタデータ: 4カラム
- **合計: 98カラム**

---

## データクラス設計

### PersonTrack (src/core/data_classes.py)

```python
@dataclass
class PersonTrack:
    """トラッキングされた人物の情報"""
    # 基本情報
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float

    # 生座標（常に保持）
    keypoints: np.ndarray  # (17, 3) [x, y, confidence]

    # 正規化データ（normalize_poses実行後に設定）
    normalized_keypoints: Optional[np.ndarray] = None  # (17, 2) [norm_x, norm_y]
    hip_center: Optional[Tuple[float, float]] = None   # (x, y) 生座標系
    scale_factor: Optional[float] = None               # 腰幅（ピクセル）
    is_normalized_valid: bool = False                  # 正規化成功フラグ
```

---

## 正規化手法

### 腰中心基準の正規化

```python
# 1. 腰の中心を計算
hip_center = (left_hip + right_hip) / 2

# 2. 腰幅を計算（スケール係数）
scale_factor = distance(left_hip, right_hip)

# 3. 各キーポイントを正規化
for keypoint in keypoints:
    norm_x = (x - hip_center_x) / scale_factor
    norm_y = (y - hip_center_y) / scale_factor
```

### 逆変換（正規化座標 → 生座標）

```python
def denormalize_keypoint(norm_x, norm_y, hip_center, scale_factor):
    """正規化座標を生座標に戻す"""
    x = norm_x * scale_factor + hip_center[0]
    y = norm_y * scale_factor + hip_center[1]
    return x, y
```

---

## 各フェーズでの使用座標

| フェーズ | 使用座標 | 理由 |
|---------|---------|------|
| **データ獲得**<br>(PlayerPoseExporter) | 生座標 → 正規化座標を追加 | YOLOPoseの出力は生座標 |
| **可視化**<br>(動画オーバーレイ) | 生座標 | フレームに直接描画するため |
| **データ拡張**<br>(PoseAugmentationPipeline) | 正規化座標のみ | スケール不変な拡張が必要 |
| **LSTM学習** | 正規化座標のみ | 動画間でスケールを統一 |
| **LSTM推論** | 正規化座標のみ | 学習時と同じ座標系 |

---

## メモリ vs CSV

### メモリ上（TrackingExporter内）

```python
# PersonTrackオブジェクトとして両方を保持
person.keypoints           # 生座標 (17, 3)
person.normalized_keypoints # 正規化座標 (17, 2)
person.hip_center          # メタデータ
person.scale_factor        # メタデータ
```

**メリット:**
- End-to-End推論時にCSV経由不要
- メモリ上で直接LSTM推論可能
- 可視化と推論の両方に対応

### CSV出力

```python
# 両方のカラムを出力
*_x, *_y, *_conf           # 生座標
*_norm_x, *_norm_y         # 正規化座標
hip_center_*, scale_factor # メタデータ
```

**メリット:**
- データの永続化
- 後から別の正規化手法を試せる
- 可視化ツールで生座標を使用可能

---

## 実装の流れ

### 1. データ獲得時（PlayerPoseExporter）

```python
from src.pipelines import PlayerPoseExporter

# パイプライン作成
exporter = PlayerPoseExporter.create_default(
    table_model_path="models/table_detection/best.pt",
    pose_model_path="models/pose/yolov8n-pose.pt"
)

# 動画処理（内部でnormalize_posesが自動実行される）
results = exporter.process_video(
    input_video="match.mp4",
    output_video="output.mp4",
    csv_output="pose_data.csv"  # 生座標 + 正規化座標
)
```

**PlayerPoseExporter内部:**
```python
# _export_results内で自動的に正規化を実行
self.tracking_exporter.normalize_poses()  # 正規化座標を追加
self.tracking_exporter.export_csv(...)    # 両方を出力
```

### 2. データ拡張時（PoseAugmentationPipeline）

```python
from src.pipelines import PoseAugmentationPipeline

pipeline = PoseAugmentationPipeline.create_default(
    augmentation_factor=5,
    random_seed=42
)

# 正規化座標のみを拡張
results = pipeline.augment_csv(
    input_csv="pose_data.csv",        # 生座標 + 正規化座標
    output_csv="augmented_data.csv"   # 生座標 + 拡張済み正規化座標
)
```

**内部処理:**
- 生座標 (`*_x`, `*_y`) → **変更なし**
- 正規化座標 (`*_norm_x`, `*_norm_y`) → **拡張適用**
- メタデータ → **保持**

### 3. LSTM学習時

```python
from src.datasets.dataset import PoseSequenceDataset

# 正規化座標のみを読み込み
dataset = PoseSequenceDataset(
    csv_path="augmented_data.csv",
    label_path="labels.csv",
    sequence_length=30
)

# 内部で *_norm_x, *_norm_y のみを使用
```

### 4. End-to-End推論時（CSV経由なし）

```python
from src.pipelines import PlayerPoseExporter

exporter = PlayerPoseExporter.create_default(...)
exporter.process_video("new_match.mp4", "output.mp4", "temp.csv")

# メモリ上の正規化データを直接取得
normalized_data = exporter.tracking_exporter.get_pose_data_for_dataset()

# LSTM推論（CSV経由なし）
predictions = lstm_model.predict(normalized_data)
```

---

## ベストプラクティス

### ✅ DO

1. **常に正規化を実行**
   ```python
   # PlayerPoseExporterは自動的に正規化を実行
   exporter.process_video(...)
   ```

2. **データ拡張は正規化座標に適用**
   ```python
   # PoseAugmentationPipelineは正規化座標のみを拡張
   pipeline.augment_csv(...)
   ```

3. **学習・推論は正規化座標を使用**
   ```python
   # DatasetやLSTMは正規化座標のみを使用
   dataset = PoseSequenceDataset(csv_path="...", ...)
   ```

4. **可視化は生座標を使用**
   ```python
   # フレームに描画する場合は生座標
   cv2.circle(frame, (int(nose_x), int(nose_y)), ...)
   ```

### ❌ DON'T

1. **生座標を拡張しない**
   ```python
   # NG: 生座標は動画ごとにスケールが異なるため無意味
   df['nose_x'] += noise  # Don't do this!
   ```

2. **学習時に生座標を使用しない**
   ```python
   # NG: 動画ごとにスケールが異なるため学習できない
   features = df[['nose_x', 'nose_y', ...]].values  # Don't do this!
   ```

3. **正規化前のデータで学習しない**
   ```python
   # NG: 必ず正規化後のデータを使用
   # normalize_poses() を実行してから export_csv()
   ```

---

## トラブルシューティング

### Q: CSVに正規化座標がない

**A:** `PlayerPoseExporter`で`normalize_poses()`が実行されていません。

```python
# 確認方法
df = pd.read_csv("pose_data.csv")
print("nose_norm_x" in df.columns)  # Trueになるべき

# 修正方法: process_video内で自動実行されるはず
# 手動で実行する場合:
exporter.tracking_exporter.normalize_poses()
exporter.tracking_exporter.export_csv("pose_data.csv", {...})
```

### Q: データ拡張時にエラー

**A:** 正規化座標カラムが見つかりません。

```python
# エラーメッセージ:
# DataInputError: 正規化座標カラム（*_norm_x, *_norm_y）が見つかりません

# 原因: normalize_poses()が実行されていないCSVを使用
# 解決: PlayerPoseExporterで正しくエクスポートされたCSVを使用
```

### Q: 逆変換が必要

**A:** `hip_center`と`scale_factor`から計算可能。

```python
def denormalize(df, kp_name):
    """正規化座標 → 生座標"""
    df[f'{kp_name}_denorm_x'] = (
        df[f'{kp_name}_norm_x'] * df['scale_factor'] + df['hip_center_x']
    )
    df[f'{kp_name}_denorm_y'] = (
        df[f'{kp_name}_norm_y'] * df['scale_factor'] + df['hip_center_y']
    )
    return df
```

---

## まとめ

### 設計のポイント

1. ✅ **生座標と正規化座標の両方を保持** - 柔軟性が高い
2. ✅ **`PersonTrack`に両方のフィールド** - メモリ上で両方利用可能
3. ✅ **CSVに両方のカラム** - データの完全性が保たれる
4. ✅ **用途に応じて使い分け** - 可視化は生座標、学習は正規化座標
5. ✅ **正規化メタデータを保存** - 逆変換が可能

### データフローの一貫性

```
動画 → [生座標] → 正規化 → [生座標 + 正規化座標]
                              ↓
                         データ拡張 → [生座標 + 拡張済み正規化座標]
                              ↓
                         LSTM学習 → [正規化座標のみ使用]
                              ↓
                         LSTM推論 → [正規化座標のみ使用]
```

**全フェーズで一貫した正規化手法を使用することで、学習と推論の整合性が保たれます。**
