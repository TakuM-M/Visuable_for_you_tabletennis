# YOLOv11-Pose ベーストラッキングシステム設計

## 概要
卓球動画から手前選手と相手選手をトラッキングし、ID付きで座標をCSV出力するシステム

## アーキテクチャ

```
入力動画
  ↓
[1] 卓球台検出 (table_detector.py)
  ↓ 卓球台の位置情報
[2] YOLOv11-Pose トラッキング (yolo_tracker.py)
  ↓ 全人物のID、バウンディングボックス、キーポイント
[3] 選手フィルタリング (player_filter.py)
  ↓ 手前選手、相手選手のみを抽出
[4] CSV出力 (tracking_exporter.py)
  ↓
出力CSV (ID, フレーム, キーポイント座標)
```

## モジュール詳細

### 1. table_detector.py - 卓球台検出モジュール

**責務**: 動画内の卓球台位置を検出・固定

**機能**:
- 色ベース検出（緑・青の卓球台）
- エッジ検出による台面抽出
- 複数フレームでの位置安定化
- 台面の4隅座標を返す

**クラス**:
```python
class TableDetector:
    def detect_table(self, frame) -> Optional[TableRegion]
    def get_stable_table_region(self, video) -> TableRegion
```

**出力**:
```python
@dataclass
class TableRegion:
    corners: np.ndarray  # 4隅の座標 (4, 2)
    center: Tuple[float, float]
    width: float
    height: float
```

---

### 2. yolo_tracker.py - YOLOv11-Poseトラッキング

**責務**: 動画内の全人物をID付きでトラッキング

**機能**:
- YOLOv11-Poseによる人物検出
- ByteTrackベースのID管理
- 姿勢キーポイント（17点）の取得
- トラッキングIDの安定化

**クラス**:
```python
class YOLOPoseTracker:
    def __init__(self, model_path: str = "yolo11n-pose.pt")
    def track_frame(self, frame) -> List[PersonTrack]
    def reset_tracker(self)
```

**出力**:
```python
@dataclass
class PersonTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    keypoints: np.ndarray  # (17, 3) [x, y, confidence]
    confidence: float
```

---

### 3. player_filter.py - 選手フィルタリング

**責務**: トラッキング結果から手前選手・相手選手を特定

**機能**:
- 卓球台位置を基準に選手を分類
- 手前選手: 台より下側（カメラに近い）
- 相手選手: 台より上側（カメラから遠い）
- 審判・コーチの除外

**クラス**:
```python
class PlayerFilter:
    def __init__(self, table_region: TableRegion)
    def classify_players(self, persons: List[PersonTrack]) -> PlayerClassification
    def assign_player_roles(self, history: List[PlayerClassification]) -> Dict[int, str]
```

**出力**:
```python
@dataclass
class PlayerClassification:
    near_player: Optional[PersonTrack]  # 手前選手
    far_player: Optional[PersonTrack]   # 相手選手
    others: List[PersonTrack]           # その他（審判等）
```

---

### 4. tracking_exporter.py - CSV出力

**責務**: トラッキング結果をCSV形式で保存

**機能**:
- フレームごとのキーポイント座標を記録
- ID、フレーム番号、タイムスタンプ付き
- キーポイント名のヘッダー付き

**クラス**:
```python
class TrackingExporter:
    def add_frame(self, frame_num: int, timestamp: float, persons: List[PersonTrack])
    def export_csv(self, output_path: str)
    def export_player_csv(self, output_path: str, player_tracks: Dict[int, str])
```

**CSV形式**:
```csv
track_id,frame,timestamp,role,nose_x,nose_y,nose_conf,left_eye_x,...
1,0,0.0,near_player,320,240,0.95,315,235,0.92,...
2,0,0.0,far_player,640,180,0.88,635,175,0.85,...
```

---

## データフロー

```python
# メインスクリプト例
video = VideoLoader("input.mp4")
table_detector = TableDetector()
tracker = YOLOPoseTracker()
exporter = TrackingExporter()

# 1. 卓球台検出（最初の数フレームで安定化）
table_region = table_detector.get_stable_table_region(video)

# 2. 選手フィルタの初期化
player_filter = PlayerFilter(table_region)

# 3. フレームごとの処理
for frame_num, frame in enumerate(video):
    # YOLO トラッキング
    persons = tracker.track_frame(frame)

    # 選手分類
    classification = player_filter.classify_players(persons)

    # CSV記録
    exporter.add_frame(frame_num, timestamp, persons)

# 4. 選手役割の決定（全フレーム処理後）
player_roles = player_filter.assign_player_roles(history)

# 5. CSV出力
exporter.export_player_csv("output.csv", player_roles)
```

---

## 技術スタック

- **YOLOv11-Pose**: Ultralytics (yolo11n-pose.pt)
- **トラッキング**: ByteTrack (YOLOに内蔵)
- **画像処理**: OpenCV
- **データ処理**: NumPy, Pandas

---

## キーポイント定義 (COCO 17点)

```
0: nose
1: left_eye
2: right_eye
3: left_ear
4: right_ear
5: left_shoulder
6: right_shoulder
7: left_elbow
8: right_elbow
9: left_wrist
10: right_wrist
11: left_hip
12: right_hip
13: left_knee
14: right_knee
15: left_ankle
16: right_ankle
```

---

## 実装の優先順位

1. **table_detector.py** - 卓球台検出
2. **yolo_tracker.py** - YOLOトラッキング
3. **player_filter.py** - 選手フィルタリング
4. **tracking_exporter.py** - CSV出力
5. **test_yolo_tracking.py** - 統合テストスクリプト

---

## Legacy実装との違い

| 項目 | Legacy (MediaPipe) | 新実装 (YOLOv11) |
|------|-------------------|------------------|
| 姿勢推定 | MediaPipe Holistic | YOLOv11-Pose |
| ID管理 | なし（フレーム単位） | ByteTrack（連続追跡） |
| 複数人対応 | エッジ検出 | YOLO物体検出 |
| 出力 | グラフ・動画 | CSV + 動画 |
| 用途 | サービス検出 | 汎用トラッキング |
