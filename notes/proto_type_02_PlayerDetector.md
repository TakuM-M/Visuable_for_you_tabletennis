# コンポーネントの名前
プレイヤー検出コンポーネント (PlayerDetector)

## 概要
YOLOv11-poseを用いて動画内の人物を検出・追跡し、PlayerTableAnalyzerと連携してプレイヤー候補を絞り込み、最終的なプレイヤーを確定するコンポーネント

## 機能要件
- YOLOv11-poseで人物の骨格データと姿勢を検出する
- ByteTrackを用いて人物を追跡し、tracking IDを付与する
- PlayerTableAnalyzerを使用してプレイヤーと卓球台の位置関係を分析する
- 運動量を計算してプレイヤー候補を絞り込む（審判などの静止人物を除外）
- tracking継続性と卓球台との関係から最終的なプレイヤーを確定する
- 奥側プレイヤーの一時的な検出途切れに対応する（track_buffer活用）

## 入力
- フレーム画像（numpy.ndarray）
- table_info: TableInfo（卓球台情報）
- frame_idx: フレーム番号

## 出力
- Dict[int, PlayerTableRelation]: tracking IDをキーとしたプレイヤーと卓球台の位置関係辞書
- finalize_players()の出力: List[int] - 確定したプレイヤーのtracking IDリスト（シングルス: 2人、ダブルス: 4人）

## 処理フロー

### detect_and_track() - フレーム単位の検出
1. フレーム画像と卓球台情報を受け取る
2. YOLOv11-pose + ByteTrackで人物検出・追跡
3. 各検出人物について:
   - tracking IDを取得
   - バウンディングボックスと骨格キーポイントを取得
   - PlayerTableAnalyzerで卓球台との位置関係を分析
   - 運動量を計算（前フレームとの骨格キーポイント差分）
4. プレイヤー候補辞書に情報を蓄積
5. tracking ID別の位置関係辞書を返す

### finalize_players() - 最終プレイヤー確定
1. 蓄積されたプレイヤー候補情報を分析
2. 各候補のスコアリング:
   - tracking継続時間（長いほど高評価）
   - 総運動量（多いほど高評価）
   - 卓球台付近にいた時間比率（高いほど高評価）
3. 卓球台の前後（またはサイドライン時は左右）で分類
4. 各エリアからスコア上位の人物を選定
5. シングルス: 2人、ダブルス: 4人のtracking IDを返す

## YOLOv11-pose設定

### モデル設定
```python
model: yolov11n-pose.pt または yolov11s-pose.pt
imgsz: 1280  # 1080p動画用に高解像度
conf: 0.3    # 検出閾値（奥側選手対策で緩め）
iou: 0.5
```

### Tracking設定（重要）
```python
tracker: bytetrack.yaml
persist: True  # tracking ID維持

# bytetrack.yaml の設定
tracker_type: bytetrack
track_high_thresh: 0.4
track_low_thresh: 0.2
new_track_thresh: 0.5
track_buffer: 60        # 60フレーム（1fpsで60秒）検出が途切れても追跡継続
match_thresh: 0.7
```

## 運動量計算
```python
# 骨格キーポイント（17点）の移動量
movement = sum(||keypoint_t - keypoint_{t-1}||) for all keypoints
normalized_movement = movement / diagonal_of_frame

# 閾値
MIN_MOVEMENT_THRESHOLD = 0.01  # これ以下は静止とみなす（審判除外）
```

## プレイヤー候補のスコアリング
```python
score = (
    tracking_duration * 0.4 +      # 長く映っている（例: 0-300秒 → 0-1.0）
    total_movement * 0.3 +          # 動いている（正規化運動量の累積）
    near_table_ratio * 0.3          # 卓球台に近い時間比率
)
```

## エリア別選定ロジック

### エンドライン / 斜め上画角
```
1. プレイヤー候補を "near" / "far" で分類
2. 各エリアからスコア上位1人（シングルス）または2人（ダブルス）を選定
```

### サイドライン画角
```
1. プレイヤー候補を "left" / "right" で分類
2. 各エリアからスコア上位1人（シングルス）または2人（ダブルス）を選定
```

## 奥側選手の検出途切れ対策
1. **track_buffer = 60**: 検出が途切れても60秒間tracking継続
2. **エリア別に検出閾値を調整**: 奥側エリアはconf閾値を緩く（実装オプション）
3. **時系列補間**: 検出失敗フレームは前後フレームから補間（後処理）

## テストケース
- 正常系: 2人のプレイヤーが含まれる動画で、正しく2人のtracking IDが返されることを確認する
- 正常系: 手前側と奥側にそれぞれ1人ずつプレイヤーが選定されることを確認する
- 正常系: 審判などの静止人物が除外されることを確認する
- 正常系: tracking IDが動画全体を通じて一貫していることを確認する
- 正常系: 奥側プレイヤーが一時的に手前側プレイヤーに隠れても、tracking IDが維持されることを確認する
- 異常系: プレイヤーが1人しか検出できない場合、検出できた人数分のIDが返されることを確認する
- 境界値: track_bufferの期限切れ後、同じ人物が再検出された場合に新しいtracking IDが付与されることを確認する

## パラメータ調整
```python
# 検出設定
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
IMGSZ = 1280

# Tracking設定
TRACK_BUFFER = 60  # 1fps時は60秒

# 運動量閾値
MIN_MOVEMENT_THRESHOLD = 0.01

# スコアリング重み
WEIGHT_TRACKING_DURATION = 0.4
WEIGHT_TOTAL_MOVEMENT = 0.3
WEIGHT_NEAR_TABLE_RATIO = 0.3

# プレイヤー数
NUM_PLAYERS_SINGLES = 2
NUM_PLAYERS_DOUBLES = 4
```

## データ構造

### PlayerCandidate（内部データ）
```python
@dataclass
class PlayerCandidate:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    positions: List[str]         # ["near", "near", "far", ...]
    keypoints_history: List[np.ndarray]  # 骨格データ履歴
    total_movement: float
    near_table_count: int
    total_frames: int

    @property
    def tracking_duration(self) -> float:
        return (self.last_seen_frame - self.first_seen_frame) / fps

    @property
    def near_table_ratio(self) -> float:
        return self.near_table_count / self.total_frames
```

## レビュー指摘事項
- []

## その他
- ソースコード: src/detection/player_detector.py
- データクラス: src/detection/models.py (PlayerCandidate)
- 依存:
  - YOLOv11-pose (ultralytics)
  - src/analysis/player_table_analyzer.py
  - src/detection/models.py (TableInfo)
- proto_type_01のLSTM判定モデル (src/models/play_classifier.py) と連携して使用することを想定
- 1fpsでのサンプリングを前提とした設計（dense処理は別ステップ）
