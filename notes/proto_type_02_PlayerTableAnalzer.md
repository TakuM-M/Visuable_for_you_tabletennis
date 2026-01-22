# コンポーネントの名前
プレイヤー-卓球台関係分析コンポーネント (PlayerTableAnalyzer)

## 概要
プレイヤーのバウンディングボックスと卓球台情報から、プレイヤーと卓球台の位置関係（前後、左右、距離）を分析するコンポーネント

## 機能要件
- プレイヤーの位置（near/far または left/right）を判定する
- 画角に応じた適応的な位置判定を行う
- 卓球台からの正規化距離を計算する
- プレイヤーがプレイエリア内にいるかを判定する
- 左右の判定（エンドライン画角時）を行う

## 入力
- player_bbox: プレイヤーのバウンディングボックス (x1, y1, x2, y2)
- track_id: プレイヤーのtracking ID
- table_info: TableInfo（卓球台情報とカメラアングル）

## 出力
- PlayerTableRelation
  - track_id: プレイヤーのtracking ID
  - position: 位置 ("near", "far", "left", "right")
  - side: 左右 ("left" or "right" or None)
  - distance_normalized: 卓球台からの正規化距離 (0.0-1.0以上)
  - is_in_play_area: プレイエリア内にいるか (bool)
  - camera_angle: カメラアングル (CameraAngle)

## 処理フロー
1. プレイヤーのバウンディングボックスと卓球台情報を受け取る
2. プレイヤー中心座標を計算
3. カメラアングルに応じた位置判定
   - サイドライン: 左右で判定 (left/right)
   - エンドライン/斜め上: 前後で判定 (near/far)
4. 左右判定（エンドライン画角時のみ）
5. 卓球台からの正規化距離を計算
6. プレイエリア内判定
7. PlayerTableRelationを生成して返す

## 画角別の判定ロジック

### サイドライン画角
```
- position判定: プレイヤー中心x座標 vs 卓球台中央x座標
  - x < center_x → "left"
  - x >= center_x → "right"
- side判定: None（前後の概念と同じため）
```

### エンドライン / 斜め上画角
```
- position判定: プレイヤー中心y座標 vs 前後境界y座標
  - y > boundary_y → "near" (手前)
  - y <= boundary_y → "far" (奥)
- side判定: プレイヤー中心x座標 vs 卓球台中央x座標
  - x < center_x → "left"
  - x >= center_x → "right"
```

## 距離計算
- 卓球台バウンディングボックスからの最短距離を計算
- 卓球台の対角線長で正規化（画角に依存しない相対距離）
- 0.0 = 卓球台に接触、1.0 = 対角線長分離れている

## プレイエリア判定
- TableInfoから取得したnear_area、far_areaのいずれかにプレイヤー中心が含まれるか
- margin_ratioで調整可能（デフォルト: 0.4 = 卓球台サイズの40%の余白）

## テストケース
- 正常系: エンドライン画角で手前側プレイヤーが正しく"near"と判定されることを確認する
- 正常系: エンドライン画角で奥側プレイヤーが正しく"far"と判定されることを確認する
- 正常系: サイドライン画角で左側プレイヤーが正しく"left"と判定されることを確認する
- 正常系: サイドライン画角で右側プレイヤーが正しく"right"と判定されることを確認する
- 正常系: 卓球台に近いプレイヤーの正規化距離が小さい値（例: < 0.3）になることを確認する
- 正常系: プレイエリア内のプレイヤーがis_in_play_area=Trueと判定されることを確認する
- 正常系: プレイエリア外のプレイヤー（審判など）がis_in_play_area=Falseと判定されることを確認する
- 境界値: 前後境界線上のプレイヤーが一貫して判定されることを確認する

## パラメータ調整
```python
# プレイエリアの余白比率
margin_ratio: float = 0.4  # 卓球台サイズに対する比率

# 距離判定の閾値（用途に応じて調整）
NEAR_TABLE_THRESHOLD = 0.5  # 正規化距離0.5以下で「卓球台に近い」
```

## レビュー指摘事項
- []

## その他
- ソースコード: src/analysis/player_table_analyzer.py
- データクラス: src/analysis/data_classes.py (PlayerTableRelation)
- 依存: src/detection/data_classes.py (TableInfo, CameraAngle)
- このコンポーネントは状態を持たない（純粋関数的）
- 複数のプレイヤーに対して繰り返し呼び出すことを想定
