# コンポーネントの名前
卓球台検出コンポーネント (TableDetector)

## 概要
卓球の試合映像から中央に近い卓球台を検出し、バウンディングボックスの座標と画角情報を出力するコンポーネント

## 機能要件
- 動画内の卓球台を検出し、バウンディングボックスの座標を取得する
- 画面中央に最も近い卓球台を選定する
- カメラアングル（エンドライン側、サイドライン側、斜め上）を自動推定する
- 卓球台の前後（奥/手前）領域を定義する
- 検出結果をキャッシングして処理を効率化する

## 入力
- フレーム画像（numpy.ndarray）
- フレーム番号（int）
- 強制再検出フラグ（bool, オプション）

## 出力
- TableInfo
  - bbox: バウンディングボックス座標 (x1, y1, x2, y2)
  - confidence: 検出信頼度
  - frame_idx: フレーム番号
  - camera_angle: 推定されたカメラアングル (ENDLINE/SIDELINE/DIAGONAL_TOP/UNKNOWN)
  - プロパティ:
    - center: 卓球台中心座標
    - width, height: 幅と高さ
    - aspect_ratio: アスペクト比
  - メソッド:
    - estimate_camera_angle(): 画角の自動推定
    - get_near_far_boundary(): 前後判定の境界y座標
    - get_near_area(): 手前側プレイエリア
    - get_far_area(): 奥側プレイエリア

## 処理フロー
1. フレーム画像を受け取る
2. キャッシュチェック（有効なら既存のTableInfoを返す）
3. YOLOv11で卓球台を検出
4. 画面中央に最も近い卓球台を選定
5. バウンディングボックスのアスペクト比と位置から画角を推定
6. TableInfoを生成してキャッシュに保存
7. TableInfoを返す

## 画角推定ロジック
- **サイドライン側**: aspect_ratio > 2.5 (非常に横長)
- **エンドライン側**: aspect_ratio < 1.2 (縦長) または 1.2 <= aspect_ratio <= 2.5
- **斜め上**: 画面上部 (vertical_position < 0.4) かつ小サイズ (area_ratio < 0.15)

## 前後判定の境界
- **サイドライン**: 卓球台の中央x座標（左右で判定）
- **エンドライン**: bbox上端 + height * 0.35
- **斜め上**: bbox上端 + height * 0.25

## テストケース
- 正常系: 有効なフレームを入力として与え、卓球台のバウンディングボックス座標とカメラアングルが正しく出力されることを確認する
- 正常系: エンドライン側、サイドライン側、斜め上の各画角で正しく画角推定されることを確認する
- 正常系: キャッシュが有効な間は再検出せず、同じTableInfoが返されることを確認する
- 異常系: 卓球台が検出できなかったフレームの場合、Noneが返されることを確認する
- 異常系: force_detect=Trueの場合、キャッシュを無視して再検出されることを確認する

## パラメータ調整
以下のパラメータは実際の動画で調整が必要:
```python
# 画角判定閾値
SIDELINE_ASPECT_THRESHOLD = 2.5
ENDLINE_ASPECT_THRESHOLD = 1.2
DIAGONAL_TOP_VERTICAL_POS = 0.4
DIAGONAL_TOP_AREA_RATIO = 0.15

# 前後境界位置
ENDLINE_BOUNDARY_RATIO = 0.35
DIAGONAL_TOP_BOUNDARY_RATIO = 0.25

# キャッシュ有効期間
CACHE_VALID_FRAMES = 100  # 0.1fps × 100 = 10秒
```

## レビュー指摘事項
- []

## その他
- ソースコード: src/detection/table_detector.py
- データクラス: src/detection/models.py (TableInfo, CameraAngle)
- 依存: YOLOv11 (ultralytics)
- カメラ固定を前提とした設計（将来的にカメラ移動に対応予定）
