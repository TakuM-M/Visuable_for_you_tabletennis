# 卓球台検出モデル学習ガイド

このガイドでは、カスタムYOLOモデルで卓球台検出精度を向上させる手順を説明します。

## 概要

1. 動画からフレーム抽出
2. アノテーション（ラベリング）
3. データセット準備
4. モデル学習
5. 評価とテスト

---

## ステップ1: フレーム抽出

動画からアノテーション用のフレームを抽出します。

### 単一動画から抽出

```bash
python src/training/extract_frames.py data/raw/sample_video_01_01.MOV \
  -o data/annotations/images \
  -i 30 \
  -m 200 \
  -w 640
```

**オプション:**
- `-o, --output`: 出力ディレクトリ（デフォルト: `data/annotations/images`）
- `-i, --interval`: フレーム抽出間隔（デフォルト: 30フレーム）
- `-m, --max-frames`: 最大抽出フレーム数（デフォルト: 200）
- `-w, --width`: リサイズ後の幅（デフォルト: 640px）

### 複数動画から一括抽出（推奨）

```bash
python src/training/extract_frames.py data/raw \
  -o data/annotations/images \
  -i 30 \
  -m 50
```

**推奨設定:**
- 動画数: 3-5本
- 1動画あたり: 50-100フレーム
- 合計: 150-500フレーム

---

## ステップ2: アノテーション

抽出したフレームに卓球台のバウンディングボックスをラベリングします。

### 推奨ツール

#### オプション1: Label Studio（推奨・GUI）

**インストール:**
```bash
pip install label-studio
```

**起動:**
```bash
label-studio
```

ブラウザで `http://localhost:8080` を開く

**設定:**
1. "Create Project" をクリック
2. Project Name: `table_tennis_detection`
3. Data Import: `data/annotations/images` フォルダをアップロード
4. Labeling Setup → Object Detection with Bounding Boxes を選択
5. Label名を `table_tennis_table` に設定
6. Start Labeling

**アノテーション手順:**
1. 画像が表示されたら、卓球台全体を囲むようにバウンディングボックスを描画
2. ラベルを `table_tennis_table` に設定
3. "Submit" で保存
4. 次の画像へ

**エクスポート:**
1. Export → YOLO 形式を選択
2. ダウンロードしたZIPを解凍
3. ラベルファイル（.txt）を `data/annotations/labels` にコピー

#### オプション2: Roboflow（オンライン・簡単）

1. https://roboflow.com にアクセス
2. プロジェクト作成
3. 画像をアップロード
4. アノテーション実行
5. YOLO形式でエクスポート

#### オプション3: CVAT（高機能）

**インストール:**
```bash
pip install cvat-cli
```

詳細: https://github.com/opencv/cvat

### YOLO形式ラベルの仕様

各画像に対応する `.txt` ファイルを作成します。

**フォーマット:**
```
<class_id> <x_center> <y_center> <width> <height>
```

**例: `sample_video_01_01_frame_00000.txt`**
```
0 0.5 0.4 0.6 0.2
```

- `class_id`: クラスID（卓球台は `0`）
- `x_center`: バウンディングボックス中心のX座標（画像幅に対する比率 0-1）
- `y_center`: バウンディングボックス中心のY座標（画像高さに対する比率 0-1）
- `width`: バウンディングボックスの幅（画像幅に対する比率 0-1）
- `height`: バウンディングボックスの高さ（画像高さに対する比率 0-1）

**重要:** 座標は全て 0-1 の範囲で正規化されます。

---

## ステップ3: データセット準備

アノテーション完了後、train/val分割を行います。

### ラベル検証

```bash
python src/training/prepare_dataset.py --validate-only
```

### データセット分割

```bash
python src/training/prepare_dataset.py \
  -i data/annotations/images \
  -l data/annotations/labels \
  -o data/table_dataset \
  -r 0.8
```

**オプション:**
- `-i, --images`: 画像ディレクトリ
- `-l, --labels`: ラベルディレクトリ
- `-o, --output`: 出力ディレクトリ
- `-r, --train-ratio`: 学習データ比率（デフォルト: 0.8 = 80%）

**出力構造:**
```
data/table_dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

---

## ステップ4: モデル学習

準備したデータセットでYOLOv11を学習します。

### 基本的な学習（CPU）

```bash
python src/training/train_table_detector.py \
  --data data/table_dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --device cpu
```

### GPU使用時（CUDA）

```bash
python src/training/train_table_detector.py \
  --data data/table_dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 32 \
  --device cuda
```

### Apple Silicon（MPS）使用時

```bash
python src/training/train_table_detector.py \
  --data data/table_dataset.yaml \
  --model yolo11n.pt \
  --epochs 100 \
  --batch 16 \
  --device mps
```

**パラメータ:**
- `--data`: データセット設定ファイル
- `--model`: ベースモデル（`yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` など）
- `--epochs`: エポック数（推奨: 100-300）
- `--batch`: バッチサイズ（GPUメモリに応じて調整）
- `--device`: デバイス（`cpu`, `cuda`, `mps`）
- `--project`: 出力ディレクトリ（デフォルト: `models/table_detector`）
- `--name`: 実験名（デフォルト: `train`）

### 学習結果

学習完了後、以下のファイルが生成されます:

```
models/table_detector/train/
├── weights/
│   ├── best.pt          # 最良モデル（使用推奨）
│   └── last.pt          # 最終エポックモデル
├── results.png          # 学習曲線
├── confusion_matrix.png # 混同行列
└── val_batch*_pred.jpg  # 検証結果の可視化
```

---

## ステップ5: モデル評価

### 検証データで評価

```bash
python src/training/train_table_detector.py \
  --validate models/table_detector/train/weights/best.pt
```

### 実際の動画でテスト

学習したモデルを実際の動画でテストします。

**table_detector.pyを更新:**

```python
# src/detection/table_detector.py の __init__ メソッド
table_detector = TableDetector(
    detection_method="yolo",
    yolo_model_path="models/table_detector/train/weights/best.pt"  # カスタムモデル
)
```

**または、コマンドラインで指定:**

```bash
python src/test_yolo_tracking.py data/raw/sample_video_01_01.MOV \
  --table-interval 0 \
  -v
```

---

## トラブルシューティング

### Q1: アノテーション数はどれくらい必要？

**A:** 最低100枚、推奨150-500枚

- 少ないデータ（50-100枚）: 過学習のリスク
- 中程度（150-300枚）: 良好な性能
- 多いデータ（500枚以上）: 最高の性能

### Q2: 学習時間はどれくらい？

**A:** 環境によります

- CPU: 100エポック = 1-3時間（データ数による）
- GPU (RTX 3060): 100エポック = 10-30分
- Apple Silicon (M1/M2): 100エポック = 30-60分

### Q3: メモリ不足エラーが出る

**A:** バッチサイズを減らす

```bash
--batch 8  # または 4
```

### Q4: 学習が進まない

**A:** 学習率を調整、またはエポック数を増やす

モデルファイルを編集して学習率を変更:
- `lr0=0.001`（より小さく）
- `epochs=200`（より長く）

### Q5: 精度が低い

**A:** 以下を確認:

1. アノテーションの品質（ラベルが正確か）
2. データの多様性（様々な角度・照明）
3. エポック数を増やす
4. より大きなモデルを使用（`yolo11s.pt`, `yolo11m.pt`）

---

## ベストプラクティス

### アノテーション時

1. **一貫性を保つ**: 卓球台の端を常に含める
2. **正確に**: できるだけ正確にバウンディングボックスを描画
3. **多様性**: 様々な角度・照明条件の画像を含める
4. **品質チェック**: 10-20枚ごとに過去のアノテーションを確認

### 学習時

1. **Early Stopping**: `--patience 50`で過学習を防ぐ
2. **定期保存**: `--save-period 10`で途中経過を保存
3. **検証**: 学習中の`results.png`で進捗を確認
4. **実験管理**: `--name`で異なる設定を試す

---

## 次のステップ

学習完了後:

1. **モデルの統合**
   - [table_detector.py](../src/detection/table_detector.py) のモデルパスを更新

2. **パフォーマンス最適化**
   - `--table-interval 0` で初回のみ検出（最速）
   - カスタムモデルは精度が高いため、低FPSでも十分

3. **本番利用**
   - 実際の試合動画で検証
   - 必要に応じてデータを追加して再学習
