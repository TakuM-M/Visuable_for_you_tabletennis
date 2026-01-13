# プレー検知LSTMモデル 使用ガイド

## 概要

卓球のプレー映像から「プレー中」と「プレー外」を自動判別するLSTMベースの深層学習モデルです。
正規化された骨格データの時系列パターンを学習し、各フレームがプレー中かどうかを分類します。

## モデルアーキテクチャ

### 1. PlayClassifierLSTM（基本モデル）

```
入力: (batch, sequence_length=30, features=34)
  ↓
BatchNormalization
  ↓
Bidirectional LSTM (hidden_size=128, num_layers=2)
  ↓
Attention機構（オプション）
  ↓
全結合層 → ReLU → Dropout
  ↓
全結合層 → ReLU → Dropout
  ↓
全結合層 → Sigmoid
  ↓
出力: (batch, sequence_length, 1) プレー中の確率
```

**特徴:**
- 双方向LSTM で前後の文脈を考慮
- Attention機構で重要なフレーム（インパクト時など）に注目
- 各フレームごとに確率を出力

### 2. PlayClassifierCNNLSTM（高度版）

```
入力: (batch, sequence_length=30, features=34)
  ↓
1D-CNN (局所的なパターン抽出)
  ↓
Bidirectional LSTM (時系列パターン)
  ↓
全結合層 → Sigmoid
  ↓
出力: (batch, sequence_length, 1)
```

**特徴:**
- CNNで局所的な動作パターンを抽出してからLSTMで処理
- より複雑なパターンを学習可能
- 学習に時間がかかる

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

必要なライブラリ:
- PyTorch >= 2.0.0
- pandas
- numpy
- tqdm
- tensorboard

### 2. プロジェクト構造

```
Visuable_for_you_tabletennis/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── play_classifier.py          # LSTMモデル定義
│   ├── data/
│   │   └── dataset.py                   # データセット・データローダー
│   └── training/
│       ├── train_play_classifier.py     # 学習スクリプト
│       └── predict_play_scenes.py       # 推論スクリプト
├── data/
│   └── labels/                          # ラベルデータ（手動アノテーション）
├── output/
│   ├── training/                        # 学習済みモデル
│   └── predictions/                     # 予測結果
└── requirements.txt
```

## 使い方

### ステップ1: ラベルデータの作成

プレー中/プレー外のラベルデータを作成します（CSV形式）。

**ラベルCSVフォーマット:**
```csv
frame,label
0,0
1,0
2,0
...
50,1
51,1
...
```

- `frame`: フレーム番号
- `label`: 0=プレー外, 1=プレー中

**ラベル作成の推奨方法:**
1. 動画を見ながら手動でラベル付け
2. サービスモーション開始〜ボールが返球されるまでを「プレー中」とする
3. ボール拾いや休憩は「プレー外」

### ステップ2: モデルの学習

```bash
python src/training/train_play_classifier.py \
  --train-csv data/normalized_pose_data.csv \
  --train-labels data/labels/train_labels.csv \
  --val-csv data/normalized_pose_data_val.csv \
  --val-labels data/labels/val_labels.csv \
  --epochs 50 \
  --batch-size 32 \
  --sequence-length 30 \
  --device cuda
```

**主要なオプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--train-csv` | 訓練データCSV（正規化済み骨格データ） | 必須 |
| `--train-labels` | 訓練ラベルCSV | 必須 |
| `--val-csv` | 検証データCSV | オプション |
| `--val-labels` | 検証ラベルCSV | オプション |
| `--model-type` | モデルタイプ（lstm/cnn_lstm） | lstm |
| `--hidden-size` | LSTM隠れ層サイズ | 128 |
| `--num-layers` | LSTM層数 | 2 |
| `--dropout` | ドロップアウト率 | 0.3 |
| `--no-attention` | Attentionを使用しない | False |
| `--epochs` | エポック数 | 50 |
| `--batch-size` | バッチサイズ | 32 |
| `--lr` | 学習率 | 0.001 |
| `--sequence-length` | シーケンス長（フレーム数） | 30 |
| `--stride` | シーケンスのストライド | 5 |
| `--device` | デバイス（cuda/cpu/mps） | cuda |
| `--output-dir` | 出力ディレクトリ | output/training |

**学習のヒント:**
- データが少ない場合は `--stride` を小さくしてデータ拡張
- 過学習する場合は `--dropout` を大きくする
- GPUメモリが足りない場合は `--batch-size` を減らす

### ステップ3: 予測

学習済みモデルを使って新しいデータを予測します。

```bash
python src/training/predict_play_scenes.py \
  --model output/training/20250113_120000/best_model.pth \
  --csv data/test_normalized_pose_data.csv \
  --output output/predictions/test_predictions.csv \
  --threshold 0.5 \
  --visualize
```

**オプション:**

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--model` | 学習済みモデルのパス | 必須 |
| `--config` | 学習時の設定ファイル | 自動検出 |
| `--csv` | 正規化された骨格データCSV | 必須 |
| `--video` | 元動画（動画出力する場合） | オプション |
| `--output` | 出力CSVパス | 自動生成 |
| `--output-video` | 出力動画パス | 自動生成 |
| `--threshold` | プレー検出の閾値 | 0.5 |
| `--device` | デバイス | cuda |
| `--visualize` | 予測結果を可視化 | False |

**出力ファイル:**
1. `*_predictions.csv` - 各フレームの予測確率
2. `*_scenes.csv` - 検出されたプレーシーン（開始/終了フレーム）
3. `*_prediction_visualization.png` - 予測結果の可視化（--visualize時）
4. `*_predicted.mp4` - 予測結果を重畳した動画（--video指定時）

### ステップ4: 動画の切り出し

検出されたプレーシーンを元に動画を切り出します。

```python
import cv2
import pandas as pd

# シーン情報読み込み
scenes = pd.read_csv('output/predictions/test_scenes.csv')

# 元動画読み込み
cap = cv2.VideoCapture('data/raw/test_video.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 各シーンを別ファイルとして保存
for i, row in scenes.iterrows():
    start_frame = row['start_frame']
    end_frame = row['end_frame']

    # 該当フレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 出力動画
    out = cv2.VideoWriter(
        f'output/scenes/scene_{i+1}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    out.release()

cap.release()
```

## モデルのテスト

### モデル定義のテスト

```bash
python src/models/play_classifier.py
```

出力例:
```
PlayClassifierLSTM モデルテスト
============================================================

1. 基本LSTMモデル
  入力サイズ: torch.Size([4, 30, 34])
  出力サイズ: torch.Size([4, 30, 1])
  パラメータ数: 215,489
  確率: torch.Size([4, 30])
  予測: torch.Size([4, 30])
  Attention重み: torch.Size([4, 30, 1])

2. CNN+LSTMモデル
  入力サイズ: torch.Size([4, 30, 34])
  出力サイズ: torch.Size([4, 30, 1])
  パラメータ数: 294,337
```

### データセットのテスト

```bash
python src/data/dataset.py
```

## トラブルシューティング

### GPU メモリ不足

```bash
# バッチサイズを減らす
--batch-size 16

# または CPUを使用
--device cpu
```

### 学習が収束しない

```bash
# 学習率を下げる
--lr 0.0001

# または層数を減らす
--num-layers 1
```

### 過学習

```bash
# ドロップアウトを増やす
--dropout 0.5

# データ拡張（ストライドを小さく）
--stride 2
```

### クラス不均衡

訓練スクリプト内の `Trainer` 初期化時に `class_weights` を指定:

```python
# プレー外:プレー中 = 7:3 の場合
class_weights = [0.3, 0.7]  # [非プレー, プレー]
```

## パフォーマンス最適化

### 学習の高速化

1. **GPUの使用**: `--device cuda`
2. **DataLoaderの並列化**: `--num-workers 4`
3. **Mixed Precision Training** (上級):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### 推論の高速化

1. **バッチ処理**: 複数シーケンスをまとめて処理
2. **TorchScript**: モデルの最適化
   ```python
   traced_model = torch.jit.trace(model, example_input)
   traced_model.save("model_traced.pt")
   ```

## 評価指標

### 学習時の指標

- **Loss**: Binary Cross Entropy Loss
- **Accuracy**: 全体の正答率
- **Precision**: プレー中と予測した中で実際にプレー中だった割合
- **Recall**: 実際のプレー中を正しく検出できた割合
- **F1 Score**: Precision と Recall の調和平均（主要指標）

### TensorBoard で確認

```bash
tensorboard --logdir output/training/20250113_120000/logs
```

ブラウザで http://localhost:6006 を開く

## 次のステップ

1. **データ拡張**: より多くの動画でラベル付けして学習
2. **ハイパーパラメータ調整**: Grid Search や Optuna で最適化
3. **アンサンブル**: 複数モデルの予測を組み合わせる
4. **特徴量エンジニアリング**: 手首の速度などの特徴量を追加
5. **後処理**: 短すぎるシーンを除外、スムージングなど

## 参考資料

- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Attention機構: https://arxiv.org/abs/1409.0473
- 時系列分類: https://arxiv.org/abs/1809.04356
