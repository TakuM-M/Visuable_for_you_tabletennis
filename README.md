# Visuable for You Table Tennis

卓球のプレー動画から必要な部分（プレー中）のみを自動抽出し、プレー間の不要な時間をカットするアプリケーション

## 概要

このプロジェクトは、卓球の練習や試合の動画を効率的に編集するためのツールです。動画内のサービスやラリーなどのプレー区間を自動検出し、待機時間やボール拾いの時間などをカットして、プレー部分のみを含む動画を生成します。

## 機能

- 動画内の動き検出
- サービスモーションの認識
- プレー区間の自動抽出
- 不要部分のカット
- 編集済み動画の出力

## プロジェクト構造

```
Visuable_for_you_tabletennis/
├── src/                    # ソースコード
│   ├── detection/         # 動作検出モジュール
│   ├── extraction/        # 動画抽出モジュール
│   ├── utils/            # ユーティリティ
│   └── main.py           # メインスクリプト
├── data/                  # データディレクトリ
│   ├── raw/              # 元動画
│   └── processed/        # 処理済み動画
├── output/               # 出力ディレクトリ
├── tests/                # テストコード
├── config/               # 設定ファイル
├── notes/                # 開発メモ
├── requirements.txt      # 依存ライブラリ
└── README.md            # このファイル
```

## セットアップ

### 1. リポジトリのクローン

```bash
cd /path/to/your/directory
```

### 2. 仮想環境の作成と有効化

```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate     # Windows
```

### 3. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

## 使い方

### 基本的な使用方法

```bash
python src/main.py data/raw/your_video.mp4
```

### オプション

```bash
python src/main.py data/raw/your_video.mp4 \
  -o output \           # 出力ディレクトリを指定
  -t 25 \              # 動き検出の閾値を指定
  -v                   # 詳細情報を表示
```

## 開発状況

現在はプロトタイプ段階です。

### 完了したタスク

- [x] プロジェクト構造の作成
- [x] Python環境のセットアップ
- [x] 基本モジュールの作成

### 進行中のタスク

- [ ] タスク0: データ処理（動画の収集）
- [ ] タスク1: 動画の中でサービスする時間を抽出する

## 技術スタック

- **Python 3.x**
- **OpenCV**: 動画処理
- **MediaPipe**: 姿勢・動作検出
- **NumPy**: 数値計算
- **scikit-learn**: 機械学習（オプション）

## ライセンス

このプロジェクトは開発中です。

## 貢献

プロジェクトは現在開発中です。

## 注意事項

- このツールは開発中のプロトタイプです
- 動画ファイルは適切な形式（MP4, AVI等）である必要があります
- 処理には時間がかかる場合があります
