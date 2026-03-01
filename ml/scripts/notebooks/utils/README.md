# Notebook Utils

Jupyter notebook用のユーティリティモジュール集です。Google Colabでの実行とローカル環境での実行の両方をサポートします。

## モジュール一覧

### 1. ColabFileManager

Google Colab環境でのファイル操作を簡素化するクラスです。

#### 主な機能

- Google Driveのマウント
- ファイルのアップロード/ダウンロード
- プロジェクトパスの管理
- ファイル存在確認

#### 使用例

```python
from utils import ColabFileManager

# ファイルマネージャーの初期化
fm = ColabFileManager()  # 自動的にプロジェクトルートを検出

# または明示的にプロジェクトルートを指定
fm = ColabFileManager(project_root='/content/drive/MyDrive/my_project')

# Google Driveをマウント（Colab環境のみ）
fm.mount_drive()

# ファイルをアップロード（Colab環境のみ）
uploaded_files = fm.upload_files()

# ファイルをダウンロード（Colab環境のみ）
fm.download_file('output/result.mp4')
fm.download_files(['output/result.mp4', 'output/data.csv'])

# プロジェクトルートに移動
fm.change_to_project_root()

# ディレクトリの作成
output_dir = fm.ensure_directory('output/models')

# パスの取得
config_path = fm.get_path('config', 'settings.json')  # プロジェクトルートからの相対パス

# ファイルの存在確認
exists = fm.check_file_exists('data/sample.csv')

# 複数のファイルの存在確認
results = fm.check_files_exist(['data/train.csv', 'data/val.csv'])
```

### 2. ConfigLoader

JSON設定ファイルの読み書きを行うユーティリティクラスです。

#### 使用例

```python
from utils import ConfigLoader

# JSON設定ファイルを読み込み
config = ConfigLoader.load_json('config/settings.json')

# 設定を保存
ConfigLoader.save_json(config, 'config/new_settings.json', indent=2)

# API設定を読み込み（Roboflowなど）
api_config = ConfigLoader.load_api_config('config/api.config')
roboflow_config = api_config['roboflow']
```

### 3. DatasetPathManager

データセットのパス管理を行うクラスです。

#### 使用例

```python
from utils import ColabFileManager, DatasetPathManager

fm = ColabFileManager()
dpm = DatasetPathManager(fm)

# 動画データのパスを取得
video_names = ['sample_video_01', 'sample_video_02', 'sample_video_03']
paths = dpm.get_video_paths(video_names, base_dir='data/detect')

# パス情報にアクセス
for video_name, file_paths in paths.items():
    print(f"{video_name}:")
    print(f"  Directory: {file_paths['dir']}")
    print(f"  Original CSV: {file_paths['original_csv']}")
    print(f"  Augmented CSV: {file_paths['augment_csv']}")
    print(f"  Labels: {file_paths['label']}")

# データの存在確認と表示
dpm.check_video_data(video_names, base_dir='data/detect')
```

出力例：
```
============================================================
データディレクトリの確認
============================================================

sample_video_01:
  Dir: ✓ data/detect/sample_video_01
  CSV (Original): ✓ data/detect/sample_video_01/original_pose_data.csv
  CSV (Augmented): ✓ data/detect/sample_video_01/augment_pose_data.csv
  Label: ✓ data/detect/sample_video_01/play_labels.csv
...
```

### 4. ModelFileManager

モデルファイルのGoogle Drive保存・読み込みを行うクラスです。

#### 使用例

```python
from utils import ColabFileManager, ModelFileManager

fm = ColabFileManager()
fm.mount_drive()  # Colab環境の場合

mfm = ModelFileManager(fm)

# モデルをGoogle Driveに保存
mfm.save_to_drive(
    source_dir='output/training/20260130_123456',
    drive_path='/content/drive/MyDrive/trained_models/my_model',
    files_to_copy=['best_model.pth', 'config.json', 'training_history.json']
)

# Google Driveからモデルを読み込み
mfm.load_from_drive(
    drive_path='/content/drive/MyDrive/trained_models/my_model',
    destination_dir='models/loaded_model',
    files_to_load=['best_model.pth', 'config.json']
)
```

## ノートブックでの使用パターン

### Google Colab環境でのセットアップ

```python
# 1. utilsをインポート
import sys
from pathlib import Path

# scriptsディレクトリをパスに追加
sys.path.insert(0, str(Path.cwd().parent))

from utils import ColabFileManager, ConfigLoader, DatasetPathManager, ModelFileManager

# 2. ファイルマネージャーを初期化
fm = ColabFileManager()

# 3. Google Driveをマウント
fm.mount_drive()

# 4. プロジェクトルートに移動
fm.change_to_project_root()

# 5. 設定ファイルをアップロード
if fm.is_colab:
    uploaded = fm.upload_files()
    config_path = list(uploaded.keys())[0]
    config = ConfigLoader.load_json(config_path)
else:
    config = ConfigLoader.load_json(fm.get_path('config/api.config'))
```

### モデル訓練後の保存

```python
# モデルファイルマネージャーを初期化
mfm = ModelFileManager(fm)

# 保存先パスを設定
if fm.is_colab:
    drive_model_dir = '/content/drive/MyDrive/my_project/models'
else:
    drive_model_dir = str(fm.get_path('models/trained'))

# モデルを保存
mfm.save_to_drive(
    source_dir='output/training/weights',
    drive_path=drive_model_dir,
    files_to_copy=['best.pt', 'best.onnx', 'best.torchscript']
)
```

### 結果のダウンロード

```python
# Colab環境でのみ有効
if fm.is_colab:
    # 単一ファイル
    fm.download_file('output/result_video.mp4')

    # 複数ファイル
    fm.download_files([
        'output/result_video.mp4',
        'output/pose_data.csv',
        'models/best_model.pth'
    ])
```

## 環境検出

utilsモジュールは自動的に環境を検出します：

```python
fm = ColabFileManager()

if fm.is_colab:
    print("Google Colab環境で実行中")
    # Colab固有の処理
else:
    print("ローカル環境で実行中")
    # ローカル固有の処理
```

## プロジェクトルートの自動検出

`ColabFileManager`は以下のマーカーファイル/ディレクトリを探してプロジェクトルートを自動検出します：

- `src/` ディレクトリ
- `.git/` ディレクトリ
- `setup.py` ファイル
- `pyproject.toml` ファイル

明示的に指定することも可能です：

```python
fm = ColabFileManager(project_root='/path/to/project')
```

## エラーハンドリング

utilsモジュールは適切なエラーメッセージを表示します：

```python
# ファイルが見つからない場合
config = ConfigLoader.load_json('nonexistent.json')
# FileNotFoundError: 設定ファイルが見つかりません: nonexistent.json

# JSON形式が不正な場合
config = ConfigLoader.load_json('invalid.json')
# ValueError: JSON形式が不正です: ...

# ファイルのダウンロードに失敗した場合
fm.download_file('nonexistent_file.txt')
# ✗ ファイルが見つかりません: nonexistent_file.txt
```

## ベストプラクティス

1. **環境に応じた処理の分岐**
   ```python
   if fm.is_colab:
       # Colab固有の処理（ファイルアップロード等）
   else:
       # ローカル環境の処理（直接パス指定等）
   ```

2. **相対パスの活用**
   ```python
   # プロジェクトルートからの相対パスを使用
   config_path = fm.get_path('config', 'settings.json')
   data_dir = fm.ensure_directory('data/processed')
   ```

3. **存在確認の実施**
   ```python
   # ファイル操作前に存在確認
   if fm.check_file_exists('data/train.csv'):
       # 処理を実行
   ```

4. **一貫したパス管理**
   ```python
   # DatasetPathManagerを使用して一貫したパス管理
   dpm = DatasetPathManager(fm)
   paths = dpm.get_video_paths(video_names)
   ```

## トラブルシューティング

### ImportError が発生する場合

```python
# sys.pathにutilsのパスを追加
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from utils import ColabFileManager
```

### Google Drive マウントエラー

```python
# 強制的に再マウント
fm.mount_drive(force_remount=True)
```

### プロジェクトルートが正しく検出されない場合

```python
# 明示的にプロジェクトルートを指定
fm = ColabFileManager(project_root='/content/drive/MyDrive/my_project')
```
