"""
File management utilities for Jupyter notebooks.

This module provides utilities for:
- Google Colab file operations (upload, download, Drive mount)
- Path management
- Configuration file loading
- Model file management
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union


class ColabFileManager:
    """Google Colab用のファイル管理クラス"""

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize file manager.

        Args:
            project_root: プロジェクトのルートディレクトリ。
                         Noneの場合は自動検出を試みる。
        """
        self.project_root = Path(project_root) if project_root else self._detect_project_root()
        self.is_colab = self._check_colab_environment()

    @staticmethod
    def _check_colab_environment() -> bool:
        """Google Colab環境かどうかをチェック"""
        try:
            import google.colab
            return True
        except ImportError:
            return False

    @staticmethod
    def _detect_project_root() -> Path:
        """プロジェクトのルートディレクトリを自動検出"""
        current = Path.cwd()

        # プロジェクトのマーカーファイル/ディレクトリを探す
        markers = ['src', '.git', 'setup.py', 'pyproject.toml']

        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                return current
            current = current.parent

        # 見つからなければカレントディレクトリを返す
        return Path.cwd()

    def mount_drive(self, force_remount: bool = False) -> bool:
        """
        Google Driveをマウント

        Args:
            force_remount: 強制的に再マウントする

        Returns:
            マウント成功したかどうか
        """
        if not self.is_colab:
            print("⚠ Google Colab環境ではありません。Driveマウントはスキップします。")
            return False

        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=force_remount)
            print("✓ Google Driveをマウントしました")
            return True
        except Exception as e:
            print(f"✗ Google Driveのマウントに失敗: {e}")
            return False

    def upload_files(self) -> Dict[str, bytes]:
        """
        Colabでファイルをアップロード

        Returns:
            アップロードされたファイルの辞書 {filename: content}
        """
        if not self.is_colab:
            print("⚠ Google Colab環境ではありません。")
            return {}

        try:
            from google.colab import files
            uploaded = files.upload()
            print(f"✓ {len(uploaded)}個のファイルをアップロードしました")
            return uploaded
        except Exception as e:
            print(f"✗ ファイルのアップロードに失敗: {e}")
            return {}

    def download_file(self, file_path: Union[str, Path]) -> bool:
        """
        Colabでファイルをダウンロード

        Args:
            file_path: ダウンロードするファイルのパス

        Returns:
            ダウンロード成功したかどうか
        """
        if not self.is_colab:
            print("⚠ Google Colab環境ではありません。")
            return False

        file_path = Path(file_path)
        if not file_path.exists():
            print(f"✗ ファイルが見つかりません: {file_path}")
            return False

        try:
            from google.colab import files
            files.download(str(file_path))
            print(f"✓ ファイルをダウンロード: {file_path}")
            return True
        except Exception as e:
            print(f"✗ ファイルのダウンロードに失敗: {e}")
            return False

    def download_files(self, file_paths: List[Union[str, Path]]) -> int:
        """
        複数のファイルをダウンロード

        Args:
            file_paths: ダウンロードするファイルのパスのリスト

        Returns:
            成功したダウンロード数
        """
        success_count = 0
        for file_path in file_paths:
            if self.download_file(file_path):
                success_count += 1

        print(f"\n✓ {success_count}/{len(file_paths)}個のファイルをダウンロードしました")
        return success_count

    def change_to_project_root(self) -> bool:
        """
        プロジェクトのルートディレクトリに移動

        Returns:
            成功したかどうか
        """
        try:
            os.chdir(self.project_root)
            print(f"✓ 作業ディレクトリを変更: {self.project_root}")
            return True
        except Exception as e:
            print(f"✗ ディレクトリの変更に失敗: {e}")
            return False

    def ensure_directory(self, dir_path: Union[str, Path], relative: bool = True) -> Path:
        """
        ディレクトリが存在することを保証（なければ作成）

        Args:
            dir_path: ディレクトリパス
            relative: Trueの場合、project_rootからの相対パスとして扱う

        Returns:
            絶対パス
        """
        if relative and not Path(dir_path).is_absolute():
            dir_path = self.project_root / dir_path
        else:
            dir_path = Path(dir_path)

        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def get_path(self, *paths: str, relative: bool = True) -> Path:
        """
        パスを取得（プロジェクトルートからの相対パスまたは絶対パス）

        Args:
            *paths: パスの要素
            relative: Trueの場合、project_rootからの相対パスとして扱う

        Returns:
            パスオブジェクト
        """
        if relative:
            return self.project_root.joinpath(*paths)
        else:
            return Path(*paths)

    def check_file_exists(self, file_path: Union[str, Path],
                         relative: bool = True,
                         verbose: bool = True) -> bool:
        """
        ファイルの存在をチェック

        Args:
            file_path: ファイルパス
            relative: Trueの場合、project_rootからの相対パスとして扱う
            verbose: メッセージを表示するかどうか

        Returns:
            ファイルが存在するかどうか
        """
        if relative:
            file_path = self.project_root / file_path
        else:
            file_path = Path(file_path)

        exists = file_path.exists()

        if verbose:
            if exists:
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}")

        return exists

    def check_files_exist(self, file_paths: List[Union[str, Path]],
                         relative: bool = True) -> Dict[str, bool]:
        """
        複数のファイルの存在をチェック

        Args:
            file_paths: ファイルパスのリスト
            relative: Trueの場合、project_rootからの相対パスとして扱う

        Returns:
            {ファイルパス: 存在するかどうか}の辞書
        """
        results = {}
        for file_path in file_paths:
            full_path = self.project_root / file_path if relative else Path(file_path)
            results[str(file_path)] = full_path.exists()

        return results


class ConfigLoader:
    """設定ファイルを読み込むユーティリティクラス"""

    @staticmethod
    def load_json(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        JSON設定ファイルを読み込む

        Args:
            config_path: 設定ファイルのパス

        Returns:
            設定の辞書
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✓ 設定ファイルを読み込みました: {config_path}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON形式が不正です: {e}")

    @staticmethod
    def save_json(config: Dict[str, Any],
                  config_path: Union[str, Path],
                  indent: int = 2) -> bool:
        """
        JSON設定ファイルを保存する

        Args:
            config: 設定の辞書
            config_path: 保存先のパス
            indent: JSONのインデント

        Returns:
            保存成功したかどうか
        """
        config_path = Path(config_path)

        try:
            # ディレクトリが存在しなければ作成
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=indent, ensure_ascii=False)

            print(f"✓ 設定ファイルを保存しました: {config_path}")
            return True
        except Exception as e:
            print(f"✗ 設定ファイルの保存に失敗: {e}")
            return False

    @staticmethod
    def load_api_config(config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        API設定ファイルを読み込む（Roboflowなど）

        Args:
            config_path: 設定ファイルのパス

        Returns:
            API設定の辞書
        """
        return ConfigLoader.load_json(config_path)


class DatasetPathManager:
    """データセットのパス管理クラス"""

    def __init__(self, file_manager: ColabFileManager):
        """
        Initialize dataset path manager.

        Args:
            file_manager: ColabFileManagerのインスタンス
        """
        self.fm = file_manager

    def get_video_paths(self, video_names: List[str],
                       base_dir: str = "data/detect") -> Dict[str, Dict[str, Path]]:
        """
        動画データのパスを取得

        Args:
            video_names: 動画名のリスト
            base_dir: データのベースディレクトリ

        Returns:
            {video_name: {csv_path, label_path, ...}}の辞書
        """
        paths = {}

        for video_name in video_names:
            video_dir = self.fm.get_path(base_dir, video_name)

            paths[video_name] = {
                'dir': video_dir,
                'original_csv': video_dir / 'original_pose_data.csv',
                'augment_csv': video_dir / 'augment_pose_data.csv',
                'label': video_dir / 'play_labels.csv'
            }

        return paths

    def check_video_data(self, video_names: List[str],
                        base_dir: str = "data/detect") -> None:
        """
        動画データの存在をチェックして表示

        Args:
            video_names: 動画名のリスト
            base_dir: データのベースディレクトリ
        """
        print("=" * 60)
        print("データディレクトリの確認")
        print("=" * 60)

        paths = self.get_video_paths(video_names, base_dir)

        for video_name, file_paths in paths.items():
            print(f"\n{video_name}:")

            # ディレクトリ
            dir_exists = file_paths['dir'].exists()
            print(f"  Dir: {'✓' if dir_exists else '✗'} {file_paths['dir']}")

            # オリジナルCSV
            csv_exists = file_paths['original_csv'].exists()
            print(f"  CSV (Original): {'✓' if csv_exists else '✗'} {file_paths['original_csv']}")

            # 拡張CSV
            aug_exists = file_paths['augment_csv'].exists()
            print(f"  CSV (Augmented): {'✓' if aug_exists else '✗'} {file_paths['augment_csv']}")

            # ラベル
            label_exists = file_paths['label'].exists()
            print(f"  Label: {'✓' if label_exists else '✗'} {file_paths['label']}")

        print("=" * 60)


class ModelFileManager:
    """モデルファイルの管理クラス"""

    def __init__(self, file_manager: ColabFileManager):
        """
        Initialize model file manager.

        Args:
            file_manager: ColabFileManagerのインスタンス
        """
        self.fm = file_manager

    def save_to_drive(self, source_dir: Union[str, Path],
                     drive_path: str,
                     files_to_copy: Optional[List[str]] = None,
                     use_timestamp: bool = True) -> tuple[int, str]:
        """
        ファイルをGoogle Driveにコピー

        Args:
            source_dir: コピー元ディレクトリ
            drive_path: Google Drive上の保存先パス
            files_to_copy: コピーするファイルのリスト（Noneの場合は全て）
            use_timestamp: Trueの場合、タイムスタンプ付きサブディレクトリを作成

        Returns:
            (コピー成功したファイル数, 実際の保存先パス)
        """
        import shutil
        from datetime import datetime

        source_dir = Path(source_dir)

        if use_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            drive_path = os.path.join(drive_path, timestamp)

        os.makedirs(drive_path, exist_ok=True)

        if files_to_copy is None:
            files_to_copy = [f.name for f in source_dir.iterdir() if f.is_file()]

        success_count = 0
        for filename in files_to_copy:
            src = source_dir / filename
            if src.exists():
                dst = os.path.join(drive_path, filename)
                try:
                    shutil.copy2(src, dst)
                    print(f"✓ コピー完了: {filename}")
                    success_count += 1
                except Exception as e:
                    print(f"✗ コピー失敗: {filename} ({e})")
            else:
                print(f"⚠ ファイルが見つかりません: {filename}")

        print(f"\n✓ {success_count}/{len(files_to_copy)}個のファイルをコピーしました")
        print(f"保存先: {drive_path}")

        return success_count, drive_path

    def load_from_drive(self, drive_path: str,
                       destination_dir: Union[str, Path],
                       files_to_load: Optional[List[str]] = None) -> int:
        """
        Google Driveからファイルを読み込み

        Args:
            drive_path: Google Drive上のソースパス
            destination_dir: コピー先ディレクトリ
            files_to_load: 読み込むファイルのリスト（Noneの場合は全て）

        Returns:
            読み込み成功したファイル数
        """
        import shutil

        drive_path = Path(drive_path)
        destination_dir = Path(destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)

        # 読み込むファイルのリストを取得
        if files_to_load is None:
            files_to_load = [f.name for f in drive_path.iterdir() if f.is_file()]

        success_count = 0
        for filename in files_to_load:
            src = drive_path / filename
            if src.exists():
                dst = destination_dir / filename
                try:
                    shutil.copy2(src, dst)
                    print(f"✓ 読み込み完了: {filename}")
                    success_count += 1
                except Exception as e:
                    print(f"✗ 読み込み失敗: {filename} ({e})")
            else:
                print(f"⚠ ファイルが見つかりません: {filename}")

        print(f"\n✓ {success_count}/{len(files_to_load)}個のファイルを読み込みました")
        return success_count
