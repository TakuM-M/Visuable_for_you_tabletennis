"""
姿勢データ拡張パイプライン

CSVフォーマットの正規化された骨格データに対してデータ拡張を適用します。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import json

from src.traing.config import AugmentationPipelineConfig
from src.traing.exceptions import (
    DataInputError,
    AugmentationError,
    ExportError
)


class PoseAugmentationPipeline:
    """姿勢データ拡張パイプライン"""

    # COCO形式のキーポイント名
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    # 左右のキーポイントのペア（反転時に入れ替える）
    LEFT_RIGHT_PAIRS = [
        ("left_eye", "right_eye"),
        ("left_ear", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_elbow", "right_elbow"),
        ("left_wrist", "right_wrist"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ]

    def __init__(self, config: AugmentationPipelineConfig):
        """
        初期化

        Args:
            config: データ拡張パイプラインの設定
        """
        self.config = config
        if config.augmentation.random_seed is not None:
            np.random.seed(config.augmentation.random_seed)

    @classmethod
    def create_default(
        cls,
        augmentation_factor: int = 5,
        random_seed: Optional[int] = None
    ) -> 'PoseAugmentationPipeline':
        """
        デフォルト設定でPoseAugmentationPipelineを作成

        Args:
            augmentation_factor: データを何倍に拡張するか
            random_seed: ランダムシード

        Returns:
            デフォルト設定のPoseAugmentationPipeline
        """
        config = AugmentationPipelineConfig.create_default(
            augmentation_factor=augmentation_factor,
            random_seed=random_seed
        )
        return cls(config)

    def augment_csv(
        self,
        input_csv: str,
        output_csv: str,
        output_metadata: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        CSVファイルのデータを拡張

        Args:
            input_csv: 入力CSVファイルのパス
            output_csv: 出力CSVファイルのパス
            output_metadata: メタデータ出力パス（Noneの場合は自動生成）

        Returns:
            拡張結果の統計情報

        Raises:
            DataInputError: 入力データの読み込みに失敗
            AugmentationError: データ拡張処理に失敗
            ExportError: 出力に失敗
        """
        print(f"\n{'='*70}")
        print("データ拡張パイプライン開始")
        print(f"{'='*70}\n")

        # データ読み込み
        df = self._load_csv(input_csv)

        # データ検証
        self._validate_csv_format(df)

        # 拡張処理
        augmented_df = self._augment_dataframe(df)

        # データ保存
        self._save_csv(augmented_df, output_csv)

        # メタデータ保存
        if self.config.save_metadata:
            if output_metadata is None:
                output_metadata = str(Path(output_csv).with_suffix('.json'))
            metadata = self._create_metadata(df, augmented_df)
            self._save_metadata(metadata, output_metadata)

        # 統計情報
        results = {
            'original_samples': len(df),
            'augmented_samples': len(augmented_df),
            'augmentation_factor': len(augmented_df) / len(df),
            'output_csv': output_csv,
            'output_metadata': output_metadata if self.config.save_metadata else None
        }

        print(f"\n{'='*70}")
        print("データ拡張完了")
        print(f"  元データ: {results['original_samples']} サンプル")
        print(f"  拡張後: {results['augmented_samples']} サンプル")
        print(f"  拡張倍率: {results['augmentation_factor']:.1f}x")
        print(f"  出力: {output_csv}")
        if self.config.save_metadata:
            print(f"  メタデータ: {output_metadata}")
        print(f"{'='*70}\n")

        return results

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        CSVファイルを読み込む

        Args:
            csv_path: CSVファイルのパス

        Returns:
            読み込んだデータフレーム

        Raises:
            DataInputError: ファイル読み込みに失敗
        """
        print(f"データ読み込み中: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(f"  ✓ {len(df)} サンプル読み込み完了\n")
            return df
        except FileNotFoundError:
            raise DataInputError(csv_path, "ファイルが存在しません")
        except Exception as e:
            raise DataInputError(csv_path, str(e))

    def _validate_csv_format(self, df: pd.DataFrame):
        """
        CSVフォーマットを検証

        Args:
            df: データフレーム

        Raises:
            DataInputError: フォーマットが不正
        """
        print("データフォーマット検証中...")

        # 必須カラムの確認
        required_columns = ['frame', 'timestamp', 'track_id']
        for col in required_columns:
            if col not in df.columns:
                raise DataInputError("", f"必須カラム '{col}' が見つかりません")

        # 正規化座標カラムの確認
        has_normalized = all(
            f'{kp_name}_norm_x' in df.columns and f'{kp_name}_norm_y' in df.columns
            for kp_name in self.KEYPOINT_NAMES
        )

        if not has_normalized:
            raise DataInputError(
                "",
                "正規化座標カラム（*_norm_x, *_norm_y）が見つかりません。\n"
                "PlayerPoseExporterでnormalize_poses()を実行してからエクスポートしてください。"
            )

        # 正規化メタデータの確認（推奨）
        has_metadata = all(
            col in df.columns
            for col in ['hip_center_x', 'hip_center_y', 'scale_factor']
        )

        if not has_metadata:
            print("  ⚠ 警告: 正規化メタデータ（hip_center, scale_factor）が見つかりません")
            print("         逆変換が必要な場合は正しく動作しない可能性があります")

        print(f"  ✓ フォーマット検証完了")
        print(f"    - 正規化座標: あり")
        print(f"    - 正規化メタデータ: {'あり' if has_metadata else 'なし'}\n")

    def _augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        データフレーム全体を拡張

        Args:
            df: 元のデータフレーム

        Returns:
            拡張されたデータフレーム
        """
        print(f"データ拡張実行中...")
        print(f"  拡張設定:")
        print(f"    - 拡張倍率: {self.config.augmentation_factor}x")
        print(f"    - 元データ保持: {self.config.preserve_original}")
        if self.config.augmentation.horizontal_flip:
            print(f"    - 左右反転: 有効 (確率 {self.config.augmentation.horizontal_flip_prob})")
        if self.config.augmentation.add_noise:
            print(f"    - ノイズ付加: 有効 (std {self.config.augmentation.noise_std})")
        if self.config.augmentation.rotation:
            print(f"    - 回転: 有効 (±{self.config.augmentation.rotation_range}度)")
        if self.config.augmentation.scaling:
            print(f"    - スケーリング: 有効 {self.config.augmentation.scale_range}")
        if self.config.augmentation.keypoint_dropout:
            print(f"    - ドロップアウト: 有効 (確率 {self.config.augmentation.dropout_prob})")
        print()

        augmented_dfs = []

        # 元データを保持する場合
        if self.config.preserve_original:
            df_copy = df.copy()
            df_copy['augmentation_id'] = 0
            augmented_dfs.append(df_copy)

        # 拡張処理
        start_aug_id = 1 if self.config.preserve_original else 0
        pbar = tqdm(
            range(start_aug_id, self.config.augmentation_factor),
            desc="Augmenting",
            disable=not self.config.show_progress
        )

        for aug_id in pbar:
            augmented_df = self._apply_augmentation(df, aug_id)
            augmented_dfs.append(augmented_df)

        # 結合
        result_df = pd.concat(augmented_dfs, ignore_index=True)

        # ソート（frame, augmentation_id順）
        result_df = result_df.sort_values(['frame', 'augmentation_id']).reset_index(drop=True)

        print(f"  ✓ 拡張完了: {len(df)} → {len(result_df)} サンプル\n")

        return result_df

    def _apply_augmentation(self, df: pd.DataFrame, aug_id: int) -> pd.DataFrame:
        """
        拡張を適用

        Args:
            df: 元のデータフレーム
            aug_id: 拡張ID

        Returns:
            拡張されたデータフレーム
        """
        augmented_df = df.copy()
        augmented_df['augmentation_id'] = aug_id

        # 各拡張を適用
        if self.config.augmentation.horizontal_flip and np.random.rand() < self.config.augmentation.horizontal_flip_prob:
            augmented_df = self._apply_horizontal_flip(augmented_df)

        if self.config.augmentation.rotation:
            augmented_df = self._apply_rotation(augmented_df)

        if self.config.augmentation.scaling:
            augmented_df = self._apply_scaling(augmented_df)

        if self.config.augmentation.add_noise:
            augmented_df = self._apply_noise(augmented_df)

        if self.config.augmentation.keypoint_dropout:
            augmented_df = self._apply_keypoint_dropout(augmented_df)

        return augmented_df

    def _apply_horizontal_flip(self, df: pd.DataFrame) -> pd.DataFrame:
        """左右反転を適用"""
        for left_kp, right_kp in self.LEFT_RIGHT_PAIRS:
            # X座標を反転
            df[f'{left_kp}_norm_x'] = -df[f'{left_kp}_norm_x']
            df[f'{right_kp}_norm_x'] = -df[f'{right_kp}_norm_x']

            # 左右を入れ替え
            df[[f'{left_kp}_norm_x', f'{right_kp}_norm_x']] = df[[f'{right_kp}_norm_x', f'{left_kp}_norm_x']].values
            df[[f'{left_kp}_norm_y', f'{right_kp}_norm_y']] = df[[f'{right_kp}_norm_y', f'{left_kp}_norm_y']].values

        # noseのX座標も反転
        df['nose_norm_x'] = -df['nose_norm_x']

        return df

    def _apply_rotation(self, df: pd.DataFrame) -> pd.DataFrame:
        """回転を適用"""
        angle_deg = np.random.uniform(-self.config.augmentation.rotation_range, self.config.augmentation.rotation_range)
        angle_rad = np.deg2rad(angle_deg)

        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        for kp_name in self.KEYPOINT_NAMES:
            x_col = f'{kp_name}_norm_x'
            y_col = f'{kp_name}_norm_y'

            x = df[x_col].values
            y = df[y_col].values

            df[x_col] = x * cos_angle - y * sin_angle
            df[y_col] = x * sin_angle + y * cos_angle

        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """スケーリングを適用"""
        scale = np.random.uniform(*self.config.augmentation.scale_range)

        for kp_name in self.KEYPOINT_NAMES:
            df[f'{kp_name}_norm_x'] *= scale
            df[f'{kp_name}_norm_y'] *= scale

        return df

    def _apply_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """ガウシアンノイズを適用"""
        for kp_name in self.KEYPOINT_NAMES:
            noise_x = np.random.normal(0, self.config.augmentation.noise_std, len(df))
            noise_y = np.random.normal(0, self.config.augmentation.noise_std, len(df))

            df[f'{kp_name}_norm_x'] += noise_x
            df[f'{kp_name}_norm_y'] += noise_y

        return df

    def _apply_keypoint_dropout(self, df: pd.DataFrame) -> pd.DataFrame:
        """関節ドロップアウトを適用"""
        for kp_name in self.KEYPOINT_NAMES:
            # ランダムにドロップアウト
            dropout_mask = np.random.rand(len(df)) < self.config.augmentation.dropout_prob
            df.loc[dropout_mask, f'{kp_name}_norm_x'] = 0.0
            df.loc[dropout_mask, f'{kp_name}_norm_y'] = 0.0

        return df

    def _save_csv(self, df: pd.DataFrame, output_path: str):
        """
        CSVファイルを保存

        Args:
            df: データフレーム
            output_path: 出力パス

        Raises:
            ExportError: 保存に失敗
        """
        print(f"データ保存中: {output_path}")
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"  ✓ 保存完了\n")
        except Exception as e:
            raise ExportError(str(output_path), str(e))

    def _create_metadata(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> Dict[str, Any]:
        """メタデータを作成"""
        return {
            'config': {
                'augmentation_factor': self.config.augmentation_factor,
                'preserve_original': self.config.preserve_original,
                'is_sequence': self.config.is_sequence,
                'augmentation': {
                    'horizontal_flip': self.config.augmentation.horizontal_flip,
                    'horizontal_flip_prob': self.config.augmentation.horizontal_flip_prob,
                    'add_noise': self.config.augmentation.add_noise,
                    'noise_std': self.config.augmentation.noise_std,
                    'rotation': self.config.augmentation.rotation,
                    'rotation_range': self.config.augmentation.rotation_range,
                    'scaling': self.config.augmentation.scaling,
                    'scale_range': list(self.config.augmentation.scale_range),
                    'keypoint_dropout': self.config.augmentation.keypoint_dropout,
                    'dropout_prob': self.config.augmentation.dropout_prob,
                    'random_seed': self.config.augmentation.random_seed
                }
            },
            'statistics': {
                'original_samples': len(original_df),
                'augmented_samples': len(augmented_df),
                'augmentation_factor_actual': len(augmented_df) / len(original_df)
            }
        }

    def _save_metadata(self, metadata: Dict[str, Any], output_path: str):
        """
        メタデータを保存

        Args:
            metadata: メタデータ
            output_path: 出力パス

        Raises:
            ExportError: 保存に失敗
        """
        print(f"メタデータ保存中: {output_path}")
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  ✓ 保存完了\n")
        except Exception as e:
            raise ExportError(str(output_path), str(e))
