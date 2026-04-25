import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import PlayClassifierLSTM
from src.detection.tracking_exporter import TrackingExporter
from src.datasets import (
    CSVPoseSequenceDataset,
    MemoryPoseSequenceDataset,
)
from src.datasets.base_dataset import collate_fn
from src.pipelines.config import PlaySceneDetectionConfig
from src.pipelines.exceptions import DataInputError


class PlaySceneDetector:
    """LSTMモデルを利用したプレーシーン検出パイプライン"""

    def __init__(self, config: PlaySceneDetectionConfig):
        """
        初期化

        Args:
            config: プレーシーン検出の設定
        """
        self.config_obj = config
        self.model_path = Path(config.model_path)
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.threshold = config.threshold
        self.min_scene_duration = config.min_scene_duration
        self.batch_size = config.batch_size
        self.config_path = Path(config.config_path) if config.config_path is not None else None

        self.model_config = self._load_config()
        self.model = self._load_model()

        print(f"PlaySceneDetector初期化完了:")
        print(f"  モデル: {self.model_path}")
        print(f"  設定: {self.config_path}")
        print(f"  デバイス: {self.device}")
        print(f"  閾値: {self.threshold}")
        print(f"  最小シーン長: {self.min_scene_duration}フレーム")

    def _load_config(self) -> Dict[str, Any]:
        """モデル設定を読み込み"""
        if not self.config_path.exists():
            print(f"警告: 設定ファイルが見つかりません: {self.config_path}")
            print("デフォルト設定を使用します")
            return {
                'model_type': 'lstm',
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3,
                'no_attention': False,
                'sequence_length': 30
            }

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        if 'model' in config:
            model_config = config['model']
            use_attention = model_config.get('use_attention', True)
            return {
                'model_type': model_config.get('model_type', 'lstm'),
                'hidden_size': model_config.get('hidden_size', 128),
                'num_layers': model_config.get('num_layers', 2),
                'dropout': model_config.get('dropout', 0.3),
                'no_attention': not use_attention,
                'sequence_length': config.get('dataset', {}).get('sequence_length', 30)
            }

        return config

    def _load_model(self) -> PlayClassifierLSTM:
        """学習済みモデルを読み込み"""
        model = PlayClassifierLSTM(
            input_size=34,  # 17 keypoints × 2 coordinates
            hidden_size=self.model_config.get('hidden_size', 128),
            num_layers=self.model_config.get('num_layers', 2),
            dropout=self.model_config.get('dropout', 0.3),
            use_attention=not self.model_config.get('no_attention', False)
        )

        # 重みの読み込み
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # state_dictの取得（チェックポイント形式に対応）
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"  エポック: {checkpoint.get('epoch', 'N/A')}")
            print(f"  Best Val F1: {checkpoint.get('best_val_f1', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model

    def detect_from_exporter(
        self,
        exporter: TrackingExporter,
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """
        TrackingExporterから骨格データを取得してプレーシーンを検出

        Args:
            exporter: TrackingExporterインスタンス（正規化済み）
            show_progress: プログレスバーを表示するか

        Returns:
            (result_df, scenes)のタプル
            - result_df: 予測結果のDataFrame
            - scenes: 検出されたシーン [(start_frame, end_frame), ...]
        """
        # 骨格データを取得
        pose_data, frames = exporter.get_pose_data_for_dataset()

        if len(pose_data) == 0:
            raise DataInputError("exporter", "有効な骨格データがありません")

        print(f"\nデータ形状:")
        print(f"  pose_data: {pose_data.shape}")  # (num_frames, 34)
        print(f"  frames: {frames.shape}")  # (num_frames,)

        # InMemoryPoseSequenceDatasetを作成
        sequence_length = self.model_config.get('sequence_length', 30)

        dataset = MemoryPoseSequenceDataset(
            pose_data=pose_data,
            frames=frames,
            sequence_length=sequence_length,
            stride=1,
            keypoint_features=None
        )

        print(f"\nInMemoryPoseSequenceDataset作成完了:")
        print(f"  総フレーム数: {len(pose_data)}")
        print(f"  シーケンス数: {len(dataset)}")
        print(f"  シーケンス長: {sequence_length}フレーム")

        # 予測実行
        result_df = self._predict(dataset, show_progress)

        # シーン検出
        scenes = self._extract_scenes(result_df)

        return result_df, scenes

    def detect_from_csv(
        self,
        csv_path: str,
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
        """
        CSVファイルから骨格データを読み込んでプレーシーンを検出

        Args:
            csv_path: 正規化済み骨格データのCSVパス
            show_progress: プログレスバーを表示するか

        Returns:
            (result_df, scenes)のタプル
            - result_df: 予測結果のDataFrame
            - scenes: 検出されたシーン [(start_frame, end_frame), ...]
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise DataInputError(str(csv_path), "CSVファイルが存在しません")

        sequence_length = self.model_config.get('sequence_length', 30)

        dataset = CSVPoseSequenceDataset(
            csv_path=str(csv_path),
            label_path=None,
            sequence_length=sequence_length,
            stride=1,
            keypoint_features=None
        )
        
        # 予測実行
        result_df = self._predict(dataset, show_progress)

        # シーン検出
        scenes = self._extract_scenes(result_df)

        return result_df, scenes

    def _predict(
        self,
        dataset,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        データセットに対してバッチ予測を実行

        Args:
            dataset: データセット
            show_progress: プログレスバーを表示するか

        Returns:
            予測結果のDataFrame
        """
        print(f"\n予測開始 (閾値: {self.threshold}, バッチサイズ: {self.batch_size})...")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        frame_probs = {}  # {frame_num: [probs]}

        with torch.no_grad():
            iterator = tqdm(dataloader, desc="予測中") if show_progress else dataloader

            for features_batch, _, metadata_batch in iterator:
                # features_batch: (batch, seq, features)
                features_batch = features_batch.to(self.device)

                # バッチ予測
                outputs = self.model(features_batch)  # (batch, seq, 1)
                probs_batch = outputs.squeeze(-1).cpu().numpy()  # (batch, seq)

                # 1次元になった場合（batch=1かつseq=1）の対応
                if probs_batch.ndim == 0:
                    probs_batch = probs_batch.reshape(1, 1)
                elif probs_batch.ndim == 1:
                    # batch=1の場合: (seq,) → (1, seq)
                    if len(metadata_batch) == 1:
                        probs_batch = probs_batch.reshape(1, -1)
                    # seq=1の場合: (batch,) → (batch, 1)
                    else:
                        probs_batch = probs_batch.reshape(-1, 1)

                # フレームごとに確率を記録
                for batch_idx, metadata in enumerate(metadata_batch):
                    start_frame = metadata['start_frame']
                    probs = probs_batch[batch_idx]

                    for i, prob in enumerate(probs):
                        frame_num = start_frame + i
                        if frame_num not in frame_probs:
                            frame_probs[frame_num] = []
                        frame_probs[frame_num].append(float(prob))

        # 各フレームの確率を平均
        predictions = []
        for frame_num in sorted(frame_probs.keys()):
            avg_prob = np.mean(frame_probs[frame_num])
            prediction = 1 if avg_prob >= self.threshold else 0
            predictions.append({
                'frame': frame_num,
                'probability': avg_prob,
                'prediction': prediction,
                'num_predictions': len(frame_probs[frame_num])
            })

        result_df = pd.DataFrame(predictions)

        # 統計表示
        num_play_frames = np.sum(result_df['prediction'] == 1)
        play_ratio = num_play_frames / len(result_df) * 100 if len(result_df) > 0 else 0

        print(f"\n予測完了:")
        print(f"  プレー中フレーム: {num_play_frames} / {len(result_df)} ({play_ratio:.1f}%)")

        return result_df

    def _extract_scenes(
        self,
        result_df: pd.DataFrame
    ) -> List[Tuple[int, int]]:
        """
        連続したプレー区間を抽出

        Args:
            result_df: 予測結果のDataFrame

        Returns:
            [(start_frame, end_frame), ...] のリスト
        """
        scenes = []
        in_scene = False
        scene_start = None

        for _, row in result_df.iterrows():
            frame = row['frame']
            pred = row['prediction']

            if pred == 1:  # プレー中
                if not in_scene:
                    scene_start = frame
                    in_scene = True
            else:  # プレー外
                if in_scene:
                    # シーン終了
                    if frame - scene_start >= self.min_scene_duration:
                        scenes.append((scene_start, frame - 1))
                    in_scene = False

        # 最後のシーン
        if in_scene and result_df['frame'].iloc[-1] - scene_start >= self.min_scene_duration:
            scenes.append((scene_start, result_df['frame'].iloc[-1]))

        print(f"\nシーン検出結果:")
        print(f"  検出シーン数: {len(scenes)}")
        print(f"  最小シーン長: {self.min_scene_duration}フレーム")

        if scenes:
            print(f"\n主要シーン（最初の10シーン）:")
            for i, (start, end) in enumerate(scenes[:10], 1):
                duration_frames = end - start + 1
                print(f"  シーン{i}: フレーム {start}-{end} ({duration_frames}フレーム)")

        return scenes

    def save_results(
        self,
        result_df: pd.DataFrame,
        scenes: List[Tuple[int, int]],
        output_dir: str,
        base_name: str,
        fps: float = 30.0
    ) -> Dict[str, str]:
        """
        予測結果を保存

        Args:
            result_df: 予測結果のDataFrame
            scenes: 検出されたシーン
            output_dir: 出力ディレクトリ
            base_name: ベース名
            fps: 動画のFPS

        Returns:
            保存したファイルのパス辞書
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        # 予測結果CSVを保存
        predictions_csv = output_dir / f"{base_name}_predictions.csv"
        result_df.to_csv(predictions_csv, index=False)
        output_paths['predictions'] = str(predictions_csv)
        print(f"予測結果CSVを保存: {predictions_csv}")

        # シーン情報をCSV保存
        if scenes:
            scenes_df = pd.DataFrame(scenes, columns=['start_frame', 'end_frame'])
            scenes_df['duration_frames'] = scenes_df['end_frame'] - scenes_df['start_frame'] + 1
            scenes_df['duration_sec'] = scenes_df['duration_frames'] / fps
            scenes_df['start_time_sec'] = scenes_df['start_frame'] / fps
            scenes_df['end_time_sec'] = scenes_df['end_frame'] / fps

            scenes_csv = output_dir / f"{base_name}_scenes.csv"
            scenes_df.to_csv(scenes_csv, index=False)
            output_paths['scenes'] = str(scenes_csv)
            print(f"シーン情報CSVを保存: {scenes_csv}")

        return output_paths
