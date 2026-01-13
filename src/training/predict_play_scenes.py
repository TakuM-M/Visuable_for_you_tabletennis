"""
プレー検知モデルの推論スクリプト

学習済みモデルを使って動画からプレーシーンを検出する
"""
import argparse
import sys
from pathlib import Path
import json

import torch
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.models.play_classifier import PlayClassifierLSTM, PlayClassifierCNNLSTM
from src.data.dataset import PoseSequenceDataset


class PlayScenePredictor:
    """プレーシーン検出器"""

    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = 'cuda',
        threshold: float = 0.5
    ):
        """
        初期化

        Args:
            model_path: 学習済みモデルのパス
            config_path: 学習時の設定ファイル（Noneの場合は自動推測）
            device: 使用デバイス
            threshold: プレー検出の閾値
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device)
        self.threshold = threshold

        # 設定ファイル読み込み
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # model_pathと同じディレクトリから探す
            config_path = self.model_path.parent / 'config.json'
            if config_path.exists():
                self.config = self._load_config(str(config_path))
            else:
                # デフォルト設定
                self.config = {
                    'model_type': 'lstm',
                    'hidden_size': 128,
                    'num_layers': 2,
                    'dropout': 0.3,
                    'no_attention': False,
                    'sequence_length': 30
                }
                print(f"警告: 設定ファイルが見つかりません。デフォルト設定を使用します。")

        # モデル読み込み
        self.model = self._load_model()
        print(f"モデル読み込み完了: {self.model_path}")

    def _load_config(self, config_path: str) -> dict:
        """設定ファイル読み込み"""
        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_model(self) -> torch.nn.Module:
        """モデル読み込み"""
        # モデル作成
        if self.config['model_type'] == 'lstm':
            model = PlayClassifierLSTM(
                input_size=34,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
                use_attention=not self.config.get('no_attention', False)
            )
        else:  # cnn_lstm
            model = PlayClassifierCNNLSTM(
                input_size=34,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout']
            )

        # 重み読み込み
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: str = None,
        visualize: bool = False
    ) -> pd.DataFrame:
        """
        CSVファイルから予測

        Args:
            csv_path: 正規化された骨格データのCSVパス
            output_path: 出力CSVパス（Noneの場合は入力ファイル名に_predictionsを付加）
            visualize: 予測結果を可視化するか

        Returns:
            予測結果のDataFrame
        """
        csv_path = Path(csv_path)

        # データセット作成
        sequence_length = self.config.get('sequence_length', 30)
        dataset = PoseSequenceDataset(
            csv_path=str(csv_path),
            label_path=None,  # ラベルなし
            sequence_length=sequence_length,
            stride=1  # 全フレームを予測するためstride=1
        )

        print(f"\n予測開始:")
        print(f"  入力CSV: {csv_path}")
        print(f"  総フレーム数: {len(dataset.data_df)}")
        print(f"  シーケンス数: {len(dataset)}")

        # 各フレームの予測確率を集計（複数のシーケンスで同じフレームが予測される）
        frame_probs = {}  # {frame_num: [probs]}

        with torch.no_grad():
            for features, _, metadata in tqdm(dataset, desc="予測中"):
                # バッチ次元を追加
                features = features.unsqueeze(0).to(self.device)

                # 予測
                outputs = self.model(features)  # (1, seq, 1)
                probs = outputs.squeeze().cpu().numpy()  # (seq,)

                # フレームごとに確率を記録
                start_frame = metadata['start_frame']
                for i, prob in enumerate(probs):
                    frame_num = start_frame + i
                    if frame_num not in frame_probs:
                        frame_probs[frame_num] = []
                    frame_probs[frame_num].append(prob)

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

        # 統計
        num_play_frames = np.sum(result_df['prediction'] == 1)
        play_ratio = num_play_frames / len(result_df) * 100
        print(f"\n予測結果:")
        print(f"  プレー中フレーム: {num_play_frames} / {len(result_df)} ({play_ratio:.1f}%)")

        # シーン検出（連続したプレー区間を抽出）
        scenes = self._extract_scenes(result_df)
        print(f"  検出シーン数: {len(scenes)}")
        for i, (start, end) in enumerate(scenes[:10], 1):  # 最初の10シーンを表示
            duration = (end - start) / 30.0  # 30fps想定
            print(f"    シーン{i}: フレーム {start}-{end} ({duration:.1f}秒)")

        # 保存
        if output_path is None:
            output_path = csv_path.parent / f"{csv_path.stem}_predictions.csv"
        result_df.to_csv(output_path, index=False)
        print(f"\n予測結果保存: {output_path}")

        # シーン情報も保存
        scenes_path = Path(str(output_path).replace('.csv', '_scenes.csv'))
        scenes_df = pd.DataFrame(scenes, columns=['start_frame', 'end_frame'])
        scenes_df['duration_sec'] = (scenes_df['end_frame'] - scenes_df['start_frame']) / 30.0
        scenes_df.to_csv(scenes_path, index=False)
        print(f"シーン情報保存: {scenes_path}")

        # 可視化
        if visualize:
            self._visualize_predictions(result_df, csv_path.stem)

        return result_df

    def _extract_scenes(
        self,
        result_df: pd.DataFrame,
        min_duration: int = 10
    ) -> list:
        """
        連続したプレー区間を抽出

        Args:
            result_df: 予測結果のDataFrame
            min_duration: 最小シーン長（フレーム数）

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
                    if frame - scene_start >= min_duration:
                        scenes.append((scene_start, frame - 1))
                    in_scene = False

        # 最後のシーン
        if in_scene and result_df['frame'].iloc[-1] - scene_start >= min_duration:
            scenes.append((scene_start, result_df['frame'].iloc[-1]))

        return scenes

    def _visualize_predictions(self, result_df: pd.DataFrame, title: str):
        """予測結果を可視化"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 5))

        # 確率のプロット
        plt.subplot(2, 1, 1)
        plt.plot(result_df['frame'], result_df['probability'], linewidth=0.5, alpha=0.7)
        plt.axhline(y=self.threshold, color='r', linestyle='--', label=f'閾値 ({self.threshold})')
        plt.ylabel('プレー中の確率')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'プレー検知結果: {title}')

        # 予測のプロット
        plt.subplot(2, 1, 2)
        plt.fill_between(
            result_df['frame'],
            result_df['prediction'],
            alpha=0.5,
            label='プレー中'
        )
        plt.ylabel('プレー中 (0/1)')
        plt.xlabel('フレーム番号')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = Path('output') / f'{title}_prediction_visualization.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150)
        print(f"可視化保存: {output_path}")
        plt.close()

    def predict_from_video(
        self,
        video_path: str,
        pose_csv_path: str,
        output_video_path: str = None
    ):
        """
        動画に予測結果を重畳表示

        Args:
            video_path: 元動画のパス
            pose_csv_path: 骨格データCSVのパス
            output_video_path: 出力動画のパス
        """
        # 予測
        result_df = self.predict_from_csv(pose_csv_path)

        # 動画読み込み
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 出力動画設定
        if output_video_path is None:
            output_video_path = Path('output') / f"{Path(video_path).stem}_predicted.mp4"
        output_video_path.parent.mkdir(exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        print(f"\n動画処理中...")
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc="動画書き込み")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 予測結果を取得
            if frame_idx < len(result_df):
                prob = result_df.iloc[frame_idx]['probability']
                pred = result_df.iloc[frame_idx]['prediction']

                # 画面に表示
                color = (0, 255, 0) if pred == 1 else (128, 128, 128)
                text = f"Play: {prob:.2f}" if pred == 1 else f"Non-play: {1-prob:.2f}"

                cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
                cv2.putText(frame, text, (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        out.release()

        print(f"出力動画保存: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description='プレー検知モデルの推論')

    # モデル
    parser.add_argument('--model', type=str, required=True,
                        help='学習済みモデルのパス')
    parser.add_argument('--config', type=str, default=None,
                        help='学習時の設定ファイル')

    # 入力
    parser.add_argument('--csv', type=str, required=True,
                        help='正規化された骨格データCSV')
    parser.add_argument('--video', type=str, default=None,
                        help='元動画のパス（動画出力する場合）')

    # 出力
    parser.add_argument('--output', type=str, default=None,
                        help='出力CSVパス')
    parser.add_argument('--output-video', type=str, default=None,
                        help='出力動画パス')

    # パラメータ
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='プレー検出の閾値')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='使用デバイス')
    parser.add_argument('--visualize', action='store_true',
                        help='予測結果を可視化')

    args = parser.parse_args()

    # デバイス設定
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        args.device = 'cpu'

    # 予測器作成
    predictor = PlayScenePredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        threshold=args.threshold
    )

    # CSV予測
    result_df = predictor.predict_from_csv(
        csv_path=args.csv,
        output_path=args.output,
        visualize=args.visualize
    )

    # 動画出力（オプション）
    if args.video:
        predictor.predict_from_video(
            video_path=args.video,
            pose_csv_path=args.csv,
            output_video_path=args.output_video
        )


if __name__ == "__main__":
    main()
