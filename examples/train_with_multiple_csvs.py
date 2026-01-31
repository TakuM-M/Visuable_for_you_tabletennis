"""
複数CSVファイルを使った学習の例

複数の動画から抽出したCSVファイルを統合して学習する3つの方法を紹介
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.datasets import PoseSequenceDataset, MultiCSVPoseDataset, collate_fn
from src.pipelines import TrainingPipeline
from src.training.train_play_classifier import Trainer
from src.models.play_classifier_lstm import PlayClassifierLSTM


# ========================================
# 方法1: MultiCSVPoseDatasetを使用（推奨）
# ========================================
def method1_multi_csv_dataset():
    """
    MultiCSVPoseDatasetを使って複数CSVを統合
    最もシンプルで推奨される方法
    """
    print("=" * 70)
    print("方法1: MultiCSVPoseDatasetを使用")
    print("=" * 70)

    # 訓練用動画のディレクトリ
    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    # 検証用動画のディレクトリ
    val_dirs = [
        'data/detect/sample_video_06_01',
    ]

    # データセット作成
    train_dataset = MultiCSVPoseDataset.from_directories(
        data_dirs=train_dirs,
        csv_filename='original_pose_data.csv',
        label_filename='play_labels.csv',
        sequence_length=30,
        stride=5
    )

    val_dataset = MultiCSVPoseDataset.from_directories(
        data_dirs=val_dirs,
        csv_filename='original_pose_data.csv',
        label_filename='play_labels.csv',
        sequence_length=30,
        stride=5
    )

    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # モデル作成
    model = PlayClassifierLSTM(
        input_size=34,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        use_attention=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Trainerで学習
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=1e-3,
        output_dir='output/training_multi_csv'
    )

    trainer.train(num_epochs=50, save_every=10)

    print("\n✓ 方法1完了")


# ========================================
# 方法2: ConcatDatasetを使用
# ========================================
def method2_concat_dataset():
    """
    PyTorchのConcatDatasetで複数のPoseSequenceDatasetを結合
    より柔軟な組み合わせが可能
    """
    print("=" * 70)
    print("方法2: ConcatDatasetを使用")
    print("=" * 70)

    # 訓練用データセットを個別に作成
    train_datasets = []
    train_videos = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    for video_dir in train_videos:
        dataset = PoseSequenceDataset(
            csv_path=f"{video_dir}/original_pose_data.csv",
            label_path=f"{video_dir}/play_labels.csv",
            sequence_length=30,
            stride=5
        )
        train_datasets.append(dataset)
        print(f"  {Path(video_dir).name}: {len(dataset)} sequences")

    # 統合
    train_dataset = ConcatDataset(train_datasets)
    print(f"\n訓練データ合計: {len(train_dataset)} sequences")

    # 検証用データセット
    val_datasets = []
    val_videos = ['data/detect/sample_video_06_01']

    for video_dir in val_videos:
        dataset = PoseSequenceDataset(
            csv_path=f"{video_dir}/original_pose_data.csv",
            label_path=f"{video_dir}/play_labels.csv",
            sequence_length=30,
            stride=5
        )
        val_datasets.append(dataset)

    val_dataset = ConcatDataset(val_datasets)

    # データローダー作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # モデルと学習（方法1と同じ）
    model = PlayClassifierLSTM()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(device),
        learning_rate=1e-3,
        output_dir='output/training_concat'
    )

    trainer.train(num_epochs=50, save_every=10)

    print("\n✓ 方法2完了")


# ========================================
# 方法3: 事前にCSVを統合
# ========================================
def method3_merge_csv_first():
    """
    事前に複数のCSVファイルを1つに統合してから学習
    最もシンプルだが、前処理が必要
    """
    print("=" * 70)
    print("方法3: 事前にCSVを統合")
    print("=" * 70)

    import pandas as pd

    # 訓練用CSVを統合
    train_videos = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    all_pose_data = []
    all_labels = []

    for video_dir in train_videos:
        # ポーズデータ
        df_pose = pd.read_csv(f"{video_dir}/original_pose_data.csv")
        df_pose['video_source'] = Path(video_dir).name
        all_pose_data.append(df_pose)

        # ラベルデータ
        df_label = pd.read_csv(f"{video_dir}/play_labels.csv")
        df_label['video_source'] = Path(video_dir).name
        all_labels.append(df_label)

    # 統合
    merged_pose = pd.concat(all_pose_data, ignore_index=True)
    merged_labels = pd.concat(all_labels, ignore_index=True)

    # フレーム番号を再採番（動画ごとにユニークにする）
    # video_sourceごとにグループ化してframe番号を振り直す
    merged_pose['original_frame'] = merged_pose['frame']
    merged_labels['original_frame'] = merged_labels['frame']

    offset = 0
    for video_name in merged_pose['video_source'].unique():
        mask = merged_pose['video_source'] == video_name
        video_frames = merged_pose.loc[mask, 'original_frame']
        merged_pose.loc[mask, 'frame'] = video_frames + offset

        mask_label = merged_labels['video_source'] == video_name
        label_frames = merged_labels.loc[mask_label, 'original_frame']
        merged_labels.loc[mask_label, 'frame'] = label_frames + offset

        offset += video_frames.max() + 1000  # 十分な間隔を空ける

    # 一時ファイルに保存
    temp_dir = Path('output/temp_merged_data')
    temp_dir.mkdir(parents=True, exist_ok=True)

    merged_pose.to_csv(temp_dir / 'train_poses_merged.csv', index=False)
    merged_labels.to_csv(temp_dir / 'train_labels_merged.csv', index=False)

    print(f"統合完了:")
    print(f"  ポーズデータ: {len(merged_pose)} 行")
    print(f"  ラベルデータ: {len(merged_labels)} 行")

    # TrainingPipelineで学習
    pipeline = TrainingPipeline.create_default(
        train_csv=str(temp_dir / 'train_poses_merged.csv'),
        train_labels=str(temp_dir / 'train_labels_merged.csv'),
        val_csv='data/detect/sample_video_06_01/original_pose_data.csv',
        val_labels='data/detect/sample_video_06_01/play_labels.csv',
        output_dir='output/training_merged',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    results = pipeline.run()

    print(f"\n✓ 方法3完了")
    print(f"  Best F1: {results['best_val_f1']:.4f}")


# ========================================
# メイン
# ========================================
def main():
    """実行例"""
    print("\n複数CSVファイルを使った学習の3つの方法\n")

    # どれか1つを選んで実行
    # method1_multi_csv_dataset()  # 推奨
    # method2_concat_dataset()
    # method3_merge_csv_first()

    # または統計情報だけ表示
    print("=" * 70)
    print("データセットの統計情報を表示")
    print("=" * 70)

    train_dirs = [
        'data/detect/sample_video_03_short',
        'data/detect/sample_video_04_short',
        'data/detect/sample_video_05_02',
    ]

    dataset = MultiCSVPoseDataset.from_directories(
        data_dirs=train_dirs,
        sequence_length=30,
        stride=5
    )

    stats = dataset.get_statistics()
    print(f"\n統計情報:")
    print(f"  総シーケンス数: {stats['total_sequences']}")
    print(f"  動画数: {stats['num_videos']}")
    print(f"  シーケンス長: {stats['sequence_length']}")
    print(f"  ストライド: {stats['stride']}")

    print(f"\n各動画の詳細:")
    for i, video_stats in enumerate(stats['videos']):
        video_name = Path(video_stats['csv_path']).parent.name
        print(f"  [{i+1}] {video_name}")
        print(f"      シーケンス数: {video_stats['num_sequences']}")


if __name__ == "__main__":
    main()
