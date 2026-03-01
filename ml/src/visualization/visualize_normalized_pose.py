"""
正規化された骨格データを可視化するスクリプト

正規化後の座標系で骨格を表示し、正規化が正しく行われているかを確認する
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


# COCO 17キーポイントのスケルトン定義（接続関係）
SKELETON = [
    (0, 1), (0, 2),  # nose -> eyes
    (1, 3), (2, 4),  # eyes -> ears
    (0, 5), (0, 6),  # nose -> shoulders
    (5, 6),          # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10), # right arm
    (5, 11), (6, 12),# shoulder -> hip
    (11, 12),        # hips
    (11, 13), (13, 15),# left leg
    (12, 14), (14, 16) # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def visualize_normalized_poses(csv_path: str, num_samples: int = 5, track_id: int = None):
    """
    正規化された骨格データを可視化

    Args:
        csv_path: 正規化されたCSVファイルのパス
        num_samples: 表示するサンプル数
        track_id: 特定のトラッキングIDのみを表示（Noneの場合は全て）
    """
    print(f"CSVファイルを読み込んでいます: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"総レコード数: {len(df)}")
    print(f"トラッキングID: {sorted(df['track_id'].unique().tolist())}")

    # 有効なデータのみを抽出
    df_valid = df[df['is_valid'] == True]
    print(f"有効なレコード数: {len(df_valid)} ({len(df_valid)/len(df)*100:.1f}%)")

    if len(df_valid) == 0:
        print("エラー: 有効なデータがありません")
        return

    # トラッキングIDでフィルタ
    if track_id is not None:
        df_valid = df_valid[df_valid['track_id'] == track_id]
        print(f"トラッキングID={track_id}のレコード数: {len(df_valid)}")

    if len(df_valid) == 0:
        print("エラー: フィルタ後のデータがありません")
        return

    # サンプリング
    sample_indices = np.linspace(0, len(df_valid) - 1, min(num_samples, len(df_valid)), dtype=int)
    samples = df_valid.iloc[sample_indices]

    # プロット（11フレーム以上の場合は2行で表示）
    n_samples = len(samples)
    if n_samples <= 10:
        # 1行で表示
        nrows, ncols = 1, n_samples
    else:
        # 2行で表示
        nrows = 2
        ncols = int(np.ceil(n_samples / 2))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # axesを1次元リストに変換
    if n_samples == 1:
        axes = [axes]
    else:
        # nrows=1の場合も nrows>1の場合も flatten() で1次元化
        axes = np.array(axes).flatten()

    # 使用しないサブプロットを非表示にする
    for i in range(n_samples, nrows * ncols):
        axes[i].axis('off')

    for idx, (ax, (_, row)) in enumerate(zip(axes[:n_samples], samples.iterrows())):
        # キーポイントを抽出
        keypoints = []
        for kp_name in KEYPOINT_NAMES:
            x = row[f'{kp_name}_norm_x']
            y = row[f'{kp_name}_norm_y']
            conf = row[f'{kp_name}_conf']
            keypoints.append((x, y, conf))

        # 骨格を描画
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # スケルトンを描画
        for start_idx, end_idx in SKELETON:
            x1, y1, conf1 = keypoints[start_idx]
            x2, y2, conf2 = keypoints[end_idx]

            # 両方のキーポイントが検出されている場合のみ描画
            if conf1 > 0.3 and conf2 > 0.3:
                ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.6)

        # キーポイントを描画
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                color = 'red' if i == 0 else 'green'  # 鼻を赤で強調
                ax.scatter(x, y, c=color, s=50, alpha=0.8, zorder=3)

        # タイトルと軸設定
        ax.set_title(
            f"Frame {row['frame']}, Track ID={row['track_id']}\n"
            f"Torso Length={row['scale_factor']:.1f}px",
            fontsize=10
        )
        ax.set_xlabel('Normalized X', fontsize=9)
        ax.set_ylabel('Normalized Y', fontsize=9)

        # Y軸を反転（画像座標系に合わせる）
        ax.invert_yaxis()

        # 軸範囲を設定（正規化された座標系）
        ax.set_xlim(-2, 2)
        ax.set_ylim(2, -2)

        # 原点に腰の中心を表示
        ax.scatter(0, 0, c='red', s=100, marker='x', linewidth=3, label='Hip Center (Origin)')

    plt.suptitle('Normalized Pose Data (Hip Center as Origin)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_path = Path(csv_path).parent / 'normalized_pose_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可視化結果を保存しました: {output_path}")

    # 統計情報を表示
    print(f"\nNormalization Statistics:")
    print(f"  Scale Factor (Torso Length):")
    print(f"    Mean: {df_valid['scale_factor'].mean():.2f}px")
    print(f"    Std Dev: {df_valid['scale_factor'].std():.2f}px")
    print(f"    Range: {df_valid['scale_factor'].min():.2f} - {df_valid['scale_factor'].max():.2f}px")

    # 正規化後の座標範囲
    all_x = []
    all_y = []
    for kp_name in KEYPOINT_NAMES:
        all_x.extend(df_valid[f'{kp_name}_norm_x'].values)
        all_y.extend(df_valid[f'{kp_name}_norm_y'].values)

    print(f"\n  Normalized Coordinate Range:")
    print(f"    X: {np.min(all_x):.3f} - {np.max(all_x):.3f}")
    print(f"    Y: {np.min(all_y):.3f} - {np.max(all_y):.3f}")

    plt.show()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Visualize normalized pose data'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='正規化されたCSVファイルパス'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=10,
        help='表示するサンプル数（デフォルト: 5）'
    )
    parser.add_argument(
        '--track-id',
        type=int,
        default=None,
        help='特定のトラッキングIDのみを表示'
    )

    args = parser.parse_args()

    visualize_normalized_poses(
        csv_path=args.input,
        num_samples=args.num_samples,
        track_id=args.track_id
    )


if __name__ == "__main__":
    main()
