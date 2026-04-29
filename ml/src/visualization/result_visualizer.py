from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_prediction_graph(
    result_df: pd.DataFrame,
    scenes: list,
    output_dir: Path,
    base_name: str,
    threshold: float,
    fps: float
) -> None:
    """
    予測結果のグラフを保存

    Args:
        result_df: 予測結果のDataFrame
        scenes: 検出されたシーン
        output_dir: 出力ディレクトリ
        base_name: ベース名
        threshold: 判定閾値
        fps: FPS
    """
    print(f"\n予測グラフを作成中...")

    plt.figure(figsize=(16, 6))

    # 予測確率をプロット
    plt.subplot(2, 1, 1)
    plt.plot(result_df['frame'], result_df['probability'], linewidth=1, alpha=0.7, color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    plt.fill_between(
        result_df['frame'],
        0,
        result_df['probability'],
        where=(result_df['probability'] >= threshold),
        alpha=0.3,
        color='green',
        label='Playing'
    )
    plt.xlabel('Frame')
    plt.ylabel('Play Probability')
    plt.title('Play Scene Prediction - Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    # 予測ラベルをプロット
    plt.subplot(2, 1, 2)
    plt.plot(result_df['frame'], result_df['prediction'], linewidth=1.5, color='green')
    plt.fill_between(
        result_df['frame'],
        0,
        result_df['prediction'],
        alpha=0.3,
        color='green'
    )

    # 検出されたシーンを赤線でマーク
    for start, end in scenes:
        plt.axvline(x=start, color='red', linestyle=':', alpha=0.5, linewidth=1)
        plt.axvline(x=end, color='red', linestyle=':', alpha=0.5, linewidth=1)

    plt.xlabel('Frame')
    plt.ylabel('Prediction (0: Not Playing, 1: Playing)')
    plt.title(f'Play Scene Prediction - Classification (Detected Scenes: {len(scenes)})')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['Not Playing', 'Playing'])

    plt.tight_layout()

    output_graph_path = output_dir / f"{base_name}_prediction_graph.png"
    plt.savefig(output_graph_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"予測グラフを保存しました: {output_graph_path}")
