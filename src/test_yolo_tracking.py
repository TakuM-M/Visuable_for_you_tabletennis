"""
YOLOv11-Pose トラッキングシステムのテストスクリプト
卓球台検出 → YOLOトラッキング → 選手フィルタリング → CSV出力
"""
import argparse
import sys
from pathlib import Path
import cv2
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from utils.video_loader import VideoLoader
from detection.table_detector import TableDetector
from detection.yolo_tracker import YOLOPoseTracker
from detection.player_filter import PlayerFilter
from detection.tracking_exporter import TrackingExporter


def test_yolo_tracking(
    video_path: str,
    output_dir: str = "output",
    sample_rate: int = 1,
    output_video: bool = False,
    conf_threshold: float = 0.5,
    device: str = "cpu"
):
    """
    YOLOトラッキングシステムのテスト実行

    Args:
        video_path: 入力動画ファイルのパス
        output_dir: 出力ディレクトリ
        sample_rate: フレームのサンプリングレート
        output_video: トラッキング結果の動画を出力するか
        conf_threshold: YOLO検出信頼度の閾値
        device: 使用デバイス（"cpu" or "cuda"）
    """
    print(f"=== YOLOトラッキングシステム ===")
    print(f"動画を読み込んでいます: {video_path}\n")

    # 出力ディレクトリの作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 動画読み込み
    video = VideoLoader(video_path)
    if not video.open():
        print("エラー: 動画ファイルを開けませんでした")
        sys.exit(1)

    info = video.get_info()
    print(f"動画情報:")
    print(f"  解像度: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  フレーム数: {info['frame_count']}")
    print(f"  長さ: {info['duration']:.2f}秒\n")

    # ===== Phase 1: 卓球台検出 =====
    print(f"Phase 1: 卓球台検出中...")
    table_detector = TableDetector()
    table_region = table_detector.get_stable_table_region(
        video,
        num_frames=30,
        sample_interval=10
    )

    if table_region is None:
        print("警告: 卓球台を検出できませんでした。デフォルト設定で続行します。")
        # デフォルトの卓球台領域を設定
        import numpy as np
        from detection.table_detector import TableRegion
        height, width = info['height'], info['width']
        table_region = TableRegion(
            corners=np.array([
                [width * 0.2, height * 0.3],
                [width * 0.8, height * 0.3],
                [width * 0.8, height * 0.5],
                [width * 0.2, height * 0.5]
            ], dtype=np.int32),
            center=(width * 0.5, height * 0.4),
            width=width * 0.6,
            height=height * 0.2
        )
    else:
        print(f"卓球台を検出しました:")
        print(f"  中心座標: ({table_region.center[0]:.1f}, {table_region.center[1]:.1f})")
        print(f"  サイズ: {table_region.width:.1f} x {table_region.height:.1f}\n")

    # ===== Phase 2: YOLOトラッカーの初期化 =====
    print(f"Phase 2: YOLOトラッカーを初期化中...")
    tracker = YOLOPoseTracker(
        model_path="yolo11n-pose.pt",
        conf_threshold=conf_threshold,
        device=device
    )
    print()

    # ===== Phase 3: 選手フィルタとエクスポータの初期化 =====
    print(f"Phase 3: 選手フィルタとCSVエクスポータを初期化...")
    player_filter = PlayerFilter(table_region)
    exporter = TrackingExporter()
    print()

    # 動画出力の準備
    video_writer = None
    if output_video:
        output_video_path = output_path / f"tracking_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            info['fps'] / sample_rate,
            (info['width'], info['height'])
        )

    # ===== Phase 4: トラッキング実行 =====
    print(f"Phase 4: トラッキング実行中...")
    print(f"  サンプリングレート: {sample_rate}")
    print(f"  卓球台Y閾値: {table_region.get_y_threshold():.1f}\n")

    frame_count = 0
    processed_count = 0

    # 動画を先頭に戻す
    video.reset()

    # プログレスバー付きで処理
    with tqdm(total=info['frame_count'], desc="トラッキング中") as pbar:
        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            # サンプリング
            if frame_count % sample_rate == 0:
                timestamp = frame_count / info['fps']

                # YOLOトラッキング
                persons = tracker.track_frame(frame)

                # 選手分類
                classification = player_filter.classify_players(frame_count, persons)

                # CSVデータに追加
                exporter.add_frame(frame_count, timestamp, persons)

                # 動画出力
                if output_video and video_writer:
                    # トラッキング結果を描画
                    annotated_frame = frame.copy()

                    # 卓球台を描画
                    annotated_frame = table_detector.draw_table_region(
                        annotated_frame, table_region
                    )

                    # 選手分類を描画
                    annotated_frame = player_filter.draw_classification(
                        annotated_frame, classification
                    )

                    # YOLOトラッキングを描画（キーポイントとスケルトン）
                    annotated_frame = tracker.draw_tracking(
                        annotated_frame,
                        persons,
                        draw_bbox=False,  # バウンディングボックスは選手分類で描画済み
                        draw_keypoints=True,
                        draw_skeleton=True,
                        draw_id=False  # IDは選手分類で描画済み
                    )

                    video_writer.write(annotated_frame)

                processed_count += 1

            frame_count += 1
            pbar.update(1)

    video.close()

    if video_writer:
        video_writer.release()
        print(f"\nトラッキング動画を保存しました: {output_video_path}")

    print(f"\n処理完了: {processed_count}フレームを分析しました\n")

    # ===== Phase 5: 選手役割の決定 =====
    print(f"Phase 5: 選手役割を決定中...")
    player_roles = player_filter.assign_player_roles()

    # 統計情報を表示
    filter_stats = player_filter.get_statistics(player_roles)
    print(f"\n選手フィルタリング統計:")
    print(f"  総トラッキングID数: {filter_stats['total_tracks']}")
    print(f"  手前選手ID: {filter_stats['near_player_ids']}")
    print(f"  相手選手ID: {filter_stats['far_player_ids']}")
    print(f"  その他ID: {filter_stats['other_ids']}")
    print(f"  手前選手検出率: {filter_stats['near_player_ratio']*100:.1f}%")
    print(f"  相手選手検出率: {filter_stats['far_player_ratio']*100:.1f}%\n")

    # ===== Phase 6: CSV出力 =====
    print(f"Phase 6: CSV出力中...")

    # ベースファイル名
    base_name = Path(video_path).stem

    # 全トラッキングデータのCSV
    all_csv_path = output_path / f"tracking_all_{base_name}.csv"
    exporter.export_csv(str(all_csv_path), player_roles)

    # 選手のみのCSV
    players_csv_path = output_path / f"tracking_players_{base_name}.csv"
    exporter.export_player_csv(str(players_csv_path), player_roles)

    # 選手別のCSV（手前と相手を分離）
    exporter.export_separate_player_csvs(
        str(output_path),
        player_roles,
        base_filename=f"tracking_{base_name}"
    )

    # サマリー出力
    summary_path = output_path / f"tracking_summary_{base_name}.txt"
    exporter.export_summary(str(summary_path), player_roles)

    print(f"\n=== 完了 ===")
    print(f"出力ファイル:")
    print(f"  - 全データCSV: {all_csv_path}")
    print(f"  - 選手データCSV: {players_csv_path}")
    print(f"  - 手前選手CSV: {output_path / f'tracking_{base_name}_near.csv'}")
    print(f"  - 相手選手CSV: {output_path / f'tracking_{base_name}_far.csv'}")
    print(f"  - サマリー: {summary_path}")
    if output_video:
        print(f"  - トラッキング動画: {output_video_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='YOLOv11-Poseトラッキングシステムのテスト'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力動画ファイルのパス'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='output',
        help='出力ディレクトリ（デフォルト: output）'
    )
    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        default=1,
        help='フレームのサンプリングレート（デフォルト: 1）'
    )
    parser.add_argument(
        '-v', '--output-video',
        action='store_true',
        help='トラッキング結果の動画を出力する'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='YOLO検出信頼度の閾値（デフォルト: 0.5）'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='使用デバイス（デフォルト: cpu）'
    )

    args = parser.parse_args()

    # 入力ファイルチェック
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # テスト実行
    test_yolo_tracking(
        str(input_path),
        output_dir=args.output,
        sample_rate=args.sample_rate,
        output_video=args.output_video,
        conf_threshold=args.conf,
        device=args.device
    )


if __name__ == "__main__":
    main()
