"""
サービス検出のテストスクリプト v2
複数人物検出に対応し、対象選手のみを追跡する
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# matplotlibで日本語を表示できるように設定
plt.rcParams['font.family'] = ['Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

from utils.video_loader import VideoLoader
from detection.pose_detector import PoseDetector
from detection.service_detector import ServiceDetector
from detection.multi_person_detector import MultiPersonDetector


def test_service_detection(
    video_path: str,
    sample_rate: int = 3,
    output_video: bool = False,
    roi_bottom_ratio: float = 0.6
):
    """
    サービス検出のテストを実行（複数人物対応版）

    Args:
        video_path: 動画ファイルのパス
        sample_rate: フレームのサンプリングレート
        output_video: 姿勢を描画した動画を出力するか
        roi_bottom_ratio: 画面下部の注目領域の割合（0.0-1.0）
    """
    print(f"動画を読み込んでいます: {video_path}")

    # 動画読み込み
    video = VideoLoader(video_path)
    if not video.open():
        print("エラー: 動画ファイルを開けませんでした")
        sys.exit(1)

    info = video.get_info()
    print(f"\n動画情報:")
    print(f"  解像度: {info['width']}x{info['height']}")
    print(f"  FPS: {info['fps']:.2f}")
    print(f"  フレーム数: {info['frame_count']}")
    print(f"  長さ: {info['duration']:.2f}秒")

    # 検出器の初期化
    pose_detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    service_detector = ServiceDetector()
    person_detector = MultiPersonDetector(roi_bottom_ratio=roi_bottom_ratio)

    # 結果を格納するリスト
    service_data = []
    confidence_data = []
    pose_types = []
    frame_numbers = []
    timestamps = []
    service_moments = []

    # 動画出力の準備
    video_writer = None
    if output_video:
        output_path = Path('output') / f"service_detection_v2_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            info['fps'] / sample_rate,
            (info['width'], info['height'])
        )

    print(f"\nサービス検出を実行中...")
    print(f"  サンプリングレート: {sample_rate}")
    print(f"  ROI（注目領域）: 画面下部{roi_bottom_ratio*100:.0f}%")

    frame_count = 0
    processed_count = 0

    # プログレスバー付きで処理
    with tqdm(total=info['frame_count'], desc="処理中") as pbar:
        while True:
            ret, frame = video.read_frame()
            if not ret:
                break

            # サンプリング
            if frame_count % sample_rate == 0:
                # 対象選手を特定
                target_bbox = person_detector.select_target_person(frame)

                # ROIマスクを作成（対象選手の領域のみ）
                if target_bbox:
                    roi_mask = person_detector.create_roi_mask(frame, target_bbox)
                    # マスク適用したフレームで姿勢検出
                    masked_frame = person_detector.apply_roi_to_frame(frame, roi_mask)
                else:
                    # 対象選手が特定できない場合は画面下部全体
                    roi_mask = person_detector.create_roi_mask(frame, None)
                    masked_frame = person_detector.apply_roi_to_frame(frame, roi_mask)

                # 姿勢検出（マスクされた領域で）
                success, pose, results = pose_detector.detect(masked_frame)

                # サービス姿勢判定
                detection = service_detector.detect_service_pose(pose)

                # サービスシーケンスの検出
                is_service_sequence, seq_confidence = service_detector.detect_service_sequence()

                # 結果を保存
                service_data.append(1 if detection.is_service_pose else 0)
                confidence_data.append(detection.confidence)
                pose_types.append(detection.pose_type)
                frame_numbers.append(frame_count)
                timestamps.append(frame_count / info['fps'])

                # サービスシーケンスが検出された場合
                if is_service_sequence and seq_confidence > 0.5:
                    service_moments.append({
                        'frame': frame_count,
                        'time': frame_count / info['fps'],
                        'confidence': seq_confidence
                    })

                # 動画出力
                if output_video and video_writer:
                    # 元のフレームに描画
                    annotated_frame = frame.copy()

                    # 対象選手のバウンディングボックスとROIを描画
                    person_regions = person_detector.detect_people_regions(frame)
                    annotated_frame = person_detector.draw_roi_and_target(
                        annotated_frame, target_bbox, person_regions
                    )

                    # 姿勢を描画
                    if results and results.pose_landmarks:
                        annotated_frame = pose_detector.draw_landmarks(annotated_frame, results)

                    # サービス検出情報を描画
                    if detection.is_service_pose:
                        text = f"Service: {detection.pose_type} ({detection.confidence:.2f})"
                        cv2.putText(annotated_frame, text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if is_service_sequence:
                        cv2.putText(annotated_frame, "SERVICE DETECTED!", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    video_writer.write(annotated_frame)

                processed_count += 1

            frame_count += 1
            pbar.update(1)

    video.close()
    pose_detector.close()

    if video_writer:
        video_writer.release()
        print(f"\n姿勢描画動画を保存しました: {output_path}")

    print(f"\n処理完了: {processed_count}フレームを分析しました")

    # 統計情報を表示
    service_array = np.array(service_data)
    confidence_array = np.array(confidence_data)

    print(f"\nサービス検出の統計:")
    print(f"  サービス姿勢フレーム: {service_array.sum()}/{len(service_array)} ({service_array.sum()/len(service_array)*100:.1f}%)")
    print(f"  平均信頼度: {confidence_array[service_array > 0].mean():.2f}" if service_array.sum() > 0 else "  平均信頼度: N/A")
    print(f"  検出されたサービス回数: {len(service_moments)}回")

    if service_moments:
        print(f"\nサービスが検出された時刻:")
        for i, moment in enumerate(service_moments, 1):
            print(f"  {i}. {moment['time']:.2f}秒 (フレーム {moment['frame']}, 信頼度: {moment['confidence']:.2f})")

    # グラフ化
    print(f"\nグラフを生成しています...")
    plot_service_graph(timestamps, service_data, confidence_data, pose_types, service_moments, info, video_path)

    return service_data, timestamps, service_moments


def plot_service_graph(timestamps, service_data, confidence_data, pose_types, service_moments, video_info, video_path):
    """サービス検出結果をグラフ化"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    # 1. 信頼度の時系列グラフ
    axes[0].plot(timestamps, confidence_data, linewidth=0.5, alpha=0.7, color='blue')
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='閾値: 0.5')
    axes[0].set_xlabel('時間 (秒)')
    axes[0].set_ylabel('信頼度')
    axes[0].set_title(f'サービス姿勢の信頼度（複数人物対応版）: {Path(video_path).name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. サービス検出の二値化
    axes[1].fill_between(timestamps, service_data, alpha=0.5, color='green')
    axes[1].set_xlabel('時間 (秒)')
    axes[1].set_ylabel('サービス姿勢')
    axes[1].set_title('サービス姿勢の検出（緑=サービス姿勢）')
    axes[1].set_yticks([0, 1], ['通常', 'サービス'])
    axes[1].grid(True, alpha=0.3)

    # 3. 姿勢タイプの分布
    pose_type_map = {"none": 0, "toss": 1, "backswing": 2, "impact": 3}
    pose_type_values = [pose_type_map.get(pt, 0) for pt in pose_types]

    axes[2].scatter(timestamps, pose_type_values, s=10, alpha=0.6, c=pose_type_values, cmap='viridis')
    axes[2].set_xlabel('時間 (秒)')
    axes[2].set_ylabel('姿勢タイプ')
    axes[2].set_title('検出された姿勢タイプ')
    axes[2].set_yticks([0, 1, 2, 3], ['通常', 'トス', 'バックスイング', 'インパクト'])
    axes[2].grid(True, alpha=0.3)

    # サービスが検出された瞬間をマーク
    if service_moments:
        for moment in service_moments:
            for ax in axes:
                ax.axvline(x=moment['time'], color='red', linestyle=':', alpha=0.7, linewidth=2)

    plt.tight_layout()

    # 保存
    output_path = Path('output') / f"service_detection_v2_{Path(video_path).stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"グラフを保存しました: {output_path}")
    plt.close()


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='サービス検出テスト（複数人物対応版）'
    )
    parser.add_argument(
        'input',
        type=str,
        help='入力動画ファイルのパス'
    )
    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        default=3,
        help='フレームのサンプリングレート（デフォルト: 3）'
    )
    parser.add_argument(
        '-o', '--output-video',
        action='store_true',
        help='姿勢を描画した動画を出力する'
    )
    parser.add_argument(
        '-r', '--roi-ratio',
        type=float,
        default=0.6,
        help='ROI（画面下部の注目領域）の割合 0.0-1.0（デフォルト: 0.6）'
    )

    args = parser.parse_args()

    # 入力ファイルチェック
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # サービス検出テストを実行
    test_service_detection(
        str(input_path),
        sample_rate=args.sample_rate,
        output_video=args.output_video,
        roi_bottom_ratio=args.roi_ratio
    )


if __name__ == "__main__":
    main()
