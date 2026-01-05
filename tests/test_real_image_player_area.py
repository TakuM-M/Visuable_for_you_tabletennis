"""
実際の卓球試合画像でプレイヤー領域を検出するテスト

sample_video_01_01_frame_00000.jpg を使用して、
実際の卓球台の座標からプレイヤー領域を検出します。
"""

import sys
import os
import cv2
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.player_area_detector import PlayerAreaDetector
from src.utils.homography import TableHomography


def main():
    """メイン処理"""
    print("=" * 60)
    print("実際の画像でプレイヤー領域検出")
    print("=" * 60)

    # 実際の画像を読み込み
    image_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'annotations',
        'images',
        'sample_video_01_01_frame_00000.jpg'
    )

    print(f"\n1. 画像を読み込み中: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print(f"エラー: 画像が読み込めません: {image_path}")
        return

    print(f"   画像サイズ: {img.shape[1]}x{img.shape[0]} px")

    # アノテーションデータから卓球台の四隅の座標を取得
    # 座標は640x360の画像に対するもの
    # keypointsの順番: id 0=左手前, 1=右手前, 2=右奥, 3=左奥
    # PlayerAreaDetectorが期待する順番: 左上、右上、右下、左下（反時計回り）
    # カメラから見て: 左上=左奥、右上=右奥、右下=右手前、左下=左手前
    table_corners_640x360 = [
        (288.4325, 148.5270),  # id 3: 左奥 → 左上
        (395.8707, 148.3543),  # id 2: 右奥 → 右上
        (331.4037, 164.8601),  # id 1: 右手前 → 右下
        (184.8551, 163.5938),  # id 0: 左手前 → 左下
    ]

    # 画像サイズに合わせてスケーリング
    scale_x = img.shape[1] / 640.0
    scale_y = img.shape[0] / 360.0

    table_corners = [
        (x * scale_x, y * scale_y) for x, y in table_corners_640x360
    ]

    print(f"\n2. 卓球台の四隅の座標 (スケーリング後):")
    labels = ["左奥", "右奥", "右手前", "左手前"]
    for i, (corner, label) in enumerate(zip(table_corners, labels)):
        print(f"   {label}: ({corner[0]:.1f}, {corner[1]:.1f})")

    # プレイヤー領域検出器を初期化（実寸法ベース）
    print("\n3. プレイヤー領域検出器を初期化中...")
    print(f"   卓球台の実寸法: 幅{PlayerAreaDetector.TABLE_WIDTH_CM}cm x 奥行き{PlayerAreaDetector.TABLE_DEPTH_CM}cm")
    player_detector = PlayerAreaDetector(
        table_corners=table_corners,
        player_area_depth_cm=200.0,         # プレイヤー領域の奥行き: 200cm (2m)
        player_area_side_margin_cm=100.0    # 左右のマージン: 100cm (1m)
    )
    print(f"   プレイヤー領域設定: 奥行き200cm、左右マージン100cm")

    # プレイヤー領域を取得
    print("\n4. プレイヤー領域を計算中...")
    player_areas = player_detector.get_both_player_areas(return_format="xyxy")

    print(f"   手前側プレイヤー領域 (near):")
    x1, y1, x2, y2 = player_areas["near"]
    print(f"      座標: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
    print(f"      サイズ: {x2-x1:.1f} x {y2-y1:.1f} px")

    print(f"   奥側プレイヤー領域 (far):")
    x1, y1, x2, y2 = player_areas["far"]
    print(f"      座標: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f})")
    print(f"      サイズ: {x2-x1:.1f} x {y2-y1:.1f} px")

    # プレイヤー領域を可視化（卓球台の辺に沿った形状）
    print("\n5. プレイヤー領域を可視化中...")
    img_with_areas = player_detector.visualize_player_areas(
        img,
        show_both=True,
        near_color=(0, 255, 0),   # 手前側: 緑
        far_color=(255, 0, 0),     # 奥側: 青
        alpha=0.3,
        use_polygon=True  # 卓球台の辺に沿った四角形を描画
    )

    # 比較用：軸に平行なバウンディングボックス版も作成
    img_with_bbox = player_detector.visualize_player_areas(
        img,
        show_both=True,
        near_color=(0, 255, 0),
        far_color=(255, 0, 0),
        alpha=0.3,
        use_polygon=False  # 軸に平行なバウンディングボックス
    )

    # 卓球台の四隅をマーク
    img_marked = img.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # BGR
    marker_labels = ["LF", "RF", "RN", "LN"]  # Left-Far, Right-Far, Right-Near, Left-Near

    for corner, color, marker_label in zip(table_corners, colors, marker_labels):
        x, y = int(corner[0]), int(corner[1])
        cv2.circle(img_marked, (x, y), 8, color, -1)
        cv2.putText(
            img_marked,
            marker_label,
            (x + 12, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # プレイヤー領域のバウンディングボックスを描画
    for side, color in [("near", (0, 255, 0)), ("far", (255, 0, 0))]:
        x1, y1, x2, y2 = player_areas[side]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_marked, (x1, y1), (x2, y2), color, 3)

        label = f"Player Area ({side})"
        label_y = y1 - 10 if side == "far" else y2 + 25
        cv2.putText(
            img_marked,
            label,
            (x1 + 10, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # ホモグラフィ変換も実行
    print("\n6. ホモグラフィ変換を実行中...")
    homography = TableHomography(
        table_corners=table_corners,
        table_width_mm=1525.0,
        table_height_mm=2740.0,
        output_scale=0.3
    )

    warped = homography.warp_image(img)
    output_size = homography.get_output_size()
    print(f"   鳥瞰図サイズ: {output_size[0]}x{output_size[1]} px")

    # 結果を保存
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    real_player_areas_path = os.path.join(output_dir, 'real_image_player_areas_polygon.jpg')
    real_bbox_path = os.path.join(output_dir, 'real_image_player_areas_bbox.jpg')
    real_marked_path = os.path.join(output_dir, 'real_image_with_corners.jpg')
    real_warped_path = os.path.join(output_dir, 'real_image_warped.jpg')

    cv2.imwrite(real_player_areas_path, img_with_areas)
    cv2.imwrite(real_bbox_path, img_with_bbox)
    cv2.imwrite(real_marked_path, img_marked)
    cv2.imwrite(real_warped_path, warped)

    print(f"\n7. 結果を保存しました:")
    print(f"   プレイヤー領域（ポリゴン）: {real_player_areas_path}")
    print(f"   プレイヤー領域（バウンディングボックス）: {real_bbox_path}")
    print(f"   卓球台コーナー表示: {real_marked_path}")
    print(f"   鳥瞰図: {real_warped_path}")

    # 画像を表示
    print("\n画像を表示しています（任意のキーを押すと終了）...")

    # ウィンドウサイズを調整して表示
    display_scale = 1.5  # 640x360の画像を拡大表示
    img_areas_display = cv2.resize(
        img_with_areas,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img_bbox_display = cv2.resize(
        img_with_bbox,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img_marked_display = cv2.resize(
        img_marked,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_LINEAR
    )

    # 鳥瞰図は適切なサイズで表示
    warped_display = cv2.resize(
        warped,
        None,
        fx=1.0,
        fy=1.0,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow('Player Areas (Polygon - Aligned with Table)', img_areas_display)
    cv2.imshow('Player Areas (Bounding Box - Axis Aligned)', img_bbox_display)
    cv2.imshow('Table Corners Marked', img_marked_display)
    cv2.imshow('Bird\'s Eye View', warped_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n完了!")


if __name__ == "__main__":
    main()
