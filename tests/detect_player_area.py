"""
卓球台の座標情報からプレイヤーのプレー領域を検出するデモ

卓球台の四隅の座標を基に、プレイヤーが立つ可能性の高い領域を
バウンディングボックスとして推定し、可視化します。
"""
# sample01の卓球台の4角の座標
# {
#     "boxes": [
#         {
#             "id": "l",
#             "label": "table",
#             "x": "291.2374",
#             "y": "156.0207",
#             "width": "230.8745",
#             "height": "48.4274",
#             "keypoints": [
#                 {
#                     "id": 0,
#                     "x": "184.8551",
#                     "y": "163.5938"
#                 },
#                 {
#                     "id": 1,
#                     "x": "331.4037",
#                     "y": "164.8601"
#                 },
#                 {
#                     "id": 2,
#                     "x": "395.8707",
#                     "y": "148.3543"
#                 },
#                 {
#                     "id": 3,
#                     "x": "288.4325",
#                     "y": "148.5270"
#                 }
#             ],
#             "flipY": true
#         }
#     ],
#     "height": 360,
#     "key": "sample_video_01_01_frame_00000.jpg",
#     "width": 640
# }

import sys
import os
import cv2
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.homography import TableHomography
from src.utils.player_area_detector import PlayerAreaDetector


def create_sample_image(width: int = 1920, height: int = 1080) -> np.ndarray:
    """
    サンプル画像を生成（グリッド付き）

    Parameters
    ----------
    width : int
        画像の幅
    height : int
        画像の高さ

    Returns
    -------
    np.ndarray
        サンプル画像
    """
    # 白背景の画像を作成
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # グリッド線を描画
    grid_size = 50
    for i in range(0, width, grid_size):
        cv2.line(img, (i, 0), (i, height), (200, 200, 200), 1)
    for i in range(0, height, grid_size):
        cv2.line(img, (0, i), (width, i), (200, 200, 200), 1)

    return img


def draw_table_on_image(img: np.ndarray, corners: list) -> np.ndarray:
    """
    画像上に卓球台を描画

    Parameters
    ----------
    img : np.ndarray
        画像
    corners : list
        卓球台の四隅の座標

    Returns
    -------
    np.ndarray
        卓球台を描画した画像
    """
    img_with_table = img.copy()

    # 卓球台の輪郭を描画
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(img_with_table, [pts], True, (0, 128, 0), 3)

    # 卓球台を緑色で塗りつぶし（半透明）
    overlay = img_with_table.copy()
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    img_with_table = cv2.addWeighted(img_with_table, 0.7, overlay, 0.3, 0)

    # 中央線を描画
    mid_left = ((corners[0][0] + corners[3][0]) // 2, (corners[0][1] + corners[3][1]) // 2)
    mid_right = ((corners[1][0] + corners[2][0]) // 2, (corners[1][1] + corners[2][1]) // 2)
    cv2.line(img_with_table, mid_left, mid_right, (255, 255, 255), 2)

    return img_with_table


def main():
    """メイン処理"""
    print("=" * 60)
    print("プレイヤー領域検出デモ")
    print("=" * 60)

    # サンプル画像を作成
    print("\n1. サンプル画像を生成中...")
    img = create_sample_image(1920, 1080)

    # 卓球台の四隅の座標を定義（カメラ視点）
    # 例: カメラが斜め上から卓球台を見ている場合
    # 左上、右上、右下、左下の順（反時計回り）
    table_corners = [
        (400, 300),   # 左上
        (1520, 280),  # 右上
        (1700, 850),  # 右下
        (220, 870)    # 左下
    ]

    print(f"2. 卓球台の四隅の座標:")
    for i, corner in enumerate(table_corners):
        labels = ["左上", "右上", "右下", "左下"]
        print(f"   {labels[i]}: {corner}")

    # 卓球台を描画
    img = draw_table_on_image(img, table_corners)

    # プレイヤー領域検出器を初期化
    print("\n3. プレイヤー領域検出器を初期化中...")
    player_detector = PlayerAreaDetector(
        table_corners=table_corners,
        margin_factor=0.8,           # 卓球台の奥行きの80%の距離
        side_extension_factor=0.3    # 卓球台の幅の30%を左右に拡張
    )

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

    # プレイヤー領域を可視化
    print("\n5. プレイヤー領域を可視化中...")
    img_with_areas = player_detector.visualize_player_areas(
        img,
        show_both=True,
        near_color=(0, 255, 0),   # 手前側: 緑
        far_color=(255, 0, 0),     # 奥側: 青
        alpha=0.3
    )

    # ホモグラフィ変換を初期化
    print("\n6. ホモグラフィ変換を初期化中...")
    homography = TableHomography(
        table_corners=table_corners,
        table_width_mm=1525.0,   # 卓球台の公式幅
        table_height_mm=2740.0,  # 卓球台の公式長さ
        output_scale=0.3         # 1mm = 0.3px
    )

    output_size = homography.get_output_size()
    print(f"   出力画像サイズ: {output_size[0]}x{output_size[1]} px")

    # 変換の可視化
    print("\n7. 変換を実行中...")
    img_marked, warped = homography.visualize_transformation(img)

    # テスト: いくつかの座標を変換
    print("\n8. 座標変換のテスト:")
    test_points = [
        (table_corners[0], "左上"),
        (table_corners[1], "右上"),
        ((960, 540), "画像中心付近"),
    ]

    for point, label in test_points:
        transformed = homography.transform_point(point)
        print(f"   {label}: {point} -> ({transformed[0]:.1f}, {transformed[1]:.1f})")

    # 選手がいる領域を想定した点を変換とテスト
    print("\n9. 選手位置の変換とプレイヤー領域判定テスト:")
    # 卓球台の手前側（下側）に選手がいると仮定
    near_player_point = ((table_corners[2][0] + table_corners[3][0]) // 2, table_corners[3][1] + 200)

    # 手前側の選手がプレイヤー領域内にいるかチェック
    is_in_near_area = player_detector.is_point_in_player_area(near_player_point, "near")
    print(f"   手前側選手位置: {near_player_point}")
    print(f"      プレイヤー領域内: {is_in_near_area}")

    # 画像に選手位置をマーク
    cv2.circle(img_marked, near_player_point, 15, (255, 0, 255), -1)
    cv2.putText(
        img_marked,
        "Near Player",
        (near_player_point[0] - 50, near_player_point[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    # 奥側の選手も追加
    far_player_point = ((table_corners[0][0] + table_corners[1][0]) // 2, table_corners[0][1] - 200)
    is_in_far_area = player_detector.is_point_in_player_area(far_player_point, "far")
    print(f"   奥側選手位置: {far_player_point}")
    print(f"      プレイヤー領域内: {is_in_far_area}")

    cv2.circle(img_marked, far_player_point, 15, (255, 165, 0), -1)
    cv2.putText(
        img_marked,
        "Far Player",
        (far_player_point[0] - 50, far_player_point[1] + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 165, 0),
        2
    )

    # プレイヤー領域を含む画像にも選手位置をマーク
    cv2.circle(img_with_areas, near_player_point, 15, (255, 0, 255), -1)
    cv2.circle(img_with_areas, far_player_point, 15, (255, 165, 0), -1)

    transformed_near_player = homography.transform_point(near_player_point)
    transformed_far_player = homography.transform_point(far_player_point)
    print(f"   手前側選手 (変換後): ({transformed_near_player[0]:.1f}, {transformed_near_player[1]:.1f})")
    print(f"   奥側選手 (変換後): ({transformed_far_player[0]:.1f}, {transformed_far_player[1]:.1f})")

    # 鳥瞰図にも選手位置をマーク
    cv2.circle(
        warped,
        (int(transformed_near_player[0]), int(transformed_near_player[1])),
        10,
        (255, 0, 255),
        -1
    )
    cv2.circle(
        warped,
        (int(transformed_far_player[0]), int(transformed_far_player[1])),
        10,
        (255, 165, 0),
        -1
    )

    # 結果を保存
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    player_areas_path = os.path.join(output_dir, 'player_areas_visualization.jpg')
    original_path = os.path.join(output_dir, 'homography_original.jpg')
    warped_path = os.path.join(output_dir, 'homography_warped.jpg')

    cv2.imwrite(player_areas_path, img_with_areas)
    cv2.imwrite(original_path, img_marked)
    cv2.imwrite(warped_path, warped)

    print(f"\n10. 結果を保存しました:")
    print(f"   プレイヤー領域可視化: {player_areas_path}")
    print(f"   元画像: {original_path}")
    print(f"   鳥瞰図: {warped_path}")

    # 画像を表示
    print("\n画像を表示しています（任意のキーを押すと終了）...")

    # ウィンドウサイズを調整して表示
    display_scale = 0.5
    img_areas_small = cv2.resize(
        img_with_areas,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA
    )
    img_marked_small = cv2.resize(
        img_marked,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA
    )
    warped_small = cv2.resize(
        warped,
        None,
        fx=display_scale,
        fy=display_scale,
        interpolation=cv2.INTER_AREA
    )

    cv2.imshow('Player Areas Detection', img_areas_small)
    cv2.imshow('Original (with table corners marked)', img_marked_small)
    cv2.imshow('Bird\'s Eye View (Warped)', warped_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n完了!")


if __name__ == "__main__":
    main()
