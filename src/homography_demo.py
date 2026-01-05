"""
ホモグラフィ変換のデモスクリプト

卓球台の四隅の座標を指定して、鳥瞰図変換を実行する例
"""

import sys
import os
import cv2
import numpy as np

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.homography import TableHomography


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
    print("卓球台ホモグラフィ変換デモ")
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

    # ホモグラフィ変換を初期化
    print("\n3. ホモグラフィ変換を初期化中...")
    homography = TableHomography(
        table_corners=table_corners,
        table_width_mm=1525.0,   # 卓球台の公式幅
        table_height_mm=2740.0,  # 卓球台の公式長さ
        output_scale=0.3         # 1mm = 0.3px
    )

    output_size = homography.get_output_size()
    print(f"   出力画像サイズ: {output_size[0]}x{output_size[1]} px")

    # 変換の可視化
    print("\n4. 変換を実行中...")
    img_marked, warped = homography.visualize_transformation(img)

    # テスト: いくつかの座標を変換
    print("\n5. 座標変換のテスト:")
    test_points = [
        (table_corners[0], "左上"),
        (table_corners[1], "右上"),
        ((960, 540), "画像中心付近"),
    ]

    for point, label in test_points:
        transformed = homography.transform_point(point)
        print(f"   {label}: {point} -> ({transformed[0]:.1f}, {transformed[1]:.1f})")

    # 選手がいる領域を想定した点を変換
    print("\n6. 選手位置の変換テスト:")
    # 卓球台の手前側（下側）に選手がいると仮定
    player_point = ((table_corners[2][0] + table_corners[3][0]) // 2, table_corners[3][1] + 200)

    # 画像に選手位置をマーク
    cv2.circle(img_marked, player_point, 15, (255, 0, 255), -1)
    cv2.putText(
        img_marked,
        "Player",
        (player_point[0] - 30, player_point[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2
    )

    transformed_player = homography.transform_point(player_point)
    print(f"   選手位置: {player_point} -> ({transformed_player[0]:.1f}, {transformed_player[1]:.1f})")

    # 鳥瞰図にも選手位置をマーク
    cv2.circle(
        warped,
        (int(transformed_player[0]), int(transformed_player[1])),
        10,
        (255, 0, 255),
        -1
    )

    # 結果を保存
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, 'homography_original.jpg')
    warped_path = os.path.join(output_dir, 'homography_warped.jpg')

    cv2.imwrite(original_path, img_marked)
    cv2.imwrite(warped_path, warped)

    print(f"\n7. 結果を保存しました:")
    print(f"   元画像: {original_path}")
    print(f"   鳥瞰図: {warped_path}")

    # 画像を表示
    print("\n画像を表示しています（任意のキーを押すと終了）...")

    # ウィンドウサイズを調整して表示
    display_scale = 0.5
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

    cv2.imshow('Original (with table corners marked)', img_marked_small)
    cv2.imshow('Bird\'s Eye View (Warped)', warped_small)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n完了!")


if __name__ == "__main__":
    main()
