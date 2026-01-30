"""プレイヤーIDを集約する処理モジュール"""
import pandas as pd


def merge_player_ids(input_csv: str, output_csv: str, player_a_ids: list, player_b_ids: list):
    """track_idをプレイヤーA (ID=1) とプレイヤーB (ID=2) に集約

    Args:
        input_csv: 入力CSVファイルパス
        output_csv: 出力CSVファイルパス
        player_a_ids: プレイヤーAのtrack_idリスト (ID 1に集約)
        player_b_ids: プレイヤーBのtrack_idリスト (ID 2に集約)
    """
    # データ読み込み
    df = pd.read_csv(input_csv)

    # 指定されたIDのみを抽出
    all_ids = player_a_ids + player_b_ids
    df_filtered = df[df['track_id'].isin(all_ids)].copy()

    print(f"フィルタリング前: {len(df)} 行")
    print(f"フィルタリング後: {len(df_filtered)} 行")

    # ID集約の実行
    df_filtered.loc[df_filtered['track_id'].isin(player_a_ids), 'track_id'] = 1
    df_filtered.loc[df_filtered['track_id'].isin(player_b_ids), 'track_id'] = 2

    print(f"\nID集約結果:")
    print(f"  プレイヤーA (ID=1): {len(df_filtered[df_filtered['track_id'] == 1])} 行")
    print(f"  プレイヤーB (ID=2): {len(df_filtered[df_filtered['track_id'] == 2])} 行")

    # 保存
    df_filtered.to_csv(output_csv, index=False)
    print(f"\n出力ファイル: {output_csv}")
