# コンポーネントの名前
- proto_type_03ではend_to_endでLSTMを用いたプレイ分類器を実装をする

## メモ
- データの正規化については獲得時に実行する．正規化処理はexoprterに実装を行う
- データセットの設計については，csvからの獲得，memoryからの獲得，複数csvからの獲得などを考慮する

## 修正するべき場所
- []datasetsについて，csvからの獲得，memoryからの獲得，複数csvからの獲得などの設計が複雑化しているため，ファイル分割による利用するクラスの明確に分割をする
    - base_dataset.py - 基底クラスと共通機能
    - csv_dataset.py - CSV読み込み専用
    - memory_dataset.py - メモリ上のデータ専用
    - multi_csv_dataset.py - 複数CSV統合（既存を改善）
    - init.py - エクスポート管理

            # 単一CSVから
            from src.datasets import CSVPoseSequenceDataset
            dataset = CSVPoseSequenceDataset(
                csv_path="normalized_data.csv",
                label_path="labels.csv",
                sequence_length=30,
                stride=5
            )

            # メモリから（推論時）
            from src.datasets import MemoryPoseSequenceDataset
            dataset = MemoryPoseSequenceDataset(
                pose_data=normalized_array,
                sequence_length=30,
                stride=1
            )

            # 複数CSVから
            from src.datasets import MultiCSVPoseDataset
            dataset = MultiCSVPoseDataset(
                csv_label_pairs=[("video1.csv", "label1.csv"), ...],
                sequence_length=30
            )

- []パイプラインの設計について，設計が複雑化しかつ内容について把握しきれていないため，整理を行う
- []モデルについて，現在はLSTMで設計をする．LSTM以外のモデルについては今後の展望としてまとめる
- []end-to-end処理の場合についてのyolo可視化の設定実装，可視化する必要性はアプリ時はない