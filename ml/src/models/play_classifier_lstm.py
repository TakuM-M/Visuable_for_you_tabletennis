import torch
import torch.nn as nn
from typing import Tuple, Optional

class PlayClassifierLSTM(nn.Module):
    """
    プレー検知用LSTMモデル

    アーキテクチャ:
    - 入力: (batch, sequence_length, features)
        - features = 17 keypoints × 2 coordinates = 34次元
    - LSTM層（双方向）で時系列パターンを抽出
    - Attention機構で重要なフレームに注目
    - 全結合層で各フレームごとに分類
    - 出力: (batch, sequence_length, 1) プレー中の確率
    """

    def __init__(
        self,
        input_size: int = 34,  # 17 keypoints × 2 coordinates
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        """
        初期化

        Args:
            input_size: 入力特徴量の次元数（デフォルト: 34 = 17キーポイント×2座標）
            hidden_size: LSTM隠れ層のサイズ
            num_layers: LSTMの層数
            dropout: ドロップアウト率
            bidirectional: 双方向LSTMを使用するか
            use_attention: Attention機構を使用するか
        """
        super(PlayClassifierLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention

        # 双方向の場合は出力サイズが2倍になる
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_size = hidden_size * self.num_directions

        # 入力層（特徴量の前処理）
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM層
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention機構（オプション）
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

        # 分類器（logits出力 — Sigmoid は損失関数側で適用）
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        順伝播

        Args:
            x: 入力テンソル (batch, sequence_length, input_size)
            lengths: 各シーケンスの実際の長さ (batch,) - パディング対応用

        Returns:
            out: logits (batch, sequence_length, 1)
                 確率に変換するにはtorch.sigmoid()を適用
        """
        batch_size, seq_len, _ = x.shape

        # Batch Normalization（次元を入れ替える必要がある）
        # (batch, seq, features) -> (batch, features, seq)
        x_transposed = x.transpose(1, 2)
        x_normalized = self.input_bn(x_transposed)
        x = x_normalized.transpose(1, 2)  # 元に戻す

        # 可変長シーケンスの場合はpack_padded_sequenceを使用
        if lengths is not None:
            # lengthsをCPUに移動してからpack
            lengths_cpu = lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

        # LSTM層
        lstm_out, (hidden, cell) = self.lstm(x)

        # pack_padded_sequenceを使った場合は元に戻す
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )

        # Attention機構（オプション）
        if self.use_attention:
            # Attention重みを計算
            attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)

            # Attentionを適用（要素ごとの重み付け）
            attended = lstm_out * attention_weights
        else:
            attended = lstm_out

        # 各フレームごとに分類（logits出力）
        out = self.classifier(attended)  # (batch, seq, 1)

        return out

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        推論用メソッド

        Args:
            x: 入力テンソル (batch, sequence_length, input_size)
            threshold: 分類の閾値（デフォルト: 0.5）

        Returns:
            probs: プレー中の確率 (batch, sequence_length)
            preds: プレー中かどうかの2値予測 (batch, sequence_length)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)  # (batch, seq, 1)
            probs = torch.sigmoid(logits).squeeze(-1)  # (batch, seq)
            preds = (probs >= threshold).long()
        return probs, preds

    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Attention重みを取得（可視化用）

        Args:
            x: 入力テンソル (batch, sequence_length, input_size)

        Returns:
            attention_weights: Attention重み (batch, sequence_length, 1)
                               use_attention=Falseの場合はNone
        """
        if not self.use_attention:
            return None

        self.eval()
        with torch.no_grad():
            # 入力の正規化
            x_transposed = x.transpose(1, 2)
            x_normalized = self.input_bn(x_transposed)
            x = x_normalized.transpose(1, 2)

            # LSTM
            lstm_out, _ = self.lstm(x)

            # Attention重み
            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)

        return attention_weights