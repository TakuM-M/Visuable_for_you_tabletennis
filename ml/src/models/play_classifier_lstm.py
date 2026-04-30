import torch
import torch.nn as nn
from typing import Optional, Tuple

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
        input_size: int = 34,  # 17 keypoints(骨格) × 2 coordinates(x, y) = 34次元
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        初期化

        Args:
            input_size: 入力特徴量の次元数
            hidden_size: LSTM隠れ層のサイズ
            num_layers: LSTMの層数
            dropout: ドロップアウト率
        """
        super(PlayClassifierLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_output_size = hidden_size * 2  # 双方向

        # 入力層（特徴量の前処理）
        self.input_bn = nn.BatchNorm1d(input_size)

        # LSTM層（双方向固定）
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention機構（シーケンス全体のコンテキストを生成）
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # 分類器（LSTM出力 + Attentionコンテキストを結合して分類）
        # 入力: lstm_output(256) + context(256) = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_size * 2, hidden_size),
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

        # Attention機構: シーケンス全体のコンテキストベクトルを生成
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1, keepdim=True)  # (batch, 1, hidden*2)
        context = context.expand_as(lstm_out)  # (batch, seq, hidden*2)

        # LSTM出力とコンテキストを結合して各フレームごとに分類
        enriched = torch.cat([lstm_out, context], dim=-1)  # (batch, seq, hidden*4)
        out = self.classifier(enriched)  # (batch, seq, 1)

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
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Attention重みを取得（可視化用）

        Args:
            x: 入力テンソル (batch, sequence_length, input_size)
            lengths: 各シーケンスの実際の長さ (batch,) - パディング対応用

        Returns:
            attention_weights: Attention重み (batch, sequence_length, 1)
        """
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape

            x_transposed = x.transpose(1, 2)
            x_normalized = self.input_bn(x_transposed)
            x = x_normalized.transpose(1, 2)

            if lengths is not None:
                lengths_cpu = lengths.cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths_cpu, batch_first=True, enforce_sorted=False
                )

            lstm_out, _ = self.lstm(x)

            if lengths is not None:
                lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                    lstm_out, batch_first=True, total_length=seq_len
                )

            attention_weights = self.attention(lstm_out)
            attention_weights = torch.softmax(attention_weights, dim=1)

        return attention_weights