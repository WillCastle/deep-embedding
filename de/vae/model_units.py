"""
TODO.
"""

import torch
from torch import nn


class AttentiveRecurrentUnit(nn.Module):
    """
    TODO.
    """
    def __init__(
            self,
            input_hidden_dim: int,
            lstm_hidden_dim: int,
            lstm_num_layers: int,
            lstm_dropout: float,
            lstm_bidirectional: bool,
            attention_embed_dim: int,
            attention_num_heads: int,
            attention_dropout: float,
            linear_output_dim: int,
            ):
        """
        TODO.
        """
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        self.attention_embed_dim = attention_embed_dim
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.linear_output_dim = linear_output_dim

        # InSequence: (source_seq_len, batch_size, input_hidden_dim)
        # InHidden: (num_directions*lstm_num_layers, batch_size, lstm_hidden_dim)
        # InCell: (num_directions*lstm_num_layers, batch_size, lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_hidden_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=self.lstm_bidirectional,
            )
        # OutSequence: (source_seq_len, batch_size, num_directions*lstm_hidden_dim)
        # OutHidden: (num_directions*lstm_num_layers, batch_size, lstm_hidden_dim)
        # OutCell: (num_directions*lstm_num_layers, batch_size, lstm_hidden_dim)

        # InKey: (source_seq_len, batch_size, embed_dim)
        # InValue: (source_seq_len, batch_size, embed_dim)
        # InQuery: (target_seq_len, batch_size, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim,
            num_heads=self.attention_num_heads,
            dropout=self.attention_dropout,
            )
        # OutSequence: (target_seq_len, batch_size, embed_dim),
        # OutWeights: (batch_size, target_seq_len, source_seq_len)

        # InSequence: (target_seq_len, batch_size, embed_dim)
        self.linear = nn.Linear(
            in_features=self.attention_embed_dim,
            out_features=self.linear_output_dim,
            )
        # OutSequence: (target_seq_len, batch_size, linear_output_dim)

    def forward(
            self,
            input_batch: torch.Tensor,
            lstm_hidden_init: torch.Tensor,
            lstm_cell_init: torch.Tensor,
            attention_value: torch.Tensor,
            attention_query: torch.Tensor,
            ):
        """
        TODO.
        """
        lstm_sequence_batch, (lstm_hidden, lstm_cell) = self.lstm(
            input_batch,
            (lstm_hidden_init, lstm_cell_init)
            )
        attention_sequence_batch, attention_output_weights = self.attention(
            query=attention_query,
            key=lstm_sequence_batch,
            value=attention_value
            )
        output_batch = self.linear(attention_sequence_batch)
        return output_batch, (lstm_hidden, lstm_cell), attention_output_weights


if __name__ == "__main__":
    sequence_length = 20
    batch_size = 5
    embedding_dim = 3
    input_batch = torch.randn(sequence_length, batch_size, embedding_dim)

    unit = AttentiveRecurrentUnit(
        input_hidden_dim=embedding_dim,
        lstm_hidden_dim=10,
        lstm_num_layers=2,
        lstm_dropout=0.5,
        lstm_bidirectional=True,
        attention_embed_dim=10,
        attention_num_heads=4,
        attention_dropout=0.1,
        linear_output_dim=6
        )

    output_batch, (lstm_hidden, lstm_cell), attention_output_weights = unit(
        input_batch=input_batch
        lstm_hidden_init=
        lstm_cell_init=
        attention_value=
        attention_query=
        )
