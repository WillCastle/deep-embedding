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
            input_embed_dim: int,
            lstm_hidden_dim: int,
            lstm_num_layers: int,
            lstm_dropout: float,
            lstm_bidirectional: bool,
            attention_num_heads: int,
            attention_dropout: float,
            linear_embed_dim: int,
            ):
        """
        TODO.
        """
        super().__init__()
        self.input_embed_dim = input_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.lstm_bidirectional = lstm_bidirectional
        self.num_directions = 2 if self.lstm_bidirectional else 1
        self.attention_num_heads = attention_num_heads
        self.attention_dropout = attention_dropout
        self.linear_embed_dim = linear_embed_dim

        # InSequence:(source_seq_len,batch_size,input_embed_dim)
        # InHidden:(num_directions*lstm_num_layers,batch_size,lstm_hidden_dim)
        # InCell:(num_directions*lstm_num_layers,batch_size,lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.input_embed_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=self.lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=self.lstm_bidirectional,
            )
        # OutSequence:(source_seq_len,batch_size,num_directions*lstm_hidden_dim)
        # OutHidden:(num_directions*lstm_num_layers,batch_size,lstm_hidden_dim)
        # OutCell:(num_directions*lstm_num_layers,batch_size,lstm_hidden_dim)

        # InKey:(source_seq_len,batch_size,num_directions*lstm_hidden_dim)
        # InValue:(source_seq_len,batch_size,num_directions*lstm_hidden_dim)
        # InQuery:(target_seq_len,batch_size,num_directions*lstm_hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.num_directions * self.lstm_hidden_dim,
            num_heads=self.attention_num_heads,
            dropout=self.attention_dropout,
            )
        # OutSequence:(target_seq_len,batch_size,num_directions*lstm_hidden_dim)
        # OutWeights:(batch_size,target_seq_len,source_seq_len)

        # InSequence:(target_seq_len,batch_size,num_directions*lstm_hidden_dim)
        self.linear = nn.Linear(
            in_features=self.num_directions * self.lstm_hidden_dim,
            out_features=self.linear_embed_dim,
            )
        # OutSequence:(target_seq_len,batch_size,linear_embed_dim)

    def forward(
            self,
            input_batch: torch.Tensor,
            lstm_hidden: torch.Tensor,
            lstm_cell: torch.Tensor,
            attention_value: torch.Tensor,
            attention_query: torch.Tensor,
            ):
        """
        TODO.
        """
        lstm_sequence_batch, (lstm_hidden, lstm_cell) = self.lstm(
            input_batch,
            (lstm_hidden, lstm_cell)
            )
        attention_sequence_batch, attention_weights = self.attention(
            query=attention_query,
            key=lstm_sequence_batch,
            value=attention_value
            )
        output_batch = self.linear(attention_sequence_batch)
        return output_batch, (lstm_hidden, lstm_cell), attention_weights


if __name__ == "__main__":
    # Input Parameters
    SOURCE_SEQ_LEN = 20
    TARGET_SEQ_LEN = 15
    BATCH_SIZE = 5
    INPUT_EMBED_DIM = 3

    # Model Parameters
    LSTM_HIDDEN_DIM = 10
    LSTM_NUM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True
    LSTM_NUM_DIRECTIONS = 2 if LSTM_BIDIRECTIONAL else False
    ATTENTION_NUM_HEADS = 2
    LINEAR_EMBED_DIM = 6

    # Inputs
    INPUT_BATCH = torch.randn(SOURCE_SEQ_LEN, BATCH_SIZE, INPUT_EMBED_DIM)
    LSTM_HIDDEN = torch.zeros(
        LSTM_NUM_DIRECTIONS * LSTM_NUM_LAYERS,
        BATCH_SIZE,
        LSTM_HIDDEN_DIM,
        )
    LSTM_CELL = torch.zeros(
        LSTM_NUM_DIRECTIONS * LSTM_NUM_LAYERS,
        BATCH_SIZE,
        LSTM_HIDDEN_DIM,
        )
    ATTENTION_VALUE = torch.zeros(
        SOURCE_SEQ_LEN,
        BATCH_SIZE,
        LSTM_NUM_DIRECTIONS * LSTM_HIDDEN_DIM,
        )
    ATTENTION_QUERY = torch.zeros(
        TARGET_SEQ_LEN,
        BATCH_SIZE,
        LSTM_NUM_DIRECTIONS * LSTM_HIDDEN_DIM,
        )

    # Construct Model
    unit = AttentiveRecurrentUnit(
        input_embed_dim=INPUT_EMBED_DIM,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        lstm_num_layers=LSTM_NUM_LAYERS,
        lstm_dropout=0.5,
        lstm_bidirectional=LSTM_BIDIRECTIONAL,
        attention_num_heads=ATTENTION_NUM_HEADS,
        attention_dropout=0.1,
        linear_embed_dim=LINEAR_EMBED_DIM,
        )

    # Outputs
    OUTPUT_BATCH, (LSTM_HIDDEN, LSTM_CELL), ATTENTION_WEIGHTS = unit(
        input_batch=INPUT_BATCH,
        lstm_hidden=LSTM_HIDDEN,
        lstm_cell=LSTM_CELL,
        attention_value=ATTENTION_VALUE,
        attention_query=ATTENTION_QUERY,
        )

    print(f"OUTPUT_BATCH shape: {OUTPUT_BATCH.shape}\n")
    print(f"LSTM_HIDDEN shape: {LSTM_HIDDEN.shape}\n")
    print(f"LSTM_CELL shape: {LSTM_CELL.shape}\n")
    print(f"ATTENTION_WEIGHTS shape: {ATTENTION_WEIGHTS.shape}\n")
