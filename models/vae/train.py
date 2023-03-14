"""
TODO.
"""

import torch
from attentive_recurrent_unit import AttentiveRecurrentUnit
from torch import nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, dataloader: DataLoader) -> nn.Module:
    """
    TODO.
    """

    return model

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
