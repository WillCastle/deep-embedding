"""
TODO.
"""

# import numpy as np
# import pytorch_lightning as pl

import torch
from pl_bolts.models.autoencoders import VAE
from torchviz import make_dot

INPUT_HEIGHT = 10
N_CHANNELS = 3

vae = VAE(input_height=INPUT_HEIGHT)

vae.eval()
BATCH_SIZE = 1
# Batch of one square, 3 channel image
test_input = torch.randn(
    BATCH_SIZE,
    N_CHANNELS,
    INPUT_HEIGHT,
    INPUT_HEIGHT
    )

test_output = vae(test_input)

graph = make_dot(test_output.mean(), params=dict(vae.named_parameters()))
graph.view(directory="C:/Users/willf/DataScience/Repos/deep-embedding/vae/graphs")
