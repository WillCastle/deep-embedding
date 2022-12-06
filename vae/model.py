"""
TODO.
"""
import numpy as np
import pytorch_lightning as pl
import torch
from pl_bolts.models.autoencoders import VAE

vae = VAE(input_height=10)

print(vae)