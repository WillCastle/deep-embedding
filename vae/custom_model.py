"""
TODO.
"""

import torch
from torch import nn


class StackedEncoder(nn.Module):
    """
    TODO.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            output_dim: int
            ):
        """
        TODO.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.encoder = nn.Sequential(*[nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU()
                ) for in_dim, out_dim in zip(self.dims, self.dims[1:])])

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        TODO.
        """
        encoded_batch = self.encoder(input_batch)
        return encoded_batch


class VAE(nn.Module):
    """
    TODO.
    """
    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            latent_dim: int
            ):
        """
        TODO.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        self.dims = [self.input_dim] + \
            self.hidden_dims + \
            [self.latent_dim] + \
            self.hidden_dims[::-1] + \
            [self.input_dim]

        self.encoder_module = StackedEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim
            )
        self.decoder_module = StackedEncoder(
            input_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],
            output_dim=input_dim
            )

        self.latent_mean = nn.Linear(latent_dim, latent_dim)
        self.latent_log_variance = nn.Linear(latent_dim, latent_dim)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """
        TODO.
        """
        latent_encoding = self.encoder_module(input_batch)
        mean = self.latent_mean(latent_encoding)
        log_variance = self.latent_log_variance(latent_encoding)
        standard_deviation = torch.exp(log_variance / 2)

        latent_distribution = torch.distributions.Normal(
            mean,
            standard_deviation
            )
        latent_sample = latent_distribution.rsample()

        decoded_batch = self.decoder_module(latent_sample)
        return decoded_batch

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """
        TODO.
        """
        return


if __name__ == '__main__':
    vae = VAE(input_dim=64, hidden_dims=[16, 4], latent_dim=2)
    vae.eval()
    print(vae(torch.zeros(1, 64)))
