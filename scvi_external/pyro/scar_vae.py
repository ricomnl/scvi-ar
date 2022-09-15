import pyro
import pyro.distributions as dist
import torch
from torch import nn as nn
from scvi import REGISTRY_KEYS
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import Encoder

from .._utils import DecoderSCAR


class SCAR_VAE(PyroBaseModuleClass):
    """
    Skeleton Variational auto-encoder Pyro model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    n_latent
        Dimensionality of the latent space
    n_hidden
        Number of nodes per hidden layer
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    """

    def __init__(
        self,
        ambient_profile: torch.tensor,
        n_input: int,
        n_latent: int,
        n_hidden: int,
        n_layers: int,
        sparsity: float,
    ):

        super().__init__()
        self.ambient_profile = ambient_profile
        self.n_input = n_input
        self.n_latent = n_latent
        self.epsilon = 5.0e-3
        self.sparsity = sparsity
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.1,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCAR(
            n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            scale_activation="softplus",
            sparsity=self.sparsity,
        )
        # This gene-level parameter modulates the variance of the observation distribution
        self.px_r = nn.Parameter(torch.ones(self.n_input))

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        x = tensor_dict[REGISTRY_KEYS.X_KEY]
        log_library = torch.log(torch.sum(x, dim=1, keepdim=True) + 1e-6)
        return (x, log_library), {}

    def model(self, x, log_library):
        # register PyTorch module `decoder` with Pyro
        pyro.module("scvi", self)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.n_latent)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.n_latent)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            px_scale, px_noise_ratio, _, px_rate, px_dropout = self.decoder(
                "gene", z, log_library
            )
            # build count distribution
            nb_logits = (px_rate + self.epsilon).log() - (
                self.px_r.exp() + self.epsilon
            ).log()
            # model logits for counts and ambient separately
            total_logits = (
                nb_logits * (1 - px_noise_ratio)
                + (self.ambient_profile.to(x.device) + self.epsilon).log()
                * px_noise_ratio
            )
            x_dist = dist.ZeroInflatedNegativeBinomial(
                gate_logits=px_dropout, total_count=self.px_r.exp(), logits=total_logits
            )
            # score against actual counts
            pyro.sample("obs", x_dist.to_event(1), obs=x)

    def guide(self, x, log_library):
        # define the guide (i.e. variational distribution) q(z|x)
        pyro.module("scvi", self)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            x_ = torch.log(1 + x)
            z_loc, z_scale, _ = self.encoder(x_)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    def generative(self, x, log_library):
        z_loc, z_scale, _ = self.encoder(x)
        z = dist.Normal(z_loc, z_scale).sample()
        px_scale, px_noise_ratio, _, px_rate, px_dropout = self.decoder(
            "gene", z, log_library
        )
        return (1 - px_noise_ratio) * px_rate

    @torch.no_grad()
    @auto_move_data
    def get_latent(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_ = torch.log(1 + x)
        z_loc, _, _ = self.encoder(x_)
        return z_loc
