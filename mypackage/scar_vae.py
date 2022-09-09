from typing import Iterable

import pyro
import pyro.distributions as dist
import torch
from torch import nn as nn
from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import PyroBaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, FCLayers


class mytanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_x):
        var_tanh = torch.tanh(input_x)
        output = (1 + var_tanh) / 2
        return output


class hnormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_x):
        return input_x / (input_x.sum(dim=1).view(-1, 1) + 1e-5)


class mysoftplus(nn.Module):
    def __init__(self, sparsity=0.9):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, input_x):
        return self._mysoftplus(input_x)

    def _mysoftplus(self, input_x):
        """customized softplus activation, output range: [0, inf)"""
        var_sp = torch.nn.functional.softplus(input_x)
        threshold = torch.nn.functional.softplus(
            torch.tensor(-(1 - self.sparsity) * 10.0, device=input_x.device)
        )
        var_sp = var_sp - threshold
        zero = torch.zeros_like(threshold)
        var_out = torch.where(var_sp <= zero, zero, var_sp)
        return var_out


class DecoderSCAR(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus", "mysoftplus"] = "mysoftplus",
        sparsity: float = 0.9,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        elif scale_activation == "mysoftplus":
            px_scale_activation = mysoftplus(sparsity)
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
            hnormalization(), # adding normalization!
        )

        # noise ratio
        self.px_noise_decoder = nn.Sequential(
            nn.Linear(n_hidden, 1),
            mytanh(),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
    ):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)

        # noise ratio
        px_noise_ratio = self.px_noise_decoder(px)

        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None

        return px_scale, px_noise_ratio, px_r, px_rate, px_dropout


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
        amb_prob: torch.tensor, 
        n_input: int, 
        n_latent: int, 
        n_hidden: int, 
        n_layers: int,
        sparsity: float
    ):

        super().__init__()
        self.amb_prob = amb_prob
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
            scale_activation="mysoftplus",
            sparsity=self.sparsity
        )
        # This gene-level parameter modulates the variance of the observation distribution
        self.px_r = torch.nn.Parameter(torch.ones(self.n_input))

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
            px_scale, px_noise_ratio, _, px_rate, px_dropout = self.decoder("gene", z, log_library)
            # build count distribution
            nb_logits = (px_rate + self.epsilon).log() - (
                self.px_r.exp() + self.epsilon
            ).log()
            # model logits for counts and ambient separately
            total_logits = nb_logits * (1-px_noise_ratio) + (self.amb_prob.to(x.device) + self.epsilon).log() * px_noise_ratio
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
        px_scale, px_noise_ratio, _, px_rate, px_dropout = self.decoder("gene", z, log_library)
        return (1-px_noise_ratio) * px_rate
        

    @torch.no_grad()
    @auto_move_data
    def get_latent(self, tensors):
        x = tensors[REGISTRY_KEYS.X_KEY]
        x_ = torch.log(1 + x)
        z_loc, _, _ = self.encoder(x_)
        return z_loc
