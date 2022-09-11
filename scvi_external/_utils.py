from typing import Iterable

from scvi.nn import FCLayers
from scvi._compat import Literal
import torch
from torch import nn as nn


class tanh(nn.Module):
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
        return input_x / (input_x.sum(dim=-1, keepdim=True) + 1e-5)


class softplus(nn.Module):
    def __init__(self, sparsity=0.9):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, input_x):
        return self._softplus(input_x)

    def _softplus(self, input_x):
        """customized softplus activation, output range: [0, inf)"""
        var_sp = nn.functional.softplus(input_x)
        threshold = nn.functional.softplus(
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
        scale_activation: Literal["softmax", "softplus", "softplus_sp"] = "softplus",
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
        elif scale_activation == "softplus_sp":
            px_scale_activation = softplus(sparsity)
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
            hnormalization(),
        )

        # noise ratio
        self.px_noise_decoder = nn.Sequential(
            nn.Linear(n_hidden, 1),
            tanh(),
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
