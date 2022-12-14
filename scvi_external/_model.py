import logging
from typing import List, Optional, Union

from anndata import AnnData
import numpy as np
import pandas as pd
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import (
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
    RNASeqMixin,
    ArchesMixin,
)
from scvi.utils import setup_anndata_dsp
from scvi._compat import Literal
import torch
from torch.distributions.multinomial import Multinomial
from tqdm import tqdm

from ._module import SCAR_VAE

logger = logging.getLogger(__name__)


class SCAR(
    RNASeqMixin, ArchesMixin, VAEMixin, UnsupervisedTrainingMixin, BaseModelClass
):
    """
    Ambient RNA removal in scRNA-seq data
    Publication: \
        [Probabilistic machine learning ensures accurate ambient denoising in droplet-based single-cell omics](https://www.biorxiv.org/content/10.1101/2022.01.14.476312v4).
    Original Github: https://github.com/Novartis/scar.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi_external.SCAR.setup_anndata`.
    ambient_profile
        The probability of occurrence of each ambient transcript.\
            If None, averaging cells to estimate the ambient profile, by default None.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    scale_activation
        Activation layer to use for px_scale_decoder
    sparsity
        The sparsity of expected native signals. It varies between datasets,
        e.g. if one prefilters genes -- use only highly variable genes --
        the sparsity should be low; on the other hand, it should be set high
        in the case of unflitered genes.
    **model_kwargs
        Keyword args for :class:`~scvi_external.SCAR`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> raw_adata = anndata.read_h5ad(path_to_raw_anndata)
    >>> scvi_external.SCAR.setup_anndata(adata, batch_key="batch")
    >>> scvi_external.SCAR.get_ambient_profile(adata=adata, raw_adata=raw_adata, feature_type="mRNA")
    >>> vae = scvi_external.SCAR(adata)
    >>> vae.train()
    >>> adata.obsm["X_scAR"] = vae.get_latent_representation()
    >>> posterior_preds = vae.posterior_predictive_sample(n_samples=15)
    >>> adata.layers['denoised'] = sp.sparse.csr_matrix(np.median(posterior_preds, axis=-1))
    """

    def __init__(
        self,
        adata: AnnData,
        ambient_profile: Union[str, np.ndarray, pd.DataFrame, torch.tensor] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        scale_activation: Literal["softmax", "softplus", "softplus_sp"] = "softplus",
        sparsity: float = 0.9,
        **model_kwargs,
    ):
        super(SCAR, self).__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        # self.summary_stats provides information about anndata dimensions and other tensor info
        if not torch.is_tensor(ambient_profile):
            if isinstance(ambient_profile, str):
                ambient_profile = np.nan_to_num(adata.varm[ambient_profile])
            elif isinstance(ambient_profile, pd.DataFrame):
                ambient_profile = ambient_profile.fillna(0).values
            elif isinstance(ambient_profile, np.ndarray):
                ambient_profile = np.nan_to_num(ambient_profile)
            elif not ambient_profile:
                ambient_profile = adata.X.sum(axis=0) / adata.X.sum(axis=0).sum()
                ambient_profile = np.nan_to_num(ambient_profile)
            else:
                raise TypeError(
                    f"Expecting str / np.array / None / pd.DataFrame, but get a {type(ambient_profile)}"
                )
            ambient_profile = (
                torch.from_numpy(np.asarray(ambient_profile)).float().reshape(1, -1)
            )

        self.module = SCAR_VAE(
            ambient_profile=ambient_profile,
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            scale_activation=scale_activation,
            sparsity=sparsity,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI-AR Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def get_ambient_profile(
        adata: AnnData,
        raw_adata: AnnData,
        prob: float = 0.995,
        min_raw_counts: int = 2,
        iterations: int = 3,
        n_batch: int = 1,
        sample: int = 50000,
    ):
        """Calculate ambient profile for relevant features
        Identify the cell-free droplets through a multinomial distribution. See EmptyDrops [Lun2019]_ for details.

        Parameters
        ----------
        adata : AnnData
            A filtered adata object, loaded from filtered_feature_bc_matrix using `scanpy.read`, gene filtering is recommended to save memory
        raw_adata : AnnData
            A raw adata object, loaded from raw_feature_bc_matrix using `scanpy.read`
        prob : float, optional
            The probability of each gene, considered as containing ambient RNA if greater than prob (joint prob euqals to the product of all genes for a droplet), by default 0.995
        min_raw_counts : int, optional
            Total counts filter for raw_adata, filtering out low counts to save memory, by default 2
        iterations : int, optional
            Total iterations, by default 3
        n_batch : int, optional
            Total number of batches, set it to a bigger number when out of memory issue occurs, by default 1
        sample : int, optional
            Randomly sample droplets to test, if greater than total droplets, use all droplets, by default 50000

        Returns
        -------
        The relevant ambient profile is added in `adata.varm`
        """
        # take subset genes to save memory
        raw_adata = raw_adata[:, raw_adata.var_names.isin(adata.var_names)]
        raw_adata = raw_adata[raw_adata.X.sum(axis=1) >= min_raw_counts]

        sample = int(sample)
        idx = np.random.choice(
            raw_adata.shape[0], size=min(raw_adata.shape[0], sample), replace=False
        )
        raw_adata = raw_adata[idx]
        print(f"Randomly sampling {sample} droplets to calculate the ambient profile.")
        # initial estimation of ambient profile, will be updated
        ambient_prof = raw_adata.X.sum(axis=0) / raw_adata.X.sum()

        for _ in tqdm(range(iterations)):
            # calculate joint probability (log) of being cell-free droplets for each droplet
            log_prob = []
            batch_idx = np.floor(
                np.array(range(raw_adata.shape[0])) / raw_adata.shape[0] * n_batch
            )
            for b in range(n_batch):
                try:
                    count_batch = raw_adata[batch_idx == b].X.astype(int).A
                except MemoryError:
                    raise MemoryError("use more batches by setting a higher n_batch")
                log_prob_batch = Multinomial(
                    probs=torch.tensor(ambient_prof), validate_args=False
                ).log_prob(torch.Tensor(count_batch))
                log_prob.append(log_prob_batch)
            log_prob = np.concatenate(log_prob, axis=0)
            raw_adata.obs["log_prob"] = log_prob
            raw_adata.obs["droplets"] = "other droplets"
            # cell-containing droplets
            raw_adata.obs.loc[
                raw_adata.obs_names.isin(adata.obs_names), "droplets"
            ] = "cells"
            # identify cell-free droplets
            raw_adata.obs["droplets"] = raw_adata.obs["droplets"].mask(
                raw_adata.obs["log_prob"] >= np.log(prob) * raw_adata.shape[1],
                "cell-free droplets",
            )
            emptydrops = raw_adata[raw_adata.obs["droplets"] == "cell-free droplets"]
            if emptydrops.shape[0] < 50:
                raise Exception("Too few emptydroplets. Lower the prob parameter")
            ambient_prof = emptydrops.X.sum(axis=0) / emptydrops.X.sum()

        # update ambient profile
        adata.varm["ambient_profile"] = emptydrops.X.sum(axis=0).reshape(-1, 1) / emptydrops.X.sum()
