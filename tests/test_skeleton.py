from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from scvi.data import synthetic_iid

from scvi_external import SCAR
from .data_generator import scrnaseq


def test_scarmodel():
    n_latent = 5
    adata = synthetic_iid()
    SCAR.setup_anndata(adata, batch_key="batch", labels_key="labels")

    model = SCAR(adata, ambient_profile=None, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_elbo()
    model.get_latent_representation()
    model.get_marginal_ll(n_mc_samples=5)
    model.get_reconstruction_error()
    model.history

    # tests __repr__
    print(model)


def test_train_scarmodel():
    n_features = 1000
    n_cells = 6000
    n_total_molecules = 20000
    n_celltypes = 8
    np.random.seed(42)
    gen = scrnaseq(
        n_cells, n_celltypes, n_features, n_total_molecules=n_total_molecules
    )
    adata = gen.generate(dirichlet_concentration_hyper=1)

    SCAR.setup_anndata(adata)
    # SCAR.get_ambient_profile(adata, raw_adata, feature_type="mRNA")
    model = SCAR(adata, ambient_profile=None)
    model.train(max_epochs=10, train_size=1, batch_size=64)
    model.get_latent_representation(adata)

    # take median of posterior samples as denoised signals
    posterior_pred = model.posterior_predictive_sample(adata, n_samples=5)
    adata.layers["denoised_signals"] = np.median(posterior_pred, axis=-1)

    plt.plot(model.history["elbo_train"])
    plt.savefig(Path(__file__).parent.joinpath("results", "elbo_loss.png"))
    plt.close()

    scrnaseq.heatmap(adata, return_obj=True)
    plt.savefig(Path(__file__).parent.joinpath("results", "heatmap.png"))
    plt.close()

    # native signals are ground truth from simulated dataset
    denoised_dist = np.mean(
        [
            wasserstein_distance(
                adata.layers["native_signals"][i, :],
                adata.layers["denoised_signals"][i, :],
            )
            for i in range(adata.shape[0])
        ]
    )
    orig_dist = np.mean(
        [
            wasserstein_distance(adata.layers["native_signals"][i, :], adata.X[i, :])
            for i in range(adata.shape[0])
        ]
    )
    print(f"Original wasserstein distance: {round(orig_dist, 3)}")
    print(f"Denoised wasserstein distance: {round(denoised_dist, 3)}")

    # tests __repr__
    print(model)
