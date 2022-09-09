from pathlib import Path
import anndata
import pyro
from scvi.data import synthetic_iid

from mypackage import SCAR


# def test_mymodel():
#     n_latent = 5
#     adata = synthetic_iid()
#     MyModel.setup_anndata(adata, batch_key="batch", labels_key="labels")
#     model = MyModel(adata, n_latent=n_latent)
#     model.train(1, check_val_every_n_epoch=1, train_size=0.5)
#     model.get_elbo()
#     model.get_latent_representation()
#     model.get_marginal_ll(n_mc_samples=5)
#     model.get_reconstruction_error()
#     model.history

#     # tests __repr__
#     print(model)


def test_scarmodel():
    adata = anndata.read_h5ad(Path(__file__).parent.joinpath('data', 'sim_adata.h5ad'))
    pyro.clear_param_store()
    SCAR.setup_anndata(adata)
    model = SCAR(adata)
    model.train(max_epochs=5, train_size=1)
    model.get_latent(adata)
    model.history

    # tests __repr__
    print(model)
