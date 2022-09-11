import anndata
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class scrnaseq:
    """Generate synthetic single-cell RNAseq data with ambient contamination

    Parameters
    ----------
    n_cells : int
        number of cells
    n_celltypes : int
        number of cell types
    n_features : int
        number of features (mRNA)
    n_total_molecules : int, optional
        total molecules per cell, by default 8000
    capture_rate : float, optional
        the probability of being captured by beads, by default 0.7

    Examples
    --------
    .. plot::
        :context: close-figs

        import numpy as np
        from scar import data_generator

        n_features = 1000  # 1000 genes, bad visualization with too big number
        n_cells = 6000  # cells
        n_total_molecules = 20000 # total mRNAs
        n_celltypes = 8  # cell types

        np.random.seed(8)
        scRNAseq = data_generator.scrnaseq(n_cells, n_celltypes, n_features, n_total_molecules=n_total_molecules)
        scRNAseq.generate(dirichlet_concentration_hyper=1)
        scRNAseq.heatmap(vmax=5)
    """
    def __init__(
        self, n_cells, n_celltypes, n_features, n_total_molecules=8000, capture_rate=0.7
    ):
        """initilization"""
        self.n_cells = n_cells
        self.n_celltypes = n_celltypes
        self.n_features = n_features
        self.n_total_molecules = n_total_molecules
        self.capture_rate = capture_rate

    def generate(self, dirichlet_concentration_hyper=0.05):
        """Generate a synthetic scRNAseq dataset.

        Parameters
        ----------
        dirichlet_concentration_hyper : None or real, optional
            the concentration hyperparameters of dirichlet distribution. \
                Determining the sparsity of native signals. \
                    If None, 1 / n_features, by default 0.005.

        Returns
        -------
            After running, several attributes are added
        """
        if dirichlet_concentration_hyper:
            alpha = np.ones(self.n_features) * dirichlet_concentration_hyper
        else:
            alpha = np.ones(self.n_features) / self.n_features

        # simulate native expression frequencies for each cell
        cell_comp_prior = random.dirichlet(np.ones(self.n_celltypes))
        celltype = random.choice(
            a=self.n_celltypes, size=self.n_cells, p=cell_comp_prior
        )
        cell_identity = np.identity(self.n_celltypes)[celltype]
        theta_celltype = random.dirichlet(alpha, size=self.n_celltypes)

        beta_in_each_cell = cell_identity.dot(theta_celltype)

        # simulate total molecules for a droplet in ambient pool
        n_total_mol = random.randint(
            low=self.n_total_molecules / 5, high=self.n_total_molecules / 2, size=1
        )

        # simulate ambient signals
        beta0 = random.dirichlet(np.ones(self.n_features))
        tot_count0 = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0]
        )

        # add empty droplets
        tot_count0_empty = random.negative_binomial(
            n_total_mol, self.capture_rate, size=self.n_cells
        )
        ambient_signals_empty = np.vstack(
            [random.multinomial(n=tot_c, pvals=beta0) for tot_c in tot_count0_empty]
        )

        # simulate native signals
        tot_trails = random.randint(
            low=self.n_total_molecules / 2,
            high=self.n_total_molecules,
            size=self.n_celltypes,
        )
        tot_count1 = [
            random.negative_binomial(tot, self.capture_rate)
            for tot in cell_identity.dot(tot_trails)
        ]

        native_signals = np.vstack(
            [
                random.multinomial(n=tot_c, pvals=theta1)
                for tot_c, theta1 in zip(tot_count1, beta_in_each_cell)
            ]
        )
        obs = ambient_signals + native_signals

        noise_ratio = tot_count0 / (tot_count0 + tot_count1)

        adata = anndata.AnnData(
            obs, 
            obs=pd.DataFrame(celltype, columns=['celltype']), 
            var=pd.DataFrame(beta0, columns=['ambient_profile'])
        )
        adata.obs['noise_ratio'] = noise_ratio
        adata.layers['ambient_signals'] = ambient_signals
        adata.layers['native_signals'] = native_signals
        # self.native_profile = beta_in_each_cell
        # self.total_counts = obs.sum(axis=1)
        # self.empty_droplets = ambient_signals_empty.astype(int)
        return adata

    @staticmethod
    def heatmap(
            adata, feature_type="mRNA", return_obj=False, vmin=0, vmax=10
        ):
        """Heatmap of synthetic data.

        Parameters
        ----------
        feature_type : str, optional
            the feature types, by default "mRNA"
        return_obj : bool, optional
            whether to output figure object, by default False
        figsize : tuple, optional
            figure size, by default (15, 5)
        vmin : int, optional
            colorbar minimum, by default 0
        vmax : int, optional
            colorbar maximum, by default 10

        Returns
        -------
        fig object
            if return_obj, return a fig object
        """
        sort_cell_idx = []
        ambient_profile_idx_sorted = adata.var['ambient_profile'].argsort()
        for f in ambient_profile_idx_sorted:
            sort_cell_idx += list(np.where(adata.obs["celltype"] == f)[0])

        native_signals = adata.layers['native_signals'][sort_cell_idx][
            :, ambient_profile_idx_sorted
        ]
        ambient_signals = adata.layers['ambient_signals'][sort_cell_idx][
            :, ambient_profile_idx_sorted
        ]
        obs = adata.X[sort_cell_idx][:, ambient_profile_idx_sorted]
        denoised_signals = None
        if 'denoised_signals' in adata.layers:
            denoised_signals = adata.layers['denoised_signals'][sort_cell_idx][
                :, ambient_profile_idx_sorted
            ]

        ncols = 4 if 'denoised_signals' in adata.layers else 3
        figsize = (ncols*4, 4)
        fig, axs = plt.subplots(ncols=ncols, figsize=figsize)
        i = 0
        sns.heatmap(
            np.log2(obs + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[i],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[i].set_title("noisy observation")
        i += 1

        if 'denoised_signals' in adata.layers:
            sns.heatmap(
                np.log2(denoised_signals + 1),
                yticklabels=False,
                vmin=vmin,
                vmax=vmax,
                cmap="coolwarm",
                center=1,
                ax=axs[i],
                rasterized=True,
                cbar_kws={"label": "log2(counts + 1)"},
            )
            axs[i].set_title("denoised observation")
            i += 1

        sns.heatmap(
            np.log2(ambient_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[i],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[i].set_title("ambient signals")
        i += 1

        sns.heatmap(
            np.log2(native_signals + 1),
            yticklabels=False,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
            center=1,
            ax=axs[i],
            rasterized=True,
            cbar_kws={"label": "log2(counts + 1)"},
        )
        axs[i].set_title("native signals")
        i += 1

        fig.supxlabel(feature_type)
        fig.supylabel("cells")
        plt.tight_layout()

        if return_obj:
            return fig