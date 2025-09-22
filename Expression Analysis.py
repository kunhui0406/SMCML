import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
from matplotlib import rcParams
from matplotlib.pyplot import rc_context



os.chdir("./data")



sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
results_file = './write/paul15.h5ad'
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(3, 3), facecolor='white')


adata = sc.read_10x_mtx(
    './',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=False)


adata.X = adata.X.astype('float64')
new_label = pd.read_csv("label.csv")


new_label = list(np.array(new_label.iloc[:,0]).ravel())
adata.obs['Cell subpopulations'] = new_label


sc.pp.normalize_total(adata, target_sum=1e4)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)


sc.tl.umap(adata)
genes = adata.var_names

sc.pl.heatmap(
    adata,
    var_names=adata.var_names[:80],
    groupby="Cell subpopulations",
    standard_scale="var",  # z-score
    dendrogram=True,
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(14, 20)

)

sc.pl.heatmap(
    adata,
    var_names=adata.var_names[80:160],
    groupby="Cell subpopulations",
    standard_scale="var",  # z-score
    dendrogram=True,
    cmap="viridis",
    show_gene_labels=True,
    swap_axes=True,
    figsize=(14, 20)

)


sc.tl.leiden(adata)
sc.tl.rank_genes_groups(adata, 'Cell subpopulations', method='t-test')
sc.pl.rank_genes_groups_stacked_violin(adata, n_genes=5, cmap='viridis_r')


with rc_context({'figure.figsize': (9, 1.5)}):
    sc.pl.rank_genes_groups_violin(adata, n_genes=20, jitter=False)

