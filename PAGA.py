import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
import os

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


sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=10, use_rep='X_diffmap')


sc.tl.draw_graph(adata)
sc.tl.leiden(adata, resolution=1.0)


sc.tl.paga(adata, groups='Cell subpopulations')
sc.pl.paga(adata, color=['Cell subpopulations'])
sc.pl.paga(adata, threshold=0.03, show=False)


sc.tl.draw_graph(adata, init_pos='paga')



rcParams['figure.figsize'] = 5, 5
sc.pl.paga_compare(
    adata, threshold=0.03, title='', right_margin=0.2, size=10, edge_width_scale=0.5,
    legend_fontsize=12, fontsize=12, frameon=False, edges=True
    ,save='160_PAGA.pdf'
    ,legend_loc='on data')