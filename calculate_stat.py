import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def load_matrix(path):
    p = Path(path)
    if p.suffix == '.h5ad':
        ad = sc.read_h5ad(str(p))
        print(f"Cells, Genes:{ad.shape} and 1% Cells: {int(ad.shape[0] * 0.01)}")
        X = ad.X
        genes = ad.var_names.to_list()
        return X, genes

    if p.suffix in ('.csv', '.tsv', '.txt'):
        sep = ',' if p.suffix == '.csv' else '\t'
        df = pd.read_csv(str(p), sep=sep, index_col=0)
        print(f"Cells, Genes:{df.shape} and 1% Cells: {int(df.shape[0] * 0.01)}")
        X = df.values
        genes = df.columns.to_list()
        return X, genes

    raise ValueError('Unsupported input format: ' + str(path))


def to_csr(X):
    return X.tocsr() if sparse.issparse(X) else sparse.csr_matrix(X)


def gene_stats(X):
    # Expect X = cells × genes
    X = to_csr(X)

    # log1p normalization
    X_log = X.copy()

    # Check if already log-transformed
    if np.max(X_log.data) > 50:  # Arbitrary threshold
        print("Applying log1p transformation")
        X_log.data = np.log1p(X_log.data)

    n_cells, n_genes = X_log.shape
    Xg = X_log.T.tocsr()

    means = np.zeros(n_genes)
    stds = np.zeros(n_genes)
    medians = np.zeros(n_genes)
    nz_counts = np.zeros(n_genes)
    missing_ratio = np.zeros(n_genes)

    for i in range(n_genes):
        row = Xg.getrow(i).toarray().ravel()
        nz = row[row > 0]
        nz_count = len(nz)
        nz_counts[i] = nz_count
        missing_ratio[i] = 1 - (nz_count / n_cells)
        if nz_count > 0:
            means[i] = nz.mean()
            stds[i] = nz.std(ddof=0)
            medians[i] = np.median(nz)
        else:
            means[i] = 0
            stds[i] = 0
            medians[i] = 0

    df = pd.DataFrame({
        'mean_log': means,
        'std_log': stds,
        'median_log': medians,
        'missing_ratio': missing_ratio,
        'n_nonzero': nz_counts.astype(int)
    })
    return df


def process(path, out_csv):
    X, genes = load_matrix(path)
    X = to_csr(X)

    stats = gene_stats(X)
    if stats.shape[0] != len(genes):
        raise ValueError(f"Mismatch: {stats.shape[0]} stats vs {len(genes)} genes")

    stats.index = genes
    stats.to_csv(out_csv, sep='\t')
    print('Wrote', out_csv)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python compute_sc_stats.py <input_path> <output_csv>')
        sys.exit(1)
    process(sys.argv[1], sys.argv[2])
    # process("data/Simulated_data_v2.tsv", "data/gene_stats_output.tsv")
