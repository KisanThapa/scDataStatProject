import numpy as np
import pandas as pd

if __name__ == "__main__":
    gene_exp = pd.read_csv("simulated_data/simulated_scRNASeq_data.tsv", sep="\t", index_col=0)
    ground_truth_df = pd.read_csv("simulated_data/simulated_ground_truth.tsv", sep="\t", index_col=0)
    prior_df = pd.read_csv("simulated_data/simulated_prior_data.tsv", sep="\t", names=["TF", "weight", "target"])

    prior_map = {"upregulates-expression": 1, "downregulates-expression": -1}
    prior_df['weight'] = prior_df['weight'].map(prior_map)

    tf_names = ground_truth_df.columns.tolist()

    # Correlation of TF activity with its up targets
    correlations = []
    for tf in tf_names:
        act = ground_truth_df[tf]
        ups = prior_df.loc[(prior_df['TF'] == tf) & (prior_df['weight'] == 1), 'target']
        mean_expr = gene_exp[ups].mean(axis=1)
        corr = np.corrcoef(act, mean_expr)[0, 1]
        # print(tf, corr)
        correlations.append(corr)

    print(f"Mean correlation of TF activity with its up targets: {np.nanmean(correlations)}")
