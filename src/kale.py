import argparse
import math

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import issparse
from scipy.stats import norm, zscore
from scipy.stats import rankdata
from tqdm.auto import tqdm


def std_dev_mean_norm_rank(n_population: int, k_sample: int) -> float:
    """Calculates the standard deviation of the mean of k ranks drawn from n (unweighted)."""
    if not (isinstance(n_population, int) and n_population > 0):
        raise ValueError("n_population must be a positive integer.")
    if not (isinstance(k_sample, int) and k_sample > 0):
        raise ValueError("k_sample must be a positive integer.")
    if k_sample > n_population:
        raise ValueError(
            "Sample size k_sample cannot exceed population size n_population."
        )

    if k_sample == n_population or n_population == 1:
        return 0.0

    var = ((n_population + 1) * (n_population - k_sample)) / (
        12 * (n_population**2) * k_sample
    )
    return math.sqrt(max(var, 0.0))


def std_dev_weighted_mean_norm_rank(n_population: int, weights: np.ndarray) -> float:
    """Calculates the standard deviation of the weighted mean of k ranks drawn from n."""
    if not (isinstance(n_population, int) and n_population > 0):
        raise ValueError("n_population must be a positive integer.")
    if not isinstance(weights, np.ndarray) or weights.size == 0:
        raise ValueError("weights must be a non-empty numpy array.")

    if n_population == 1:
        return 0.0

    sum_sq_weights = np.sum(np.square(weights))
    numerator = (n_population + 1) * ((n_population * sum_sq_weights) - 1)
    denominator = 12 * (n_population**2)

    if denominator == 0:
        return 0.0

    return math.sqrt(numerator / denominator)


def _process_single_cell_unweighted_ranks(
    cell_name: str,
    expression_series: pd.Series,
    priors_df: pd.DataFrame,
    min_targets: int = 0,
    scores_series: pd.Series = None,
    weighted_power_factor: int = None,
) -> tuple[str, list, list, list]:
    """Processes a single cell using the UNWEIGHTED mean rank method."""
    expression_series.replace(0.0, np.nan, inplace=True)
    expr = expression_series.dropna().sort_values(ascending=False)
    n_genes = expr.size

    if n_genes == 0:
        return cell_name, [], [], []

    ranks_raw = rankdata(
        -expr.values, method="average"
    )  # Ranks in descending order (tie handled by average)
    ranks_norm = (ranks_raw - 0.5) / n_genes
    rank_df = pd.DataFrame({"target": expr.index, "Rank": ranks_norm})

    p = priors_df.merge(rank_df, on="target", how="left").dropna(subset=["Rank"])
    p["AdjustedRank"] = np.where(p["weight"] < 0, 1 - p["Rank"], p["Rank"])

    tf_summary = (
        p.groupby("source", observed=True)
        .agg(
            AvailableTargets=("AdjustedRank", "size"),
            RankMean=("AdjustedRank", "mean"),
        )
        .reset_index()
    )

    tf_summary = tf_summary[tf_summary["AvailableTargets"] > min_targets]
    if tf_summary.empty:
        return cell_name, [], [], []

    tf_summary["ActivationDir"] = np.where(tf_summary["RankMean"] < 0.5, 1, -1)

    sigma = tf_summary["AvailableTargets"].apply(
        lambda k: std_dev_mean_norm_rank(n_genes, k)
    )
    sigma = sigma.replace(0, np.nan)
    tf_summary["Z"] = (tf_summary["RankMean"] - 0.5) / sigma

    # tf_summary["P_two_tailed"] = 2 * norm.sf(np.abs(tf_summary["Z"]))
    tf_summary["P_two_tailed"] = np.where(
        tf_summary["RankMean"] < 0.5,
        2 * norm.cdf(tf_summary["Z"]),
        2 * (1 - norm.cdf(tf_summary["Z"])),
    )

    return (
        cell_name,
        tf_summary["source"].tolist(),
        tf_summary["P_two_tailed"].tolist(),
        tf_summary["ActivationDir"].tolist(),
    )


def _process_single_cell_weighted_ranks(
    cell_name: str,
    expression_series: pd.Series,
    priors_df: pd.DataFrame,
    min_targets: int = 0,
    scores_series: pd.Series = None,
    weighted_power_factor: int = None,
) -> tuple[str, list, list, list]:
    """Processes a single cell using the WEIGHTED mean rank method."""
    expression_series.replace(0.0, np.nan, inplace=True)
    expr = expression_series.dropna().sort_values(ascending=False)
    n_genes = expr.size

    if n_genes == 0:
        return cell_name, [], [], []

    # Calculate weight_factors if scores_series is provided
    if scores_series is not None and weighted_power_factor is not None:
        # Use power_scale, as a power to scale the weights, to amplify the differences
        priors_df["scores"] = priors_df["source"].map(scores_series.abs())
        priors_df["scores_power_raised"] = priors_df["scores"] ** weighted_power_factor
        priors_df["weight_factor"] = priors_df["scores_power_raised"] / priors_df.groupby("target")["scores_power_raised"].transform("sum")

    ranks_raw = rankdata(-expr.values, method="average")  # Ranks in descending order
    ranks_norm = (ranks_raw - 0.5) / n_genes
    rank_df = pd.DataFrame({"target": expr.index, "Rank": ranks_norm})

    p = priors_df.merge(rank_df, on="target", how="left").dropna(subset=["Rank"])
    p["AdjustedRank"] = np.where(p["weight"] < 0, 1 - p["Rank"], p["Rank"])

    # Step 1: Group the DataFrame by 'source' (transcription factor)
    grouped = p.groupby("source", observed=True)

    # Step 2: For each group, calculate weighted statistics
    def calculate_tf_stats(grp):
        # Step 2a: Count how many targets are available for this TF
        num_targets = len(grp)

        # Step 2b: Calculate normalized weights for this group
        normalized_weights = grp["weight_factor"] / grp["weight_factor"].sum()

        # Step 2c: Calculate weighted mean of AdjustedRank
        # This gives more importance to targets with higher weight_factor
        weighted_mean_rank = np.average(grp["AdjustedRank"], weights=normalized_weights)

        # Step 2d: Calculate standard deviation using the normalized weights
        sigma_n_k = std_dev_weighted_mean_norm_rank(n_genes, normalized_weights.values)

        # Step 2e: Return a Series with all calculated statistics
        value = pd.Series(
            {
                "AvailableTargets": num_targets,
                "RankMean": weighted_mean_rank,
                "Sigma_n_k": sigma_n_k,
            }
        )
        return value

    # Step 3: Apply the function to each group
    tf_summary = grouped.apply(
        calculate_tf_stats, include_groups=False
    )  # Don't include the group key in the result)

    # Step 4: Reset index to convert the grouped result back to a regular DataFrame
    tf_summary = tf_summary.reset_index()

    tf_summary = tf_summary[tf_summary["AvailableTargets"] > min_targets]
    if tf_summary.empty:
        return cell_name, [], [], []

    tf_summary["ActivationDir"] = np.where(tf_summary["RankMean"] < 0.5, 1, -1)
    # tf_summary["Z"] = (tf_summary["RankMean"] - 0.5) / tf_summary["Sigma_n_k"].replace(0, np.nan)
    tf_summary["Z"] = (tf_summary["RankMean"] - 0.5) / tf_summary["Sigma_n_k"]

    # tf_summary["P_two_tailed"] = 2 * norm.sf(np.abs(tf_summary["Z"]))
    tf_summary["P_two_tailed"] = np.where(
        tf_summary["RankMean"] < 0.5,
        2 * norm.cdf(tf_summary["Z"]),
        2 * (1 - norm.cdf(tf_summary["Z"])),
    )

    return (
        cell_name,
        tf_summary["source"].tolist(),
        tf_summary["P_two_tailed"].tolist(),
        tf_summary["ActivationDir"].tolist(),
    )


def _process_single_cell_stouffer(
    cell_name: str, expression_z_scores: pd.Series, priors_df: pd.DataFrame
) -> tuple[str, list, list, list]:
    priors_group = priors_df.groupby("source", observed=True).agg(
        {"target": list, "weight": list}
    )

    valid_z_scores = expression_z_scores.dropna()
    if valid_z_scores.empty:
        return cell_name, [], [], []

    z_score_map = valid_z_scores.to_dict()
    regulators, p_values, directions = [], [], []

    # Iterate through the pre-grouped priors
    for tf, data in priors_group.iterrows():
        target_genes = data["target"]
        effects = data["weight"]

        evidence_z = []
        for i, gene in enumerate(target_genes):
            z = z_score_map.get(gene)
            if z is not None:
                evidence_z.append(-z if effects[i] < 0 else z)

        k = len(evidence_z)

        # Calculate Stouffer's Z
        z_sum = np.sum(evidence_z)
        stouffer_z = z_sum / np.sqrt(k) if k > 0 else 0

        # Use a two-tailed test for the p-value
        p_value = 2 * norm.sf(abs(stouffer_z))

        # Direction is the sign of the combined evidence
        direction = np.sign(stouffer_z)

        regulators.append(tf)
        p_values.append(p_value)
        directions.append(direction)

    return (cell_name, regulators, p_values, directions)


def run_tf_analysis(
    adata: AnnData,
    priors: pd.DataFrame,
    ignore_zeros: bool,
    min_targets: int,
    analysis_method: str,
    weighted: bool,
    weighted_power_factor:int,
    cores: int,
    scores: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if adata is None or adata.n_obs == 0:
            raise ValueError("Input data is empty or invalid")

        X = adata.X.toarray() if issparse(adata.X) else adata.X.copy()

        # Filter priors based on min_targets and presence in the data
        priors = priors[priors["target"].isin(adata.var_names)]
        priors = priors.groupby("source").filter(lambda x: len(x) >= min_targets)

        match analysis_method:
            case "ranks_from_zscore":
                print("Calculating Z-scores as required by the selected method...")
                if ignore_zeros:
                    X[X == 0] = np.nan
                    z_mat = zscore(X, axis=0, nan_policy="omit")
                else:
                    z_mat = zscore(X, axis=0)
                data_for_processing = pd.DataFrame(
                    z_mat, index=adata.obs_names, columns=adata.var_names
                )

            case "rank_of_ranks":
                print("Using raw gene expression as input for per-cell ranking...")
                data_for_processing = pd.DataFrame(
                    X, index=adata.obs_names, columns=adata.var_names, dtype=float
                )
                if ignore_zeros:
                    mask = data_for_processing != 0
                    data_for_processing[mask] = data_for_processing[mask].rank(
                        axis=0, method="average", ascending=True
                    )
                    data_for_processing[~mask] = 0
                else:
                    data_for_processing = data_for_processing.rank(
                        axis=0, method="average", ascending=True
                    )

            # case "stouffers_zscore":
            #     print("Using Z-scores as input for Stouffer's method...")
            #     process_func = _process_single_cell_stouffer
            #     # TODO: Stouffer's method does not work as of now.

            case _:
                raise ValueError(
                    f"Unknown analysis method family for: {analysis_method}"
                )

        # Step 2: Select the appropriate worker function
        if weighted:
            print("Calculating weighted gene expression as input for per-cell ranking...")
            process_func = _process_single_cell_weighted_ranks
            target_source_counts = priors.groupby('target')['source'].nunique()
            priors['weight_factor'] = priors['target'].map(target_source_counts).apply(lambda x: 1 / x)
        else:
            process_func = _process_single_cell_unweighted_ranks

        if process_func is None or data_for_processing is None:
            raise RuntimeError(
                "Failed to select a processing function or prepare data."
            )

        # ───── Parallel processing of cells ───────────────────────────────────────
        if scores is not None:
            print(
                f"Starting TF activity using {cores if cores > 0 else 'all available'} cores."
            )
            tasks = [
                delayed(process_func)(
                    cell_name,
                    data_for_processing.loc[cell_name],
                    priors,
                    min_targets,
                    scores.loc[cell_name],
                    weighted_power_factor,
                )
                for cell_name in data_for_processing.index
            ]

            if cores == 1:
                print("Running sequentially (CORES_USED=1)...")
                cell_results_list = [
                    process_func(
                        cell_name,
                        data_for_processing.loc[cell_name],
                        priors,
                        min_targets,
                        scores.loc[cell_name],
                        weighted_power_factor,
                    )
                    for cell_name in tqdm(
                        data_for_processing.index, desc="Processing cells in sequence"
                    )
                ]
            else:
                print(f"Running in parallel with CORES_USED={cores}.")
                cell_results_list = Parallel(n_jobs=cores, backend="loky", verbose=1)(
                    tqdm(tasks, desc="Processing cells in parallel", total=len(tasks))
                )
        else:
            print(
                f"Starting TF activity using {cores if cores > 0 else 'all available'} cores."
            )
            tasks = [
                delayed(process_func)(
                    cell_name,
                    data_for_processing.loc[cell_name],
                    priors,
                    min_targets
                )
                for cell_name in data_for_processing.index
            ]

            if cores == 1:
                print("Running sequentially (CORES_USED=1)...")
                cell_results_list = [
                    process_func(
                        cell_name,
                        data_for_processing.loc[cell_name],
                        priors,
                        min_targets,
                    )
                    for cell_name in tqdm(
                        data_for_processing.index, desc="Processing cells in sequence"
                    )
                ]
            else:
                print(f"Running in parallel with CORES_USED={cores}.")
                cell_results_list = Parallel(n_jobs=cores, backend="loky", verbose=1)(
                    tqdm(tasks, desc="Processing cells in parallel", total=len(tasks))
                )

        # print(f"Starting TF activity using {cores if cores > 0 else 'all available'} cores.")
        # tasks = [
        #     delayed(process_func)(cell_name, data_for_processing.loc[cell_name], priors, min_targets)
        #     for cell_name in data_for_processing.index
        # ]
        #
        # if cores == 1:
        #     print("Running sequentially (CORES_USED=1)...")
        #     cell_results_list = [
        #         process_func(cell_name, data_for_processing.loc[cell_name], priors, min_targets)
        #         for cell_name in tqdm(data_for_processing.index, desc="Processing cells in sequence")
        #     ]
        # else:
        #     print(f"Running in parallel with CORES_USED={cores}.")
        #     cell_results_list = Parallel(n_jobs=cores, backend="loky", verbose=1)(
        #         tqdm(tasks, desc="Processing cells in parallel", total=len(tasks))
        #     )

        # ───── Aggregate results into two separate DataFrames ───────────────────
        print("\nAggregating results...")
        records = [
            (cell, reg, pval, direc)
            for cell, regs, pvals, direcs in cell_results_list
            for reg, pval, direc in zip(regs, pvals, direcs)
        ]

        if not records:
            print(
                "Warning: No TF activities could be computed. Returning empty DataFrame."
            )
            return pd.DataFrame(index=adata.obs_names), pd.DataFrame(
                index=adata.obs_names
            )

        results_df = pd.DataFrame(
            records, columns=["cell", "regulator", "p_value", "direction"]
        )
        pvalue_df = results_df.pivot(
            index="cell", columns="regulator", values="p_value"
        )
        activation_df = results_df.pivot(
            index="cell", columns="regulator", values="direction"
        )

        pvalue_df = pvalue_df.reindex(adata.obs_names)
        activation_df = activation_df.reindex(adata.obs_names)

        scores = (
            -np.log(np.clip(pvalue_df.to_numpy(), 1e-300, None))
            * activation_df.to_numpy()
        )
        scores = pd.DataFrame(scores, index=pvalue_df.index, columns=pvalue_df.columns)

        print("kale completed")
        return scores, pvalue_df

    except Exception as e:
        print(f"Error during TF analysis: {e}")
        raise


def func_kale(
    adata: AnnData = None,
    net: pd.DataFrame = None,
    method: str = "rank_of_ranks",
    weighted: bool = False,
    weighted_power_factor: int = 0,
    min_targets: int = 1,
    ignore_zeros: bool = True,
    cores: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Check that adata and net are provided, since they are now the primary inputs
    if adata is None or net is None:
        raise ValueError("`adata` and `net` must be provided for the kale method.")

    if weighted:
        net["weight_factor"] = 1

    scores, pvalues = run_tf_analysis(
        adata,
        net,
        ignore_zeros,
        min_targets=min_targets,
        analysis_method=method,
        weighted=weighted,
        weighted_power_factor=weighted_power_factor,
        cores=cores,
    )

    # Fill NaNs with 0 in scores (no activity detected)
    scores.fillna(0)
    pvalues.fillna(1)

    # Ensure the final output aligns with all sources in the net, filling missing with NaN
    all_sources = net["source"].unique()
    scores = scores.reindex(columns=all_sources, fill_value=np.nan)
    pvalues = pvalues.reindex(columns=all_sources, fill_value=np.nan)

    # Sophisticated way to calculating weight_factor based on scores
    # Then run analysis again with updated weight_factors
    # Two-pass approach to incorporate dynamic weighting based on initial scores
    if weighted and weighted_power_factor > 0:
        print("-" * 150)
        print(
            "Refining TF activity scores with dynamic weighting based on initial results..."
        )
        scores, pvalues = run_tf_analysis(
            adata,
            net,
            ignore_zeros,
            min_targets=min_targets,
            analysis_method=method,
            weighted=weighted,
            weighted_power_factor=weighted_power_factor,
            cores=cores,
            scores=scores,
        )
        scores.fillna(0)
        pvalues.fillna(1)
        all_sources = net["source"].unique()
        scores = scores.reindex(columns=all_sources, fill_value=np.nan)
        pvalues = pvalues.reindex(columns=all_sources, fill_value=np.nan)

    return scores, pvalues


if __name__ == "__main__":

    def str_to_bool(x):
        return x.lower() == "true"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_exp_file", required=True)
    parser.add_argument("--prior_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--pvalue_output_file", required=True)
    parser.add_argument("--ignore_zeros", required=True, type=str_to_bool)
    parser.add_argument("--cores", default=8, type=int)
    parser.add_argument("--method", default="rank_of_ranks")
    parser.add_argument("--min_targets", default=1, type=int)
    parser.add_argument("--weighted", required=False, type=str_to_bool)
    parser.add_argument("--weighted_power_factor", default=1, type=int)
    args = parser.parse_args()

    # gene_exp_file = "simulated_data/simulated_scRNASeq_data.tsv"
    # prior_file = "simulated_data/simulated_prior_data.tsv"
    # output_file = "simulated_data/test_kale_scores.tsv"

    # check gene_exp_file
    if args.gene_exp_file.endswith(".h5ad"):
        adata = sc.read_h5ad(args.gene_exp_file)
    elif args.gene_exp_file.endswith(".tsv"):
        gene_exp = pd.read_csv(args.gene_exp_file, sep="\t", index_col=0)
        adata = sc.AnnData(gene_exp)
    else:
        raise ValueError("gene_exp_file must be either .h5ad or .tsv format")

    effect_map = {"upregulates-expression": 1, "downregulates-expression": -1}

    net = pd.read_csv(
        args.prior_file,
        sep="\t",
        names=["source", "weight", "target"],
        usecols=[0, 1, 2],
        converters={"weight": effect_map.get},
    )[["source", "target", "weight"]]

    score_kale, pvalue_kale = func_kale(
        adata,
        net,
        method=args.method,
        weighted=args.weighted,
        weighted_power_factor=args.weighted_power_factor,
        min_targets=args.min_targets,
        ignore_zeros=args.ignore_zeros,
        cores=args.cores,
    )

    score_kale.to_csv(args.output_file, sep="\t", index=True)
    pvalue_kale.to_csv(args.pvalue_output_file, sep="\t", index=True)

    print("Kale TF activity scores and p-values have been saved.")
