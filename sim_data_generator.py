import json
import sys

import numpy as np
import pandas as pd


def generate_prior(
        n_genes: int,
        n_tfs: int,
        min_num_of_targets_per_tf: int,
        max_num_of_targets_per_tf: int,
) -> pd.DataFrame:
    print("Generating prior data...")

    if n_genes <= 0 or n_tfs <= 0:
        raise ValueError("n_genes and n_tfs must be positive.")
    if not (0 <= min_num_of_targets_per_tf <= max_num_of_targets_per_tf):
        raise ValueError(
            "min_num_of_targets_per_tf must be <= max_num_of_targets_per_tf and >= 0."
        )
    if min_num_of_targets_per_tf > n_genes:
        raise ValueError("min_num_of_targets_per_tf cannot exceed n_genes.")
    if max_num_of_targets_per_tf > n_genes:
        max_num_of_targets_per_tf = n_genes  # clamp to valid range

    rng = np.random.default_rng()

    tfs = [f"TF_{i + 1}" for i in range(n_tfs)]
    genes = [f"G_{i + 1}" for i in range(n_genes)]

    rows = []
    gene_indices = np.arange(n_genes)

    for tf in tfs:
        # Number of up/down targets for this TF
        n_up = int(
            rng.integers(min_num_of_targets_per_tf, max_num_of_targets_per_tf + 1)
        )
        n_down = int(
            rng.integers(min_num_of_targets_per_tf, max_num_of_targets_per_tf + 1)
        )

        # Sample up and down targets without overlap
        up_idx = set(rng.choice(gene_indices, size=n_up, replace=False).tolist())

        remaining = np.array(list(set(gene_indices) - up_idx))
        if n_down > remaining.size:
            n_down = remaining.size  # clamp down to avoid overdraw

        down_idx = set(rng.choice(remaining, size=n_down, replace=False).tolist())

        for gi in up_idx:
            rows.append((tf, genes[gi], 1))
        for gi in down_idx:
            rows.append((tf, genes[gi], -1))

    df = pd.DataFrame(rows, columns=["TF", "target", "weight"])

    # Ensure uniqueness (defensive; should already be unique by construction)
    df = df.drop_duplicates(subset=["TF", "target", "weight"], ignore_index=True)

    return df


def generate_ground_truth(
        n_cells: int,
        n_tfs: int,
        activation_prob: float = 0.05,
        inactivation_prob: float = 0.05,
        random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Create a (n_cells x n_tfs) matrix initialized to 0.
    Randomly set a subset to +1 (activated) and a disjoint subset to -1 (inactivated).
    No cell–TF pair gets both; conflicts are resolved by random tie-break.
    """
    if n_cells <= 0 or n_tfs <= 0:
        raise ValueError("n_cells and n_tfs must be positive.")
    if not (0.0 <= activation_prob <= 1.0 and 0.0 <= inactivation_prob <= 1.0):
        raise ValueError("activation_prob and inactivation_prob must be in [0, 1].")

    rng = np.random.default_rng(random_seed)

    # Start with all zeros
    gt = np.zeros((n_cells, n_tfs), dtype=np.int8)

    # Independent Bernoulli draws for activation/inactivation
    act_mask = rng.random((n_cells, n_tfs)) < activation_prob
    inact_mask = rng.random((n_cells, n_tfs)) < inactivation_prob

    # Resolve overlaps: randomly keep either activation or inactivation where both are True
    overlap = act_mask & inact_mask
    if overlap.any():
        # Random tie-break: True -> keep activation, False -> keep inactivation
        keep_act = rng.random(overlap.sum()) < 0.5
        # Flatten indices of overlaps
        ov_i, ov_j = np.where(overlap)
        # For positions where we keep activation, clear inactivation; else clear activation
        act_keep_idx = (ov_i[keep_act], ov_j[keep_act])
        inact_keep_idx = (ov_i[~keep_act], ov_j[~keep_act])

        # Clear the opposite flags
        inact_mask[act_keep_idx] = False
        act_mask[inact_keep_idx] = False

    # Assign values
    gt[act_mask] = 1
    gt[inact_mask] = -1

    # Label axes
    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    tf_names = [f"TF_{j + 1}" for j in range(n_tfs)]
    return pd.DataFrame(gt, index=cell_ids, columns=tf_names)


#################################################
# Expression is drawn from a Gaussian distribution (np.random.normal).
# Values are continuous and smooth.
#################################################
def generate_gene_expression(
        n_cells: int,
        n_genes: int,
        include_tfs_in_expression: bool,
        prediction_difficulty: str,
        missing_percentage: int,
        random_seed: int,
        prior_dfs: pd.DataFrame,
        ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    # Validate shapes
    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")

    # Difficulty -> effect size in units of baseline standard deviation
    # Easy => stronger signal; Hard => weaker signal
    difficulty = str(prediction_difficulty).lower()
    diff_to_sd = {
        "supereasy": 4.0,
        "easy": 2.0,
        "medium": 1.0,
        "hard": 0.5,
        "superhard": 0.25
    }

    if difficulty not in diff_to_sd:
        raise ValueError("prediction_difficulty must be one of {'supereasy','easy','medium','hard','superhard'}.")

    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")
    missing_ratio = float(missing_percentage) / 100.0

    rng = np.random.default_rng(random_seed)

    # Construct gene list
    gene_names = [f"G_{i + 1}" for i in range(n_genes)]

    # Optionally include TFs as additional expression features
    tf_names = list(ground_truth_dfs.columns)
    all_feature_names = list(gene_names)
    if include_tfs_in_expression:
        all_feature_names = gene_names + tf_names

    # Baseline expression generation:
    # - Use a normal distribution with positive clipping to emulate normalized expression
    # - Mean chosen to be moderately expressed; sd gives variability
    baseline_mean = 10.0
    baseline_sd = 2.5

    # Draw baseline for all cells x features
    baseline = rng.normal(loc=baseline_mean, scale=baseline_sd,
                          size=(n_cells, len(all_feature_names)))
    # Clip to non-negative values
    baseline = np.clip(baseline, a_min=0.0, a_max=None)

    # Prepare output frame
    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    expr = pd.DataFrame(baseline, index=cell_ids, columns=all_feature_names)

    # Build TF -> targets by sign mapping from prior_df
    # prior_df columns: ["TF", "target", "weight"] where weight in {1 (up), -1 (down)}
    tf_to_up = {}
    tf_to_down = {}

    # Ensure targets exist among gene names only (regulatory targets are genes)
    gene_set = set(gene_names)

    for tf, tgt, w in prior_dfs[["TF", "target", "weight"]].itertuples(index=False):
        if tgt not in gene_set:
            continue  # ignore targets outside the simulated gene list
        if w == 1:
            tf_to_up.setdefault(tf, []).append(tgt)
        elif w == -1:
            tf_to_down.setdefault(tf, []).append(tgt)

    # Determine regulation step size
    step = diff_to_sd[difficulty] * baseline_sd

    # Apply regulation per cell–TF using ground_truth_df values in {-1, 0, 1}
    # +1 (activation): up targets increase, down targets decrease
    # -1 (inactivation): up targets decrease, down targets increase
    for ci, cell in enumerate(cell_ids):
        # Access the row once for speed
        tf_states = ground_truth_dfs.iloc[ci, :] if ci < len(ground_truth_dfs.index) else ground_truth_dfs.iloc[0, :]
        for tf in tf_names:
            state = tf_states.get(tf, 0)
            if state == 0:
                continue

            ups = tf_to_up.get(tf, [])
            downs = tf_to_down.get(tf, [])

            if state == 1:
                # Activation
                if ups:
                    expr.loc[cell, ups] = np.clip(expr.loc[cell, ups].to_numpy() + step, a_min=0.0, a_max=None)
                if downs:
                    expr.loc[cell, downs] = np.clip(expr.loc[cell, downs].to_numpy() - step, a_min=0.0, a_max=None)
            elif state == -1:
                # Inactivation
                if ups:
                    expr.loc[cell, ups] = np.clip(expr.loc[cell, ups].to_numpy() - step, a_min=0.0, a_max=None)
                if downs:
                    expr.loc[cell, downs] = np.clip(expr.loc[cell, downs].to_numpy() + step, a_min=0.0, a_max=None)

    # Final non-negativity safeguard
    expr[all_feature_names] = np.clip(expr[all_feature_names].to_numpy(), a_min=0.0, a_max=None)

    # Apply dropout (zero-inflation) to match target missing percentage
    # Use Beta distribution to get per-gene dropout probabilities centered near missing_ratio
    a, b = (missing_ratio * 20.0, (1.0 - missing_ratio) * 20.0)
    per_gene_dropout = rng.beta(a, b, size=len(all_feature_names))  # shape (features,)
    drop_mask = rng.random(size=expr.shape) < per_gene_dropout[None, :]
    expr = expr * (~drop_mask)

    return expr


#################################################
# Expression is drawn from a Negative Binomial distribution (Gamma-Poisson).
# Values are counts with realistic sparsity and variability.
#################################################
def generate_gene_expression2(
        n_cells: int,
        n_genes: int,
        include_tfs_in_expression: bool,
        prediction_difficulty: str,
        missing_percentage: int,
        random_seed: int,
        prior_dfs: pd.DataFrame,
        ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    # Validate inputs
    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")

    # Difficulty controls both effect size and sparsity (dropout)
    prediction_difficulty = str(prediction_difficulty).lower()

    diff_effect = {
        "supereasy": 4.0,
        "easy": 2.0,
        "medium": 1.0,
        "hard": 0.5,
        "superhard": 0.25
    }  # multiplicative fold-change magnitude

    if prediction_difficulty not in diff_effect:
        raise ValueError("prediction_difficulty must be one of {'supereasy','easy','medium','hard','superhard'}.")

    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")
    missing_ratio = float(missing_percentage) / 100.0

    rng = np.random.default_rng(random_seed)

    # Names
    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = [f"G_{i + 1}" for i in range(n_genes)]
    tf_names = list(ground_truth_dfs.columns)

    # ------------------------------------------
    # 1) Simulate raw counts with NB (Gamma-Poisson)
    # ------------------------------------------
    # Per-gene propensities (unnormalized), heavy-tailed like real data
    gene_prop = rng.lognormal(mean=0.0, sigma=1.0, size=n_genes)
    gene_prop = gene_prop / gene_prop.sum()  # convert to proportions that sum to 1

    # Per-cell library sizes (UMI totals), lognormal variability
    target_libsize = 1.0e4  # typical 10k UMIs per cell median
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.35, size=n_cells)

    # Mean matrix (cells x genes): expected counts before dispersion
    mean_mat = np.outer(libsize, gene_prop)  # shape (n_cells, n_genes)

    # Negative binomial via Gamma-Poisson:
    #   rate_ij ~ Gamma(shape=theta, scale=mean/theta), count_ij ~ Poisson(rate_ij)
    # Choose moderate overdispersion; smaller theta => more variance
    theta = 5.0
    gamma_shape = theta
    gamma_scale = mean_mat / theta
    rate = rng.gamma(shape=gamma_shape, scale=gamma_scale)  # same shape as mean_mat
    counts = rng.poisson(rate)

    # ------------------------------------------
    # 2) Dropout (zero-inflation) to match sparsity
    # ------------------------------------------
    # Beta simulated_data to center near target_zero with moderate spread
    a, b = (missing_ratio * 20.0, (1.0 - missing_ratio) * 20.0)
    per_gene_dropout = rng.beta(a, b, size=n_genes)  # shape (genes,)

    # Apply dropout mask (Bernoulli per cell-gene, using gene-specific probs)
    drop_mask = rng.random(size=counts.shape) < per_gene_dropout[None, :]
    counts = counts * (~drop_mask)

    # ------------------------------------------
    # 3) Regulatory effects from TF states
    # ------------------------------------------
    # Build maps TF -> up/down targets (only among simulated genes)
    gene_set = set(gene_names)
    tf_to_up, tf_to_down = {}, {}
    for tf, tgt, w in prior_dfs[["TF", "target", "weight"]].itertuples(index=False):
        if tgt not in gene_set:
            continue
        if w == 1:
            tf_to_up.setdefault(tf, []).append(tgt)
        elif w == -1:
            tf_to_down.setdefault(tf, []).append(tgt)

    # Convert counts to DataFrame for easier column selection during regulation
    expr_counts = pd.DataFrame(counts, index=cell_ids, columns=gene_names)

    # Effect size as multiplicative factor (e.g., 1.25 means +25%)
    alpha = diff_effect[prediction_difficulty]
    up_act_factor = 1.0 + alpha
    down_act_factor = 1.0 - alpha  # keep non-negative by rounding later
    up_inact_factor = 1.0 - alpha
    down_inact_factor = 1.0 + alpha

    # Apply per-cell regulations
    for ci, cell in enumerate(cell_ids):
        tf_states = ground_truth_dfs.iloc[ci, :] if ci < len(ground_truth_dfs.index) else ground_truth_dfs.iloc[0, :]
        for tf in tf_names:
            state = tf_states.get(tf, 0)
            if state == 0:
                continue

            ups = tf_to_up.get(tf, [])
            downs = tf_to_down.get(tf, [])

            if state == 1:
                # Activation: up targets increase; down targets decrease
                if ups:
                    expr_counts.loc[cell, ups] = np.floor(expr_counts.loc[cell, ups].to_numpy() * up_act_factor)
                if downs:
                    expr_counts.loc[cell, downs] = np.floor(expr_counts.loc[cell, downs].to_numpy() * down_act_factor)
            elif state == -1:
                # Inactivation: up targets decrease; down targets increase
                if ups:
                    expr_counts.loc[cell, ups] = np.floor(expr_counts.loc[cell, ups].to_numpy() * up_inact_factor)
                if downs:
                    expr_counts.loc[cell, downs] = np.floor(expr_counts.loc[cell, downs].to_numpy() * down_inact_factor)

    # Ensure non-negative integer counts
    expr_counts[gene_names] = np.clip(expr_counts[gene_names].to_numpy(), a_min=0, a_max=None).astype(int)

    # ------------------------------------------
    # 4) Normalize and log-transform to realistic scale
    # ------------------------------------------
    # Counts-per-10k (CP10K) then log1p; this yields values mostly in [0, ~6]
    lib_after = expr_counts.sum(axis=1).to_numpy()
    lib_after[lib_after == 0] = 1  # avoid division by zero
    cp10k = (expr_counts.to_numpy() / lib_after[:, None]) * 1.0e4
    log_expr = np.log1p(cp10k)

    expr = pd.DataFrame(log_expr, index=cell_ids, columns=gene_names)

    # Optionally include TFs as expression features: simulate similarly sparse, weak signal
    if include_tfs_in_expression:
        tf_cols = []
        # Small independent noise for TF "expression"
        tf_counts = rng.poisson(0.1, size=(n_cells, len(tf_names)))
        # add mild dropout to keep mostly zeros
        tf_drop = rng.random(size=tf_counts.shape) < 0.8
        tf_counts = tf_counts * (~tf_drop)
        tf_cp10k = (tf_counts / lib_after[:, None]) * 1.0e4
        tf_log = np.log1p(tf_cp10k)
        for j, tf in enumerate(tf_names):
            tf_cols.append(pd.Series(tf_log[:, j], index=cell_ids, name=tf))
        tf_df = pd.concat(tf_cols, axis=1)
        expr = pd.concat([expr, tf_df], axis=1)

    return expr


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python sim_data_generator.py simulated_data.json")
        sys.exit(1)

    param_file = sys.argv[1]
    # param_file = "simulated_data/simulated_data.json"

    with open(param_file, "r") as f:
        params = json.load(f)
        print(params)

    prior_df = generate_prior(
        params["n_genes"],
        params["n_tfs"],
        params["min_num_of_targets_per_tf"],
        params["max_num_of_targets_per_tf"],
    )

    ground_truth_df = generate_ground_truth(
        params["n_cells"],
        params["n_tfs"],
        activation_prob=params["ground_truth_activation_prob"],
        inactivation_prob=params["ground_truth_inactivation_prob"],
        random_seed=params["random_seed"],
    )

    gene_exp = generate_gene_expression(
        params["n_cells"],
        params["n_genes"],
        params["include_tfs_in_expression"],
        params["prediction_difficulty"],
        params["missing_percentage"],
        params["random_seed"],
        prior_df,
        ground_truth_df
    )

    # move target column to the end for easier viewing
    prior_df = prior_df[["TF", "weight", "target"]]
    prior_df["weight"] = prior_df["weight"].map({1: "upregulates-expression", -1: "downregulates-expression"})
    prior_df.to_csv(f"{params["output_dir"]}/{params["output_prior_file"]}", sep="\t", index=False, header=False)
    print(f"Wrote prior to {params['output_prior_file']}")

    ground_truth_df.to_csv(f"{params['output_dir']}/{params['output_ground_truth_file']}", sep="\t", index=True)
    print(f"Wrote ground truth to {params['output_ground_truth_file']}")

    gene_exp.to_csv(f"{params['output_dir']}/{params['output_exp_file']}", sep="\t", index=True)
    print(f"Wrote gene expression to {params['output_exp_file']}")
