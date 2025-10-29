import json
import sys

import numpy as np
import pandas as pd


def generate_prior(
        n_genes: int,
        n_tfs: int,
        min_num_of_targets_per_tf: int,
        max_num_of_targets_per_tf: int,
        random_seed: int | None = None,
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

    rng = np.random.default_rng(random_seed)

    tfs = [f"TF_{i + 1}" for i in range(n_tfs)]
    genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    rows_tf = []
    rows_target = []
    rows_weight = []

    gene_indices = np.arange(n_genes)
    # Vectorized per-TF construction using one permutation to avoid overlap efficiently
    for tf in tfs:
        n_up = int(rng.integers(min_num_of_targets_per_tf, max_num_of_targets_per_tf + 1))
        n_down = int(rng.integers(min_num_of_targets_per_tf, max_num_of_targets_per_tf + 1))

        # Single permutation, then slice
        perm = rng.permutation(gene_indices)
        if n_up > n_genes:
            n_up = n_genes
        if n_down > n_genes - n_up:
            n_down = n_genes - n_up

        up_idx = perm[:n_up]
        down_idx = perm[n_up: n_up + n_down]

        if n_up:
            rows_tf.extend([tf] * n_up)
            rows_target.extend(genes.take(up_idx))
            rows_weight.extend([1] * n_up)
        if n_down:
            rows_tf.extend([tf] * n_down)
            rows_target.extend(genes.take(down_idx))
            rows_weight.extend([-1] * n_down)

    df = pd.DataFrame({"TF": rows_tf, "target": rows_target, "weight": rows_weight})
    df = df.drop_duplicates(subset=["TF", "target", "weight"], ignore_index=True)
    return df


def generate_prior_lognormal(
        n_genes: int,
        n_tfs: int,
        min_num_of_targets_per_tf: int,
        max_num_of_targets_per_tf: int,
        random_seed: int | None = None,
) -> pd.DataFrame:
    print("Generating prior data...")
    min_num_of_targets_per_tf = int(min_num_of_targets_per_tf / 2)  # adjust for up/down

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

    rng = np.random.default_rng(random_seed)

    tfs = [f"TF_{i + 1}" for i in range(n_tfs)]
    genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    rows_tf = []
    rows_target = []
    rows_weight = []

    gene_indices = np.arange(n_genes)

    # Use log-normal distribution with mu=4, sigma=0.75
    mu = 0
    sigma = 1.7
    # sigma = 0.1

    # Generate target counts from log-normal distribution
    min_targets = max(1, min_num_of_targets_per_tf)  # Ensure at least 1 target
    max_targets = min(n_genes, max_num_of_targets_per_tf)

    # Generate log-normal distributed values
    # log-normal: X ~ exp(N(mu, sigma^2))
    log_normal_up = rng.lognormal(mean=mu, sigma=sigma, size=n_tfs)
    log_normal_down = rng.lognormal(mean=mu, sigma=sigma, size=n_tfs)

    # Scale to desired range [min_targets, max_targets]
    # Map from log-normal range to [min_targets, max_targets]
    n_up_targets = np.clip(
        np.interp(log_normal_up, (log_normal_up.min(), log_normal_up.max()), (min_targets, max_targets)),
        min_targets, max_targets
    ).astype(int)

    n_down_targets = np.clip(
        np.interp(log_normal_down, (log_normal_down.min(), log_normal_down.max()), (min_targets, max_targets)),
        min_targets, max_targets
    ).astype(int)

    # Vectorized per-TF construction using one permutation to avoid overlap efficiently
    for i, tf in enumerate(tfs):
        n_up = n_up_targets[i]
        n_down = n_down_targets[i]

        # Single permutation, then slice
        perm = rng.permutation(gene_indices)
        if n_up > n_genes:
            n_up = n_genes
        if n_down > n_genes - n_up:
            n_down = n_genes - n_up

        up_idx = perm[:n_up]
        down_idx = perm[n_up: n_up + n_down]

        if n_up:
            rows_tf.extend([tf] * n_up)
            rows_target.extend(genes.take(up_idx))
            rows_weight.extend([1] * n_up)
        if n_down:
            rows_tf.extend([tf] * n_down)
            rows_target.extend(genes.take(down_idx))
            rows_weight.extend([-1] * n_down)

    df = pd.DataFrame({"TF": rows_tf, "target": rows_target, "weight": rows_weight})
    df = df.drop_duplicates(subset=["TF", "target", "weight"], ignore_index=True)
    return df


def generate_prior_poisson(
        n_genes: int,
        n_tfs: int,
        lambda_param: float = 100.0,
        random_seed: int | None = None,
) -> pd.DataFrame:
    print("Generating prior data (Poisson targets)...")

    if n_genes <= 0 or n_tfs <= 0:
        raise ValueError("n_genes and n_tfs must be positive.")
    if lambda_param <= 0:
        raise ValueError("lambda_param must be > 0.")

    rng = np.random.default_rng(random_seed)

    tfs = [f"TF_{i + 1}" for i in range(n_tfs)]
    genes = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)

    rows_tf = []
    rows_target = []
    rows_weight = []

    gene_indices = np.arange(n_genes)

    # Draw up/down target counts per TF from Poisson(λ), then clip to [1, n_genes]
    n_up_targets = rng.poisson(lam=lambda_param, size=n_tfs)
    n_down_targets = rng.poisson(lam=lambda_param, size=n_tfs)

    n_up_targets = np.clip(n_up_targets, 1, n_genes).astype(int)
    n_down_targets = np.clip(n_down_targets, 1, n_genes).astype(int)

    for i, tf in enumerate(tfs):
        n_up = int(n_up_targets[i])
        n_down = int(n_down_targets[i])

        # One permutation per TF to avoid overlap; split for up/down
        perm = rng.permutation(gene_indices)
        if n_up > n_genes:
            n_up = n_genes
        if n_down > n_genes - n_up:
            n_down = n_genes - n_up

        up_idx = perm[:n_up]
        down_idx = perm[n_up: n_up + n_down]

        if n_up:
            rows_tf.extend([tf] * n_up)
            rows_target.extend(genes.take(up_idx))
            rows_weight.extend([1] * n_up)
        if n_down:
            rows_tf.extend([tf] * n_down)
            rows_target.extend(genes.take(down_idx))
            rows_weight.extend([-1] * n_down)

    df = pd.DataFrame({"TF": rows_tf, "target": rows_target, "weight": rows_weight})
    df = df.drop_duplicates(subset=["TF", "target", "weight"], ignore_index=True)
    return df


def generate_ground_truth(
        n_cells: int,
        n_tfs: int,
        ground_truth_active_inactive_prob: float = 0.1,
        random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Create a (n_cells x n_tfs) matrix initialized to 0.
    Randomly set a subset to +1 (activated) and a disjoint subset to -1 (inactivated).
    No cell–TF pair gets both; conflicts are resolved by random tie-break.
    """
    if n_cells <= 0 or n_tfs <= 0:
        raise ValueError("n_cells and n_tfs must be positive.")
    if not (0.0 <= ground_truth_active_inactive_prob <= 1.0 and 0.0 <= ground_truth_active_inactive_prob <= 1.0):
        raise ValueError("activation_prob and inactivation_prob must be in [0, 1].")

    rng = np.random.default_rng(random_seed)

    gt = np.zeros((n_cells, n_tfs), dtype=np.int8)

    act_mask = rng.random((n_cells, n_tfs)) < ground_truth_active_inactive_prob
    inact_mask = rng.random((n_cells, n_tfs)) < ground_truth_active_inactive_prob

    overlap = act_mask & inact_mask
    if overlap.any():
        keep_act = rng.random(overlap.sum()) < 0.5
        ov_i, ov_j = np.where(overlap)
        act_keep_idx = (ov_i[keep_act], ov_j[keep_act])
        inact_keep_idx = (ov_i[~keep_act], ov_j[~keep_act])
        inact_mask[act_keep_idx] = False
        act_mask[inact_keep_idx] = False

    gt[act_mask] = 1
    gt[inact_mask] = -1

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    tf_names = [f"TF_{j + 1}" for j in range(n_tfs)]
    return pd.DataFrame(gt, index=cell_ids, columns=tf_names)


#################################################
# Expression is drawn from a Gaussian distribution (np.random.normal).
#
# Base model: Gaussian (Normal(10, 2.5))
# Type of Noise: Multiplicative (× factor)
#################################################
def generate_gene_expression_gaussian(
        n_cells: int,
        n_genes: int,
        include_tfs_in_expression: bool,
        tf_effect_factor: float,
        missing_percentage: int,
        random_seed: int,
        prior_dfs: pd.DataFrame,
        ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")

    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")
    missing_ratio = float(missing_percentage) / 100.0

    rng = np.random.default_rng(random_seed)

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

    # 1) Generate a baseline expression from Gaussian distribution
    baseline_mean = 10.0
    baseline_sd = 2.5
    baseline = rng.normal(loc=baseline_mean, scale=baseline_sd, size=(n_cells, n_genes))
    baseline = np.clip(baseline, a_min=0.0, a_max=None)
    expr = baseline.astype(np.float32, copy=False)

    # 2) Build TF -> target mappings
    gene_index = {g: i for i, g in enumerate(gene_names)}
    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    if not {"TF", "target", "weight"}.issubset(prior_dfs.columns):
        raise ValueError("prior_df must contain columns {'TF','target','weight'}")

    # Convert to numpy arrays once for faster iteration
    prior_tf = prior_dfs["TF"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_weight = prior_dfs["weight"].to_numpy()

    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}

    # Vectorized mapping using boolean indexing
    for tf, tgt, w in zip(prior_tf, prior_target, prior_weight):
        gi = gene_index.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if w == 1:
            tf_to_up_idx[ti].append(gi)
        elif w == -1:
            tf_to_down_idx[ti].append(gi)

    # 3) Apply regulatory effects using multiplicative approach
    tf_factor = tf_effect_factor
    gt_states = ground_truth_dfs.to_numpy(copy=False)

    for ti in range(n_tfs):
        ups = np.array(tf_to_up_idx[ti], dtype=np.int64)
        downs = np.array(tf_to_down_idx[ti], dtype=np.int64)

        if ups.size == 0 and downs.size == 0:
            continue

        states = gt_states[:, ti]
        act_cells = np.where(states == 1)[0]
        inact_cells = np.where(states == -1)[0]

        if act_cells.size:
            if ups.size:
                # +ve targets, TF activated: multiply by factor
                expr[np.ix_(act_cells, ups)] *= tf_factor
            if downs.size:
                # -ve targets, TF activated: divide by factor
                expr[np.ix_(act_cells, downs)] /= tf_factor

        if inact_cells.size:
            if ups.size:
                # +ve targets, TF inactivated: divide by factor
                expr[np.ix_(inact_cells, ups)] /= tf_factor
            if downs.size:
                # -ve targets, TF inactivated: multiply by factor
                expr[np.ix_(inact_cells, downs)] *= tf_factor

    # Ensure non-negativity
    np.maximum(expr, 0.0, out=expr)

    # 4) Dropout (zero-inflation)
    eps = 1e-6
    a = max(missing_ratio, eps) * 20.0
    b = max(1.0 - missing_ratio, eps) * 20.0
    per_gene_dropout = rng.beta(a, b, size=n_genes)
    drop_mask = rng.random(size=expr.shape) < per_gene_dropout[None, :]
    expr = expr * (~drop_mask)

    # 5) Create final DataFrame
    expr_df = pd.DataFrame(expr, index=cell_ids, columns=gene_names)

    # 9) Add TF expression if requested
    if include_tfs_in_expression:
        tf_names_arr = np.array(tf_names, dtype=object)
        # Sparse TF expression (lower baseline)
        tf_baseline = rng.normal(
            loc=baseline_mean * 0.1,
            scale=baseline_sd * 0.5,
            size=(n_cells, len(tf_names_arr)),
        )
        tf_baseline = np.clip(tf_baseline, a_min=0.0, a_max=None)

        # Apply dropout to TFs
        tf_drop = rng.random(size=tf_baseline.shape) < 0.8
        tf_baseline = tf_baseline * (~tf_drop)

        tf_df = pd.DataFrame(tf_baseline, index=cell_ids, columns=tf_names_arr)
        expr_df = pd.concat([expr_df, tf_df], axis=1)

    return expr_df


#################################################
# Expression is drawn from a Negative Binomial distribution (Gamma-Poisson).
#
# Base model: Negative Binomial (mean from Gamma-Poisson with moderate overdispersion)
# Type of Noise: Multiplicative (× factor)
#################################################
def generate_gene_expression_neg_binomial(
        n_cells: int,
        n_genes: int,
        include_tfs_in_expression: bool,
        tf_effect_factor: float,
        missing_percentage: int,
        random_seed: int,
        prior_dfs: pd.DataFrame,
        ground_truth_dfs: pd.DataFrame,
) -> pd.DataFrame:
    print("Generating gene expression data...")

    if n_cells <= 0 or n_genes <= 0:
        raise ValueError("n_cells and n_genes must be positive.")
    if prior_dfs.empty:
        raise ValueError("prior_df is empty.")
    if ground_truth_dfs.empty:
        raise ValueError("ground_truth_df is empty.")
    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")

    missing_ratio = float(missing_percentage) / 100.0

    rng = np.random.default_rng(random_seed)

    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

    # 1) Generate gene propensities (don't normalize yet)
    # gene_prop = rng.lognormal(mean=0, sigma=0.75, size=n_genes)
    gene_prop = rng.lognormal(mean=0, sigma=1, size=n_genes)

    # 2) Apply gamma directly to create cell-to-cell variability
    theta = 5.0
    gamma_shape = theta
    gamma_scale = gene_prop / theta
    rate_per_gene = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=(n_cells, n_genes))

    # 3) Build TF-target mappings
    gene_index = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}
    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    if not {"TF", "target", "weight"}.issubset(prior_dfs.columns):
        raise ValueError("prior_df must contain columns {'TF','target','weight'}")

    prior_tf = prior_dfs["TF"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_weight = prior_dfs["weight"].to_numpy()

    for tf, tgt, w in zip(prior_tf, prior_target, prior_weight):
        gi = gene_index.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if w == 1:
            tf_to_up_idx[ti].append(gi)
        elif w == -1:
            tf_to_down_idx[ti].append(gi)

    # 4) Apply regulatory effects directly to gamma output
    tf_factor = tf_effect_factor
    gt_states = ground_truth_dfs.to_numpy(copy=False)
    # Apply per TF in a vectorized manner over all cells
    for ti in range(n_tfs):
        ups = np.array(tf_to_up_idx[ti], dtype=np.int64)
        downs = np.array(tf_to_down_idx[ti], dtype=np.int64)

        if ups.size == 0 and downs.size == 0:
            continue

        states = gt_states[:, ti]
        act_cells = np.where(states == 1)[0]
        inact_cells = np.where(states == -1)[0]

        if act_cells.size:
            if ups.size:
                rate_per_gene[np.ix_(act_cells, ups)] *= tf_factor  # +ve targets, TF activated
            if downs.size:
                rate_per_gene[np.ix_(act_cells, downs)] /= tf_factor  # -ve targets, TF activated

        if inact_cells.size:
            if ups.size:
                rate_per_gene[np.ix_(inact_cells, ups)] /= tf_factor  # +ve targets, TF inactivated
            if downs.size:
                rate_per_gene[np.ix_(inact_cells, downs)] *= tf_factor  # -ve targets, TF inactivated

    # 5) For each cell, normalize sum to 1
    cell_totals = rate_per_gene.sum(axis=1, keepdims=True)
    cell_totals = np.where(cell_totals == 0, 1, cell_totals)  # Avoid division by zero
    rate_per_gene = rate_per_gene / cell_totals

    # 6) Normalize to library sizes
    target_libsize = 10.0e4  # 50,000
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.35, size=n_cells)
    mean_mat = rate_per_gene * libsize[:, None]  # Scale by library size

    # 7) Apply poisson to get final counts
    expr_counts = rng.poisson(mean_mat).astype(np.int64)

    # 8) Dropout (zero-inflation)
    n_zeros = np.sum(expr_counts == 0)
    current_missing_ratio = n_zeros / expr_counts.size
    print("Zero percentage before dropout:", 100.0 * current_missing_ratio)
    # Simple random dropout: for each value, randomly decide if it should be zero

    drop_mask = rng.random(size=expr_counts.shape) < missing_ratio
    expr_counts = expr_counts * (~drop_mask)

    n_zeros = np.sum(expr_counts == 0)
    current_missing_ratio = n_zeros / expr_counts.size
    print("Zero percentage after dropout:", 100.0 * current_missing_ratio)

    expr = pd.DataFrame(expr_counts, index=cell_ids, columns=gene_names)
    expr.clip(lower=0.0)

    # 9) Include TFs as expression features
    if include_tfs_in_expression:
        tf_names_arr = np.array(tf_names, dtype=object)
        # Sparse TF expression (lower baseline)
        tf_baseline = rng.lognormal(
            mean=1.0, sigma=0.75, size=(n_cells, len(tf_names_arr))
        )

        # Apply dropout to TFs
        tf_drop = rng.random(size=tf_baseline.shape) < 0.8
        tf_baseline = tf_baseline * (~tf_drop)

        tf_df = pd.DataFrame(tf_baseline.astype(np.int64), index=cell_ids, columns=tf_names_arr)
        expr = pd.concat([expr, tf_df], axis=1)

    return expr


def generate_gene_expression_neg_binomial_old(
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
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

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

    # Negative binomial via Gamma-Poisson
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
    gene_index = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}
    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    if not {"TF", "target", "weight"}.issubset(prior_dfs.columns):
        raise ValueError("prior_df must contain columns {'TF','target','weight'}")

    # Convert to numpy arrays once for faster iteration
    prior_tf = prior_dfs["TF"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_weight = prior_dfs["weight"].to_numpy()

    # Vectorized mapping using boolean indexing
    for tf, tgt, w in zip(prior_tf, prior_target, prior_weight):
        gi = gene_index.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if w == 1:
            tf_to_up_idx[ti].append(gi)
        elif w == -1:
            tf_to_down_idx[ti].append(gi)

    # Effect size as multiplicative factor (e.g., 1.25 means +25%)
    alpha = diff_effect[prediction_difficulty]
    up_act_factor = 1.0 + alpha
    down_act_factor = 1.0 - alpha  # keep non-negative by rounding later
    up_inact_factor = 1.0 - alpha
    down_inact_factor = 1.0 + alpha

    # Convert ground truth to numpy for vectorized operations
    gt_states = ground_truth_dfs.to_numpy(copy=False)  # shape (n_cells, n_tfs)

    # Apply regulatory effects using vectorized operations
    for ti in range(n_tfs):
        ups = np.array(tf_to_up_idx[ti], dtype=np.int64)
        downs = np.array(tf_to_down_idx[ti], dtype=np.int64)

        if ups.size == 0 and downs.size == 0:
            continue

        states = gt_states[:, ti]  # shape (n_cells,)
        act_cells = np.where(states == 1)[0]
        inact_cells = np.where(states == -1)[0]

        if act_cells.size:
            if ups.size:
                # +ve targets, TF activated: multiply by factor
                counts[np.ix_(act_cells, ups)] = np.floor(counts[np.ix_(act_cells, ups)] * up_act_factor)
            if downs.size:
                # -ve targets, TF activated: multiply by factor
                counts[np.ix_(act_cells, downs)] = np.floor(counts[np.ix_(act_cells, downs)] * down_act_factor)

        if inact_cells.size:
            if ups.size:
                # +ve targets, TF inactivated: multiply by factor
                counts[np.ix_(inact_cells, ups)] = np.floor(counts[np.ix_(inact_cells, ups)] * up_inact_factor)
            if downs.size:
                # -ve targets, TF inactivated: multiply by factor
                counts[np.ix_(inact_cells, downs)] = np.floor(counts[np.ix_(inact_cells, downs)] * down_inact_factor)

    # Ensure non-negative integer counts
    counts = np.clip(counts, a_min=0, a_max=None).astype(int)

    # ------------------------------------------
    # 4) Normalize and log-transform to realistic scale
    # ------------------------------------------
    # Counts-per-10k (CP10K) then log1p; this yields values mostly in [0, ~6]
    lib_after = counts.sum(axis=1)  # shape (n_cells,)
    lib_after[lib_after == 0] = 1  # avoid division by zero
    cp10k = (counts / lib_after[:, None]) * 1.0e4  # shape (n_cells, n_genes)
    log_expr = np.log1p(cp10k)

    expr = pd.DataFrame(log_expr, index=cell_ids, columns=gene_names)

    # Optionally include TFs as expression features: simulate similarly sparse, weak signal
    if include_tfs_in_expression:
        # Small independent noise for TF "expression"
        tf_counts = rng.poisson(0.1, size=(n_cells, n_tfs))
        # add mild dropout to keep mostly zeros
        tf_drop = rng.random(size=tf_counts.shape) < 0.8
        tf_counts = tf_counts * (~tf_drop)
        tf_cp10k = (tf_counts / lib_after[:, None]) * 1.0e4
        tf_log = np.log1p(tf_cp10k)

        tf_df = pd.DataFrame(tf_log, index=cell_ids, columns=tf_names)
        expr = pd.concat([expr, tf_df], axis=1)

    return expr


# Updated: direct TF effect factor
# Instead of old difficulty-based approach, use direct multiplicative factor
def generate_gene_expression_neg_binomial_old_updated(
        n_cells: int,
        n_genes: int,
        include_tfs_in_expression: bool,
        tf_effect_factor: float,
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

    if not (0 <= missing_percentage <= 100):
        raise ValueError("missing_percentage must be in [0, 100].")
    missing_ratio = float(missing_percentage) / 100.0

    rng = np.random.default_rng(random_seed)

    # Names
    cell_ids = [f"Cell_{i + 1}" for i in range(n_cells)]
    gene_names = np.array([f"G_{i + 1}" for i in range(n_genes)], dtype=object)
    tf_names = list(ground_truth_dfs.columns)
    n_tfs = len(tf_names)

    # ------------------------------------------
    # 1) Simulate raw counts with NB (Gamma-Poisson)
    # ------------------------------------------
    # Per-gene propensities (unnormalized), heavy-tailed like real data
    gene_prop = rng.lognormal(mean=0.0, sigma=1.0, size=n_genes)
    gene_prop = gene_prop / gene_prop.sum()  # convert to proportions that sum to 1

    # Per-cell library sizes (UMI totals), lognormal variability
    target_libsize = 1.0e4  # typical 10k UMIs per cell median
    libsize = rng.lognormal(mean=np.log(target_libsize), sigma=0.35, size=n_cells)

    # ------------------------------------------
    # 2) Apply TF regulatory effects BEFORE count generation
    # ------------------------------------------
    # Build maps TF -> up/down targets
    gene_index = {g: i for i, g in enumerate(gene_names)}
    tf_name_to_idx = {name: i for i, name in enumerate(tf_names)}
    tf_to_up_idx = [[] for _ in range(n_tfs)]
    tf_to_down_idx = [[] for _ in range(n_tfs)]

    if not {"TF", "target", "weight"}.issubset(prior_dfs.columns):
        raise ValueError("prior_df must contain columns {'TF','target','weight'}")

    prior_tf = prior_dfs["TF"].to_numpy()
    prior_target = prior_dfs["target"].to_numpy()
    prior_weight = prior_dfs["weight"].to_numpy()

    for tf, tgt, w in zip(prior_tf, prior_target, prior_weight):
        gi = gene_index.get(tgt, None)
        ti = tf_name_to_idx.get(tf, None)
        if gi is None or ti is None:
            continue
        if w == 1:
            tf_to_up_idx[ti].append(gi)
        elif w == -1:
            tf_to_down_idx[ti].append(gi)

    # Use tf_effect_factor directly as multiplicative factor
    tf_factor = tf_effect_factor

    # Convert ground truth to numpy for vectorized operations
    gt_states = ground_truth_dfs.to_numpy(copy=False)  # shape (n_cells, n_tfs)

    # Apply regulatory effects per cell (most realistic)
    mean_mat = np.zeros((n_cells, n_genes))
    for ci in range(n_cells):
        cell_gene_prop = gene_prop.copy()

        for ti in range(n_tfs):
            ups = np.array(tf_to_up_idx[ti], dtype=np.int64)
            downs = np.array(tf_to_down_idx[ti], dtype=np.int64)

            if ups.size == 0 and downs.size == 0:
                continue

            state = gt_states[ci, ti]

            if state == 1:  # Activation
                if ups.size:
                    cell_gene_prop[ups] *= tf_factor
                if downs.size:
                    cell_gene_prop[downs] /= tf_factor
            elif state == -1:  # Inactivation
                if ups.size:
                    cell_gene_prop[ups] /= tf_factor
                if downs.size:
                    cell_gene_prop[downs] *= tf_factor

        # Normalize per cell
        cell_gene_prop = cell_gene_prop / cell_gene_prop.sum()

        # Generate counts for this cell
        mean_mat[ci, :] = libsize[ci] * cell_gene_prop

    # ------------------------------------------
    # 3) Generate counts with NB (Gamma-Poisson)
    # ------------------------------------------
    theta = 5.0
    gamma_shape = theta
    gamma_scale = mean_mat / theta
    rate = rng.gamma(shape=gamma_shape, scale=gamma_scale)
    counts = rng.poisson(rate)

    # ------------------------------------------
    # 4) Apply dropout (zero-inflation)
    # ------------------------------------------
    a, b = (missing_ratio * 20.0, (1.0 - missing_ratio) * 20.0)
    per_gene_dropout = rng.beta(a, b, size=n_genes)
    drop_mask = rng.random(size=counts.shape) < per_gene_dropout[None, :]
    counts = counts * (~drop_mask)

    # Ensure non-negative integer counts
    counts = np.clip(counts, a_min=0, a_max=None).astype(int)

    # ------------------------------------------
    # 5) Normalize and log-transform to realistic scale
    # ------------------------------------------
    # Counts-per-10k (CP10K) then log1p; this yields values mostly in [0, ~6]
    lib_after = counts.sum(axis=1)  # shape (n_cells,)
    lib_after[lib_after == 0] = 1  # avoid division by zero
    cp10k = (counts / lib_after[:, None]) * 1.0e4  # shape (n_cells, n_genes)
    log_expr = np.log1p(cp10k)

    expr = pd.DataFrame(log_expr, index=cell_ids, columns=gene_names)

    # Optionally include TFs as expression features: simulate similarly sparse, weak signal
    if include_tfs_in_expression:
        # Small independent noise for TF "expression"
        tf_counts = rng.poisson(0.1, size=(n_cells, n_tfs))
        # add mild dropout to keep mostly zeros
        tf_drop = rng.random(size=tf_counts.shape) < 0.8
        tf_counts = tf_counts * (~tf_drop)
        tf_cp10k = (tf_counts / lib_after[:, None]) * 1.0e4
        tf_log = np.log1p(tf_cp10k)

        tf_df = pd.DataFrame(tf_log, index=cell_ids, columns=tf_names)
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

    # prior_df = generate_prior_poisson(
    #     n_genes=params["n_genes"],
    #     n_tfs=params["n_tfs"],
    #     lambda_param=15,
    #     random_seed=params["random_seed"],
    # )

    prior_df = generate_prior(
        n_genes=params["n_genes"],
        n_tfs=params["n_tfs"],
        min_num_of_targets_per_tf=params["min_num_of_targets_per_tf"],
        max_num_of_targets_per_tf=params["max_num_of_targets_per_tf"],
        random_seed=params["random_seed"]
    )

    # prior_df = generate_prior_lognormal(
    #     n_genes=params["n_genes"],
    #     n_tfs=params["n_tfs"],
    #     min_num_of_targets_per_tf=params["min_num_of_targets_per_tf"],
    #     max_num_of_targets_per_tf=params["max_num_of_targets_per_tf"],
    #     random_seed=params["random_seed"],
    # )

    ground_truth_df = generate_ground_truth(
        n_cells=params["n_cells"],
        n_tfs=params["n_tfs"],
        ground_truth_active_inactive_prob=params["ground_truth_active_inactive_prob"],
        random_seed=params["random_seed"],
    )

    # Choose an expression generator version
    distribution_type = params.get("distribution_type", "negative_binomial")
    if distribution_type == "normal":
        gene_exp = generate_gene_expression_gaussian(
            n_cells=params["n_cells"],
            n_genes=params["n_genes"],
            include_tfs_in_expression=False,
            tf_effect_factor=params["tf_effect_factor"],
            missing_percentage=params["missing_percentage"],
            random_seed=params["random_seed"],
            prior_dfs=prior_df,
            ground_truth_dfs=ground_truth_df,
        )

    elif distribution_type == "negative_binomial":
        # gene_exp = generate_gene_expression_neg_binomial(
        #     n_cells=params["n_cells"],
        #     n_genes=params["n_genes"],
        #     include_tfs_in_expression=False,
        #     tf_effect_factor=params["tf_effect_factor"],
        #     missing_percentage=params["missing_percentage"],
        #     random_seed=params["random_seed"],
        #     prior_dfs=prior_df,
        #     ground_truth_dfs=ground_truth_df
        # )

        # gene_exp = generate_gene_expression_neg_binomial_old(
        #     n_cells=params["n_cells"],
        #     n_genes=params["n_genes"],
        #     include_tfs_in_expression=False,
        #     prediction_difficulty="medium",
        #     missing_percentage=params["missing_percentage"],
        #     random_seed=params["random_seed"],
        #     prior_dfs=prior_df,
        #     ground_truth_dfs=ground_truth_df,
        # )

        gene_exp = generate_gene_expression_neg_binomial_old_updated(
            n_cells=params["n_cells"],
            n_genes=params["n_genes"],
            include_tfs_in_expression=False,
            tf_effect_factor=params["tf_effect_factor"],
            missing_percentage=params["missing_percentage"],
            random_seed=params["random_seed"],
            prior_dfs=prior_df,
            ground_truth_dfs=ground_truth_df,
        )

    else:
        raise ValueError("distribution_type must be one of {'normal','negative_binomial'}.")

    # move the target column to the end for easier viewing
    prior_out = prior_df[["TF", "weight", "target"]].copy()
    prior_out["weight"] = prior_out["weight"].map(
        {1: "upregulates-expression", -1: "downregulates-expression"}
    )
    prior_out.to_csv(
        f"{params['output_dir']}/{params['output_prior_file']}",
        sep="\t",
        index=False,
        header=False,
    )
    print(f"Wrote prior to {params['output_prior_file']}")

    ground_truth_df.to_csv(
        f"{params['output_dir']}/{params['output_ground_truth_file']}",
        sep="\t",
        index=True,
    )
    print(f"Wrote ground truth to {params['output_ground_truth_file']}")

    gene_exp.to_csv(
        f"{params['output_dir']}/{params['output_exp_file']}", sep="\t", index=True
    )
    print(f"Wrote gene expression to {params['output_exp_file']}")
