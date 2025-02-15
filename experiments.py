"""
High-level orchestration of multi-run experiments:
1) Building folder names & condition number info.
2) Running multiple seeds/configs, aggregating logs.
3) Plotting aggregate metrics across seeds.
"""


import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import random

from transformer import Transformer
from training import train_transformer_model
from data import random_orthogonal, generate_linear_regression_data


def build_experiment_folder_name(
    B_train,
    val_batch_size,
    N_val,
    d_val,
    num_steps,
    metric_interval,
    compare_mode,
    activation_choice,
    leaky_alpha,
    eigenvalues_list,
    diagonal_covariance,
    noise_variance,
    line_search_min,
    line_search_max,
    line_search_steps
):
    """
    Build a log folder name reflecting hyperparameters, including line search params.

    Args:
        B_train: Training batch size.
        val_batch_size: Validation batch size.
        N_val: Validation number of tokens.
        d_val: Feature dimension for validation.
        num_steps: Training steps.
        metric_interval: Steps between metric logging.
        compare_mode: "manual"/"constructed".
        activation_choice: e.g., None, "relu", "leakyrelu".
        leaky_alpha: Negative slope for leakyrelu.
        eigenvalues_list: Eigenvalues for constructing Sigma, used for cond number.
        diagonal_covariance: True for Sigma diagonal matrix.
        noise_variance: Variance of additive noise.
        line_search_min: Minimum η tested.
        line_search_max: Maximum η tested.
        line_search_steps: How many discrete η values.

    Returns:
        A string folder_name with hyperparams + condition number + line-search range.
    """
    eigenvals = np.array(eigenvalues_list)
    cond_num = eigenvals.max() / eigenvals.min()

    if activation_choice is None:
        act_str = "linear"
    else:
        lower_act = activation_choice.lower()
        if lower_act == "leakyrelu":
            act_str = f"leakyrelu{leaky_alpha}"
        else:
            act_str = lower_act

    folder_name = (
        f"exp_N{N_val}_d{d_val}_act_{act_str}_noise{noise_variance:.2f}_"
        f"cond{cond_num:.2f}_diagCov_{diagonal_covariance}_steps{num_steps}_"
        f"valB{val_batch_size}_B{B_train}_int{metric_interval}_"
        f"ls({line_search_min}-{line_search_max}-{line_search_steps})_cmp_{compare_mode}"
    )

    return folder_name


#############################
# Aggregation of Seeds
#############################

def compute_mean_std_logs(all_logs, metric_key):
    """
    Compute mean±std arrays for a given metric_key across all_logs.

    Args:
        all_logs: List of dict logs from different seeds.
        metric_key: e.g. "in_context_loss_mean".

    Returns:
        mean_vals, std_vals: Numpy arrays of shape (num_steps_in_log,).
    """
    arr = np.array([log[metric_key] for log in all_logs])
    mean_vals = arr.mean(axis=0)
    std_vals = arr.std(axis=0)
    return mean_vals, std_vals


def plot_incontext_vs_pgd(steps, inc_mean, inc_std, pgd_mean, pgd_std, out_path):
    """
    Plot aggregated in-context vs. PGD losses (with shading).

    Args:
        steps: 1D array of step indices.
        inc_mean: 1D array, transformer in-context mean.
        inc_std: 1D array, transformer in-context std.
        pgd_mean: 1D array, PGD mean.
        pgd_std: 1D array, PGD std.
        out_path: PDF filename to save.
    """
    plt.figure()
    plt.plot(steps, inc_mean, label="Transformer In-context", color="blue")
    plt.fill_between(
        steps, inc_mean - inc_std, inc_mean + inc_std,
        alpha=0.3, color="blue", label="Transformer ±1 STD"
    )
    plt.plot(steps, pgd_mean, label="PGD Loss", color="red", linestyle="--")
    plt.fill_between(
        steps, pgd_mean - pgd_std, pgd_mean + pgd_std,
        alpha=0.3, color="red", label="PGD ±1 STD"
    )
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Transformer vs. PGD Loss (Aggregated)")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def plot_extra_metric(steps, mean_vals, std_vals, metric, out_folder):
    """
    Plot an extra metric (e.g. l2_pred_diff_mean) with shading.

    Args:
        steps: 1D array of step indices.
        mean_vals: 1D array, metric means.
        std_vals: 1D array, metric stds.
        metric: Name of the metric for labeling.
        out_folder: Directory to save PDF.
    """
    plt.figure()
    plt.plot(steps, mean_vals, label=f"{metric} (mean)")
    plt.fill_between(
        steps, mean_vals - std_vals, mean_vals + std_vals,
        alpha=0.3, label="±1 STD"
    )
    plt.xlabel("Training Steps")
    plt.ylabel(metric)
    plt.title(f"{metric} vs. Training Steps (Aggregated)")
    plt.legend()
    plt.savefig(os.path.join(out_folder, f"{metric}_aggregated.pdf"))
    plt.close()


def find_best_incontext_index(aggregated_data):
    """
    Locate the index of minimal in-context loss from the aggregated dictionary.

    Args:
        aggregated_data: dictionary that must have 'in_context_loss_mean': { 'mean': array, 'std': array }.

    Returns:
        best_idx: integer index of minimal in-context mean.
    """
    inc_means = aggregated_data['in_context_loss_mean']['mean']
    return np.argmin(inc_means)


def save_aggregated_best_row(steps, best_idx, aggregated_data, out_folder):
    """
    Write CSV of best row in in-context loss to "aggregated_best_metrics.csv".

    Args:
        steps: 1D array of step indices.
        best_idx: int index for minimal in-context mean.
        aggregated_data: dictionary containing all aggregated means/stds.
        out_folder: directory to save the CSV.
    """
    best_row = {
        "step": steps[best_idx],
        "in_context_loss": aggregated_data['in_context_loss_mean']['mean'][best_idx],
        "pgd_loss": aggregated_data['pgd_loss_mean']['mean'][best_idx],
        "l2_pred_diff": aggregated_data['l2_pred_diff_mean']['mean'][best_idx],
        "cosine_sens": aggregated_data['cosine_sens_mean']['mean'][best_idx],
        "l2_sens_diff": aggregated_data['l2_sens_diff_mean']['mean'][best_idx],
    }

    csv_filename = os.path.join(out_folder, "aggregated_best_metrics.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "in_context_loss", "pgd_loss", "l2_pred_diff", "cosine_sens", "l2_sens_diff"])
        writer.writerow([
            best_row["step"], best_row["in_context_loss"], best_row["pgd_loss"],
            best_row["l2_pred_diff"], best_row["cosine_sens"], best_row["l2_sens_diff"]
        ])


def save_aggregated_tensors(aggregated_data, out_folder, all_logs):
    """
    Save aggregated_data to a .pt file, and if 'Lambda' is in the logs, save it as Lambda.csv.

    Args:
        aggregated_data: dict of mean/std arrays.
        out_folder: folder to save.
        all_logs: list of logs, possibly containing 'Lambda'.
    """
    torch.save(aggregated_data, os.path.join(out_folder, "aggregated_metrics.pt"))

    # If 'Lambda' is in the first log
    if 'Lambda' in all_logs[0]:
        lamb = np.array(all_logs[0]['Lambda'])
        np.savetxt(os.path.join(out_folder, "Lambda.csv"), lamb, delimiter=",")


def aggregate_and_plot(all_logs, exp_folder="results", base_folder_name=None):
    """
    Aggregate logs from multiple experiments, plot and save final data.

    Short:
      We compute mean±std for in-context, PGD, l2_pred_diff, cos_sens, l2_sens_diff,
      then plot them, find best in-context index, and save CSV with that row.

    Args:
        all_logs: list of logs (one per seed).
        exp_folder: root folder for saving.
        base_folder_name: name of subfolder to store aggregated results.

    Returns:
        aggregated_data: dict with aggregated means/stds.
    """
    if base_folder_name is None:
        base_folder_name = "exp_B32_valB1000_N20_d5_steps1000_int50_cmp_manual"
    aggregated_folder = os.path.join(exp_folder, base_folder_name, "aggregated")
    os.makedirs(aggregated_folder, exist_ok=True)

    steps = np.array(all_logs[0]['steps'])
    aggregated_data = {}

    # 1) In-context vs. PGD
    inc_mean, inc_std = compute_mean_std_logs(all_logs, 'in_context_loss_mean')
    pgd_mean, pgd_std = compute_mean_std_logs(all_logs, 'pgd_loss_mean')
    aggregated_data['in_context_loss_mean'] = {'mean': inc_mean, 'std': inc_std}
    aggregated_data['pgd_loss_mean'] = {'mean': pgd_mean, 'std': pgd_std}

    plot_incontext_vs_pgd(
        steps, inc_mean, inc_std, pgd_mean, pgd_std,
        out_path=os.path.join(aggregated_folder, "loss_comparison_aggregated.pdf")
    )

    # 2) Extra metrics
    for metric_key in ['l2_pred_diff_mean', 'cosine_sens_mean', 'l2_sens_diff_mean']:
        mean_vals, std_vals = compute_mean_std_logs(all_logs, metric_key)
        aggregated_data[metric_key] = {'mean': mean_vals, 'std': std_vals}
        plot_extra_metric(steps, mean_vals, std_vals, metric_key, aggregated_folder)

    # 3) Best row for in-context
    best_idx = find_best_incontext_index(aggregated_data)
    save_aggregated_best_row(steps, best_idx, aggregated_data, aggregated_folder)

    # 4) Save final aggregated
    save_aggregated_tensors(aggregated_data, aggregated_folder, all_logs)
    return aggregated_data


#############################
# Main (Singular) Experiment
#############################

def construct_sigma_and_inverse(d, eigenvalues_list, device, diagonal_covariance):
    """
    Build Sigma = U * diag(eigenvalues) * U^T and Sigma_inv from random orthogonal U.

    Args:
        d: Feature dimension.
        eigenvalues_list: e.g. [2.0, 1.0, 0.5,...].
        device: Torch device.
        diagonal_covariance: True for Sigma diagonal matrix

    Returns:
        Sigma, Sigma_inv (both d x d),
        eig_tensor (the diag of eigenvalues).
    """
    eig_tensor = torch.tensor(eigenvalues_list, dtype=torch.float32, device=device)
    Lambda = torch.diag(eig_tensor)
    if diagonal_covariance:
        U = torch.eye(d, device=device)
    else:
        U = random_orthogonal(d, device=device)
    Sigma = U @ Lambda @ U.t()
    Lambda_inv = torch.diag(1.0 / eig_tensor)
    Sigma_inv = U @ Lambda_inv @ U.t()
    return Sigma, Sigma_inv, eig_tensor


def setup_data_generator(B, N, d, Sigma, noise_variance, device):
    """
    Create a closure that returns (Z, y_test) using generate_linear_regression_data
    with the given Sigma.

    Args:
        B: Training batch size.
        N: Number of tokens.
        d: Feature dimension.
        Sigma: Covariance matrix.
        noise_variance: Variance of additive noise.
        device: Torch device.

    Returns:
        data_gen function with no args that returns Z, y_test each call.
    """
    def data_gen():
        Z, y_test = generate_linear_regression_data(
            batch_size=B,
            num_tokens=N,
            feature_dim=d,
            sigma=Sigma,
            noise_variance=noise_variance,
            device=device
        )
        return Z, y_test
    return data_gen


def train_and_log(
    seed,
    B,
    N,
    d,
    num_steps,
    metric_interval,
    val_batch_size,
    device,
    compare_mode,
    activation_choice,
    leaky_alpha,
    Sigma,
    Sigma_inv,
    eig_tensor,
    diagonal_covariance,
    noise_variance,
    line_search_min,
    line_search_max,
    line_search_steps
):
    """
    Train a single-layer Transformer with given params, store Sigma in logs.

    Args:
        seed, B, N, d, etc.: typical hyperparameters.
        Sigma, Sigma_inv: Cov matrix + its inverse for preconditioning.
        eig_tensor: The diag of eigenvalues.
        diagonal_covariance: True for Sigma diagonal matrix.
        noise_variance: Variance of additive noise.
        line_search_min, line_search_max, line_search_steps: range and resolution for eta search.

    Returns:
        metric_log (dict): All training/val metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transformer_model = Transformer(
        n_layer=1,
        n_head=1,
        d=d,
        var=0.0001,
        activation=activation_choice,
        leaky_alpha=leaky_alpha
    ).to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)

    data_gen = setup_data_generator(B, N, d, Sigma, noise_variance, device)

    base_folder = build_experiment_folder_name(
        B, val_batch_size, N, d, num_steps, metric_interval,
        compare_mode, activation_choice, leaky_alpha, eig_tensor.cpu().numpy(), diagonal_covariance,
        noise_variance, line_search_min, line_search_max, line_search_steps
    )

    metric_log = train_transformer_model(
        model=transformer_model,
        optimizer=optimizer,
        num_steps=num_steps,
        data_generator=data_gen,
        metric_interval=metric_interval,
        device=device,
        validation_batch_size=val_batch_size,
        validation_num_tokens=N,
        validation_feature_dim=d,
        training_batch_size=B,
        experiment_folder="results",
        seed=seed,
        compare_mode=compare_mode,
        preconditioning_matrix=Sigma_inv,
        base_folder=base_folder,
        covariance_matrix=Sigma,
        noise_variance=noise_variance,
        line_search_min=line_search_min,
        line_search_max=line_search_max,
        line_search_steps=line_search_steps
    )

    metric_log['Sigma'] = Sigma.cpu().numpy()
    metric_log['Lambda'] = eig_tensor.cpu().numpy()
    return metric_log


def main_experiment(
    seed,
    B=32,
    N=20,
    d=5,
    num_steps=1000,
    metric_interval=50,
    val_batch_size=1000,
    device=None,
    compare_mode="manual",
    activation_choice=None,
    leaky_alpha=0.5,
    eigenvalues_list=None,
    diagonal_covariance=False,
    noise_variance=0.0,
    line_search_min=0.001,
    line_search_max=2.0,
    line_search_steps=1000
):
    """
    A single-run experiment combining data generation, a single-layer Transformer,
    and the chosen baseline compare mode, now with line_search range as a hyperparam.

    Args:
        seed: Random seed.
        B, N, d: Train batch size, tokens, dimension.
        num_steps, metric_interval: Steps + logging interval.
        val_batch_size: Validation batch size.
        device: Torch device.
        compare_mode: "manual"/"constructed".
        activation_choice: e.g. None,"relu","leakyrelu","softmax".
        leaky_alpha: slope if leakyrelu.
        eigenvalues_list: list of eigenvals for constructing Sigma.
        diagonal_covariance: True for Sigma diagonal matrix.
        noise_variance: Variance of additive noise to labels.
        line_search_min: minimal eta.
        line_search_max: maximal eta.
        line_search_steps: number of eta candidates.

    Returns:
        metric_log with training/validation info for this seed.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eigenvalues_list is None:
        eigenvalues_list = [1.0] * d

    Sigma, Sigma_inv, eig_tensor = construct_sigma_and_inverse(d, eigenvalues_list, device, diagonal_covariance)

    return train_and_log(
        seed=seed,
        B=B,
        N=N,
        d=d,
        num_steps=num_steps,
        metric_interval=metric_interval,
        val_batch_size=val_batch_size,
        device=device,
        compare_mode=compare_mode,
        activation_choice=activation_choice,
        leaky_alpha=leaky_alpha,
        Sigma=Sigma,
        Sigma_inv=Sigma_inv,
        eig_tensor=eig_tensor,
        diagonal_covariance=diagonal_covariance,
        noise_variance=noise_variance,
        line_search_min=line_search_min,
        line_search_max=line_search_max,
        line_search_steps=line_search_steps
    )


#############################
# Run Several Experiments
#############################

def run_experiments(
    num_experiments=5,
    B=32,
    N=20,
    d=5,
    num_steps=1000,
    metric_interval=50,
    val_batch_size=1000,
    device=None,
    compare_mode="manual",
    activation_choice=None,
    leaky_alpha=0.5,
    eigenvalues_list=None,
    diagonal_covariance=False,
    exp_folder="results",
    noise_variance=0.0,
    line_search_min=0.001,
    line_search_max=2.0,
    line_search_steps=1000
):
    """
    Run main_experiment over multiple seeds, then aggregate results, with line-search params as hyperparams.

    Args:
        num_experiments: number of seeds.
        B, N, d: train batch size, #tokens, dimension.
        num_steps, metric_interval: steps + logging interval.
        val_batch_size: validation batch size.
        device: Torch device or None for auto.
        compare_mode: "manual" or "constructed".
        activation_choice: None, "relu", "leakyrelu", "softmax".
        leaky_alpha: negative slope for leakyrelu.
        eigenvalues_list: eigenvals for Sigma.
        diagonal_covariance: True for diagonal Sigma matrix.
        exp_folder: root logs folder.
        noise_variance: variance of additive noise to labels.
        line_search_min: minimal eta.
        line_search_max: maximum eta.
        line_search_steps: number of discrete eta candidates.

    Returns:
        (all_logs, aggregated_data)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if eigenvalues_list is None:
        eigenvalues_list = [1.0] * d

    cond = np.array(eigenvalues_list).max() / np.array(eigenvalues_list).min()
    if activation_choice is None:
        act_str = "linear"
    else:
        if activation_choice.lower() == "leakyrelu":
            act_str = f"leakyrelu{leaky_alpha}"
        else:
            act_str = activation_choice.lower()

    # Build base folder with line search info
    base_folder = build_experiment_folder_name(
        B, val_batch_size, N, d, num_steps, metric_interval,
        compare_mode, activation_choice, leaky_alpha, eigenvalues_list, diagonal_covariance,
        noise_variance, line_search_min, line_search_max, line_search_steps
    )

    print("====== Common Experiment Parameters ======")
    print(f"Training batch size (B): {B}")
    print(f"Number of training tokens (N): {N}")
    print(f"Feature dimension (d): {d}")
    print(f"Number of training steps: {num_steps}")
    print(f"Metric interval: {metric_interval}")
    print(f"Validation batch size: {val_batch_size}")
    print(f"Compare mode: {compare_mode}")
    print(f"Activation: {act_str}")
    print(f"Eigenvalues (Lambda): {eigenvalues_list}")
    print(f"Condition number: {cond:.2f}")
    print(f"Diagonal covariance: {diagonal_covariance}")
    print(f"Noise variance: {noise_variance:.2f}")
    print(f"Line search range: [{line_search_min}, {line_search_max}], {line_search_steps} steps.")
    print(f"Results folder: {exp_folder}/{base_folder}")
    print("============================================")

    all_logs = []
    for seed in range(num_experiments):
        print(f"\n--- Running experiment with seed {seed} ---")
        log = main_experiment(
            seed=seed,
            B=B,
            N=N,
            d=d,
            num_steps=num_steps,
            metric_interval=metric_interval,
            val_batch_size=val_batch_size,
            device=device,
            compare_mode=compare_mode,
            activation_choice=activation_choice,
            leaky_alpha=leaky_alpha,
            eigenvalues_list=eigenvalues_list,
            diagonal_covariance=diagonal_covariance,
            noise_variance=noise_variance,
            line_search_min=line_search_min,
            line_search_max=line_search_max,
            line_search_steps=line_search_steps
        )
        all_logs.append(log)

    aggregated_data = aggregate_and_plot(all_logs, exp_folder=exp_folder, base_folder_name=base_folder)
    return all_logs, aggregated_data