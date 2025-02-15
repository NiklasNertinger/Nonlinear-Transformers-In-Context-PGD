"""
Implements:
1) A training loop for the Transformer on linear-regression prompts.
2) Validation logic comparing to baselines via line search.
3) Utility functions to save logs, pick best metrics, and plot results.
"""


import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from metrics import (
    in_context_loss_transformer,
    l2_prediction_difference,
    compute_sensitivity,
    cosine_sensitivity,
    l2_sensitivity_difference,
    line_search_eta
)
from baselines import GradientDescentOneStep, constructed_transformer_predict
from transformer import transformer_predict
from data import generate_linear_regression_data


def validate_transformer_on_regression(
    model,
    Z_val,
    y_val,
    compare_mode,
    preconditioning_matrix=None,
    device=None,
    line_search_min=0.001,
    line_search_max=2.0,
    line_search_steps=1000
):
    """
    Validate a trained Transformer on linear-regression prompts vs. a baseline.

    A line search finds the best one-step GD stepsize (eta), then metrics like
    in-context loss, baseline loss, L2 pred diff, and sensitivity are computed.

    Args:
        model (nn.Module): The trained Transformer model.
        Z_val (torch.Tensor): shape (B, n+1, d+1) validation prompts.
        y_val (torch.Tensor): shape (B,) true test labels.
        compare_mode (str): "manual" or "constructed" baseline approach.
        preconditioning_matrix (torch.Tensor or None): Sigma^-1 or None for identity.
        device (torch.device or None): Torch device.
        line_search_min (float): minimal candidate for eta.
        line_search_max (float): maximal candidate for eta.
        line_search_steps (int): how many discrete values in the line search.

    Returns:
        dict with:
            "best_eta", "in_context_loss", "baseline_loss",
            "l2_pred_diff", "cosine_sens", "l2_sens_diff".
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build linearly spaced eta candidates on same device
    eta_candidates = torch.linspace(
        line_search_min,
        line_search_max,
        steps=line_search_steps,
        device=device
    ).cpu().tolist()

    best_eta, _ = line_search_eta(
        eta_candidates,
        GradientDescentOneStep,
        Z_val,
        y_val,
        Sigma_inv=preconditioning_matrix
    )
    # Transformer in-context loss
    in_context_val_loss = in_context_loss_transformer(model, Z_val, y_val).item()

    # Baseline predictions
    if compare_mode == "manual":
        gd_model = GradientDescentOneStep(eta=best_eta, Sigma_inv=preconditioning_matrix)
        y_baseline = gd_model.predict_single(Z_val)
    elif compare_mode == "constructed":
        y_baseline = constructed_transformer_predict(
            Z_val, best_eta, device, precond=preconditioning_matrix
        )
    else:
        raise ValueError(f"Unknown compare_mode: {compare_mode}")

    baseline_loss_val = F.mse_loss(y_baseline, y_val).item()

    # L2 prediction difference
    y_tf = transformer_predict(model, Z_val)
    l2_pred_diff_val = l2_prediction_difference(y_tf, y_baseline).item()

    # Sensitivity metrics
    def tf_predict_fn(z_in):
        return transformer_predict(model, z_in)

    if compare_mode == "manual":
        def baseline_predict_fn(z_in):
            return gd_model.predict_single(z_in)
    else:
        def baseline_predict_fn(z_in):
            return constructed_transformer_predict(z_in, best_eta, device, precond=preconditioning_matrix)

    sens_tf = compute_sensitivity(tf_predict_fn, Z_val)
    sens_bl = compute_sensitivity(baseline_predict_fn, Z_val)
    cos_val = cosine_sensitivity(sens_tf, sens_bl).item()
    l2_sens_val = l2_sensitivity_difference(sens_tf, sens_bl).item()

    return {
        "best_eta": best_eta,
        "in_context_loss": in_context_val_loss,
        "baseline_loss": baseline_loss_val,
        "l2_pred_diff": l2_pred_diff_val,
        "cosine_sens": cos_val,
        "l2_sens_diff": l2_sens_val
    }


def save_best_metrics_csv(metric_log, folder):
    """
    Save a CSV of the best (lowest in-context loss) metrics from metric_log.

    Args:
        metric_log (dict): containing 'steps', 'in_context_loss_mean', etc.
        folder (str): Path to write 'best_metrics.csv'.
    """
    losses = np.array(metric_log['in_context_loss_mean'])
    best_idx = int(np.argmin(losses))

    header = [
        "step", "in_context_loss", "baseline_loss", "l2_pred_diff",
        "cosine_sens", "l2_sens_diff", "best_eta"
    ]
    row = [
        metric_log['steps'][best_idx],
        metric_log['in_context_loss_mean'][best_idx],
        metric_log['pgd_loss_mean'][best_idx],
        metric_log['l2_pred_diff_mean'][best_idx],
        metric_log['cosine_sens_mean'][best_idx],
        metric_log['l2_sens_diff_mean'][best_idx],
        metric_log['best_eta'][best_idx]
    ]
    csv_path = os.path.join(folder, "best_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)


def plot_training_curves(metric_log, folder):
    """
    Create and save training curves (losses, differences, sensitivities) to PDF files.

    Args:
        metric_log: Dictionary of metric lists.
        folder: Directory to save the plots.
    """
    steps = np.array(metric_log['steps'])

    # A) Transformer vs. Baseline loss
    plt.figure()
    plt.plot(steps, metric_log['in_context_loss_mean'], label="Transformer In-context", color="blue")
    plt.plot(steps, metric_log['pgd_loss_mean'], label="Baseline Loss", color="red", linestyle="--")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Transformer vs. Baseline Loss")
    plt.legend()
    plt.savefig(os.path.join(folder, "loss_comparison.pdf"))
    plt.close()

    # B) L2 prediction difference
    plt.figure()
    plt.plot(steps, metric_log['l2_pred_diff_mean'], label="L2 Prediction Diff", color="green")
    plt.xlabel("Training Steps")
    plt.ylabel("L2 Prediction Diff")
    plt.title("Prediction Difference")
    plt.legend()
    plt.savefig(os.path.join(folder, "l2_prediction_diff.pdf"))
    plt.close()

    # C) Cosine sensitivity
    plt.figure()
    plt.plot(steps, metric_log['cosine_sens_mean'], label="Cosine Sensitivity", color="purple")
    plt.xlabel("Training Steps")
    plt.ylabel("Cosine Similarity")
    plt.title("Cosine Sensitivity")
    plt.legend()
    plt.savefig(os.path.join(folder, "cosine_sensitivity.pdf"))
    plt.close()

    # D) L2 sensitivity difference
    plt.figure()
    plt.plot(steps, metric_log['l2_sens_diff_mean'], label="L2 Sensitivity Diff", color="orange")
    plt.xlabel("Training Steps")
    plt.ylabel("L2 Sensitivity Difference")
    plt.title("Sensitivity Difference")
    plt.legend()
    plt.savefig(os.path.join(folder, "l2_sensitivity_diff.pdf"))
    plt.close()


def train_transformer_model(
    model,
    optimizer,
    num_steps,
    data_generator,
    metric_interval,
    device,
    validation_batch_size=1000,
    validation_num_tokens=20,
    validation_feature_dim=5,
    training_batch_size=32,
    experiment_folder="results",
    seed=0,
    compare_mode="manual",
    preconditioning_matrix=None,
    base_folder=None,
    covariance_matrix=None,
    noise_variance=0.0,
    line_search_min=0.001,
    line_search_max=2.0,
    line_search_steps=1000
):
    """
    Train a Transformer on linear-regression prompts, record metrics, and save results.

    In each iteration, data_generator provides training data. Every metric_interval steps,
    validation data is created using the same covariance_matrix, and we evaluate:

    1) One-step GD baseline via line search for best eta.
    2) Transformer's in-context loss, baseline loss, L2 pred difference, sensitivity.

    Collected metrics are stored and plotted, plus we save a CSV of best metrics.

    Args:
        model (nn.Module): The Transformer to train.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        num_steps (int): Number of training steps.
        data_generator (callable): Returns (Z, y_true) for each iteration.
        metric_interval (int): Steps between validation/evaluation.
        device (torch.device): For computations.
        validation_batch_size (int): Batch size for validation.
        validation_num_tokens (int): # training tokens for validation prompts.
        validation_feature_dim (int): Feature dim for validation prompts.
        training_batch_size (int): For naming logs only.
        experiment_folder (str): Root dir for logs and plots.
        seed (int): Random seed used in folder naming.
        compare_mode (str): "manual" or "constructed" baseline approach.
        preconditioning_matrix (torch.Tensor or None): Sigma^-1 or None for identity.
        base_folder (str or None): If None, automatically generated for logs.
        covariance_matrix (torch.Tensor or None): If provided, we store it in logs
                                                  and use it for validation data gen.
        noise_variance (float): Variance of noise added to training labels.
        line_search_min (float): Minimal candidate for eta line search.
        line_search_max (float): Maximal candidate for eta line search.
        line_search_steps (int): Number of discrete values in the line search.

    Returns:
        dict: metric_log of training/validation metrics over time.
    """
    model.train()

    # Initialize logs
    metric_log = {
        'steps': [],
        'in_context_loss_mean': [],
        'l2_pred_diff_mean': [],
        'cosine_sens_mean': [],
        'l2_sens_diff_mean': [],
        'pgd_loss_mean': [],
        'pgd_loss_std': [],
        'best_eta': []
    }

    if base_folder is None:
        base_folder = (
            f"exp_B{training_batch_size}_valB{validation_batch_size}_N{validation_num_tokens}"
            f"_d{validation_feature_dim}_steps{num_steps}_int{metric_interval}"
            f"_cmp_{compare_mode}_act_{model.act_str}"
        )
    seed_folder = os.path.join(experiment_folder, base_folder, f"seed_{seed}")
    os.makedirs(seed_folder, exist_ok=True)

    # If we have a known Sigma, store it
    if covariance_matrix is not None:
        metric_log['Sigma'] = covariance_matrix.cpu().numpy()

    progress_bar = tqdm(range(num_steps), desc="Train_Transformer", unit="step")
    for step in progress_bar:
        Z_train, y_train = data_generator()
        Z_train, y_train = Z_train.to(device), y_train.to(device)

        optimizer.zero_grad()
        loss_train = in_context_loss_transformer(model, Z_train, y_train)
        loss_train.backward()
        optimizer.step()

        if step % metric_interval == 0:
            # Construct validation data
            Z_val, y_val = generate_linear_regression_data(
                batch_size=validation_batch_size,
                num_tokens=validation_num_tokens,
                feature_dim=validation_feature_dim,
                sigma=covariance_matrix,
                noise_variance=noise_variance,
                device=device
            )
            # Evaluate
            val_dict = validate_transformer_on_regression(
                model=model,
                Z_val=Z_val,
                y_val=y_val,
                compare_mode=compare_mode,
                preconditioning_matrix=preconditioning_matrix,
                device=device,
                line_search_min=line_search_min,
                line_search_max=line_search_max,
                line_search_steps=line_search_steps
            )
            # Record
            metric_log['steps'].append(step)
            metric_log['best_eta'].append(val_dict["best_eta"])
            metric_log['in_context_loss_mean'].append(val_dict["in_context_loss"])
            metric_log['pgd_loss_mean'].append(val_dict["baseline_loss"])
            metric_log['pgd_loss_std'].append(0.0)  # Not computing std here
            metric_log['l2_pred_diff_mean'].append(val_dict["l2_pred_diff"])
            metric_log['cosine_sens_mean'].append(val_dict["cosine_sens"])
            metric_log['l2_sens_diff_mean'].append(val_dict["l2_sens_diff"])

            progress_bar.set_postfix({
                "TrainLoss": f"{loss_train.item():.4f}",
                "ValLoss": f"{val_dict['in_context_loss']:.4f}",
                "BaselineLoss": f"{val_dict['baseline_loss']:.4f}",
                "L2PredDiff": f"{val_dict['l2_pred_diff']:.4f}",
                "CosSens": f"{val_dict['cosine_sens']:.4f}",
                "L2SensDiff": f"{val_dict['l2_sens_diff']:.4f}",
                "BestEta": f"{val_dict['best_eta']:.4f}"
            })

    # Save logs
    torch.save(metric_log, os.path.join(seed_folder, "metric_log.pt"))
    if 'Sigma' in metric_log:
        np.savetxt(os.path.join(seed_folder, "Sigma.csv"), metric_log['Sigma'], delimiter=",")

    # Save best metrics
    save_best_metrics_csv(metric_log, seed_folder)

    # Final plots
    plot_training_curves(metric_log, seed_folder)

    return metric_log