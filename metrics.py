"""
Provides evaluation routines, including:
1) in-context loss for the Transformer.
2) Prediction alignment metrics (L2 difference).
3) Sensitivity computation and comparison (cosine & L2).
4) A line_search_eta function for one-step GD baselines.
"""


import torch
import torch.nn.functional as F


def in_context_loss_transformer(transformer_model, Z, y_true):
    """
    Compute a scalar MSE loss for the transformer's test prediction vs. y_true.

    The transformer's final coordinate for the test token is stored as -prediction,
    so we negate that coordinate before comparing to y_true.

    Args:
        transformer_model: The trained Transformer model.
        Z: shape (B, n+1, d+1), containing training + test tokens.
        y_true: shape (B,), the ground-truth test labels.

    Returns:
        A scalar mean-squared-error loss.
    """
    output = transformer_model(Z)
    # The last token's last coordinate is -pred, so flip sign.
    y_pred = -output[:, -1, -1]
    return F.mse_loss(y_pred, y_true)


def in_context_loss(model, Z, y):
    """
    Compute mean-squared error for the test token output in the model's final layer.

    The test label is predicted as output[:, N, d], but user code might store the label as negative.
    Check usage carefully.

    Args:
        model: The Transformer.
        Z: (B, N+1, d+1).
        y: (B,) true test labels.

    Returns:
        A scalar MSE over the batch.
    """
    B, Np1, dp1 = Z.shape
    d = dp1 - 1
    N = Np1 - 1

    out = model(Z)
    pred_test = out[:, N, d]
    diff = pred_test + y
    return (diff ** 2).mean()


def l2_prediction_difference(y_pred_tf, y_pred_gd):
    """
    Compute the average L2 norm difference between two sets of scalar predictions.

    Args:
        y_pred_tf: shape (B,).
        y_pred_gd: shape (B,).

    Returns:
        A scalar representing the mean L2 norm difference.
    """
    diff = y_pred_tf - y_pred_gd
    return torch.norm(diff, p=2) / diff.shape[0]


def compute_sensitivity(model_predict_fn, Z):
    """
    Compute gradient of a model's test prediction wrt. the test token's features.

    Short summary:
      We clone Z with requires_grad, sum the model's predictions,
      and backprop to gather grad in Z_clone.grad at the test token.

    Args:
        model_predict_fn: Function mapping Z -> predictions of shape (B,).
        Z: shape (B, n+1, d+1). We'll clone and set requires_grad.

    Returns:
        shape (B, d). Gradients wrt. the test token's features.
    """
    Z_clone = Z.clone().detach().requires_grad_(True)
    y_pred_sum = model_predict_fn(Z_clone).sum()
    y_pred_sum.backward()
    return Z_clone.grad[:, -1, :-1]


def cosine_sensitivity(sens_tf, sens_gd):
    """
    Compute mean cosine similarity between two (B, d) sets of sensitivity vectors.

    Returns a scalar average similarity in [−1, 1].
    """
    return F.cosine_similarity(sens_tf, sens_gd, dim=1).mean()


def l2_sensitivity_difference(sens_tf, sens_gd):
    """
    Compute average L2 norm difference between two (B, d) sensitivity sets.

    Returns a single scalar: the mean L2 difference.
    """
    diff = sens_tf - sens_gd
    return torch.norm(diff, p=2) / diff.shape[0]


def line_search_eta(eta_values, gd_model_class, Z_val, y_val, Sigma_inv=None):
    """
    Grid search over candidate eta to minimize MSE for a one-step GD baseline.

    Short:
      For each eta in eta_values, instantiate the GD model, predict the test labels,
      compute MSE, and pick the best eta.

    Args:
        eta_values: A sequence of eta candidates (float).
        gd_model_class: Class implementing `predict_single(...)`.
        Z_val: shape (B_val, n+1, d+1), validation prompt.
        y_val: shape (B_val,), validation labels.
        Sigma_inv: Optional preconditioning matrix.

    Returns:
        best_eta: The float in eta_values that yields minimal MSE.
        losses: A list of MSE values for each candidate.
    """
    losses = []
    for eta in eta_values:
        gd_model = gd_model_class(eta=eta, Sigma_inv=Sigma_inv)
        y_pred = gd_model.predict_single(Z_val)
        losses.append(F.mse_loss(y_pred, y_val).item())

    losses_t = torch.tensor(losses, dtype=torch.float32)
    best_eta_idx = torch.argmin(losses_t)
    best_eta = eta_values[best_eta_idx]
    return best_eta, losses