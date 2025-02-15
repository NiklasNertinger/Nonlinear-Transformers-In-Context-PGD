"""
Implements baseline methods for in-context learning comparisons:
1) One-step (preconditioned) GradientDescentOneStep class.
2) A constructed_transformer_predict function that encodes a step of GD in transformer-like attention parameters with constructed weights.
"""


import torch

from transformer import attention


class GradientDescentOneStep:
    """
    Single-step gradient descent for linear regression.

    Short:
      W_1 = (eta / N) sum_i y_i x_i, optional preconditioning by Sigma_inv.

    Longer:
      The class can predict all tokens or just the final token (test).
      Preconditioning is applied if Sigma_inv is provided, else identity.
    """
    def __init__(self, eta, Sigma_inv=None):
        self.eta = eta
        self.Sigma_inv = Sigma_inv

    def _compute_W1(self, X, Y):
        """
        Internal routine computing the weight vector for a single step.

        Args:
            X: (B, N, d).
            Y: (B, N).

        Returns:
            (B, d) weight vector after one-step GD.
        """
        raw = torch.einsum('bni->bi', Y.unsqueeze(-1) * X)  # sum of y_i x_i
        W1 = self.eta / X.shape[1] * raw

        d = X.shape[2]
        if self.Sigma_inv is None:
            Sigma_inv_local = torch.eye(d, device=X.device)
        else:
            Sigma_inv_local = self.Sigma_inv.to(X.device)

        B_size = X.shape[0]
        Sigma_inv_batch = Sigma_inv_local.unsqueeze(0).expand(B_size, -1, -1)
        W1 = torch.bmm(Sigma_inv_batch, W1.unsqueeze(-1)).squeeze(-1)
        return W1

    def predict(self, Z):
        """
        Predict for all tokens in prompt (B, N+1, d+1).

        Args:
            Z: The full prompt with N training tokens + 1 test token.

        Returns:
            (B, N+1) predictions for each token's x_i.
        """
        X = Z[:, :, :-1]
        Y = Z[:, :-1, -1]
        W1 = self._compute_W1(X[:, :-1], Y)
        return torch.einsum('bd,bnd->bn', W1, X)

    def predict_single(self, Z):
        """
        Predict only the test token (the last token).

        Args:
            Z: (B, N+1, d+1).

        Returns:
            (B,) predicted test labels.
        """
        X = Z[:, :, :-1]
        Y = Z[:, :-1, -1]
        x_test = Z[:, -1, :-1]
        W1 = self._compute_W1(X[:, :-1], Y)
        return torch.einsum('bd,bd->b', W1, x_test)


def constructed_transformer_predict(Z, eta, device, precond=None):
    """
    One-step preconditioned GD via a single-layer linear attention formula.

    Short:
      Q_construct = -eta * (precond or I), P_construct = 0 so bottom-right is 1.
      Apply attention, final test pred is negative of out[:, -1, -1].

    Args:
        Z: (B, N+1, d+1).
        eta: scalar stepsize.
        device: Torch device.
        precond: if not None, multiply by that matrix. Else identity.

    Returns:
        (B,) predicted test labels.
    """
    d = Z.shape[2] - 1

    # Zero P => 1 in bottom-right after extension
    P_construct = torch.zeros(d, d, device=device)

    if precond is None:
        Q_construct = -eta * torch.eye(d, device=device)
    else:
        Q_construct = -eta * precond

    out = attention(P_construct, Q_construct, Z, activation=None)
    return -out[:, -1, -1]