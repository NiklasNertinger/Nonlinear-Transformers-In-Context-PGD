"""
Contains:
1) A single-head or multi-head linear (optionally nonlinear) attention function.
2) A Transformer class for in-context learning.
3) A helper transformer_predict function to extract the model’s test prediction.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def attention(P, Q, Z, activation=None):
    """
    Compute self-attention for a single head with extended (d+1) dimension.

    A short summary:
      The function extends P, Q to handle the label dimension, constructs an attention
      mask to exclude the test label from referencing itself, computes attn_mat = Z Q Z^T,
      optionally applies an activation, then multiplies by key = P Z.

    Args:
        P: The (d x d) 'value' projection, extended to (d+1 x d+1) internally.
        Q: The (d x d) 'query-key' product, similarly extended.
        Z: (B, N+1, d+1) tokens, with the last token typically being (x_test, 0).
        activation: Callable or None. If None, it's linear; otherwise ReLU/softmax etc.

    Returns:
        (B, N+1, d+1) The updated tokens after attention, scaled by 1/N.
    """
    B = Z.shape[0]
    N = Z.shape[1] - 1
    d = Z.shape[2] - 1

    # Extend P to P_full, Q to Q_full, each shape (d+1, d+1).
    P_full = torch.cat([P, torch.zeros(1, d, device=device)], dim=0)
    P_full = torch.cat([P_full, torch.zeros(d + 1, 1, device=device)], dim=1)
    P_full[d, d] = 1

    Q_full = torch.cat([Q, torch.zeros(1, d, device=device)], dim=0)
    Q_full = torch.cat([Q_full, torch.zeros(d + 1, 1, device=device)], dim=1)

    # Construct the mask A that excludes the test label from referencing itself.
    A = torch.eye(N + 1, device=device)
    A[N, N] = 0

    # attn_mat = Z * Q_full * Z^T
    attn_mat = torch.einsum('BNi, ij, BMj->BNM', Z, Q_full, Z)
    if activation is not None:
        attn_mat = activation(attn_mat)

    # key = P_full * Z
    key = torch.einsum('ij, BNj->BNi', P_full, Z)
    # final output = attn_mat * A * key, scaled by 1/N
    out = torch.einsum('BNM, ML, BLi->BNi', attn_mat, A, key)
    return out / N


class Transformer(nn.Module):
    """
    A multi-layer, multi-head Transformer with extended linear attention.

    Short:
      Stores parameters in allparam for P/Q, builds up attention across n_layer layers,
      n_head heads each, adding them residually.

    Longer:
      - allparam shape: (n_layer, n_head, 2, d, d).
      - forward pass: for each layer, sum attention heads -> residual connection.
      - activation can be linear, ReLU, LeakyReLU, or softmax.

    Attributes:
        n_layer: Number of self-attention layers.
        n_head: Number of heads in each layer.
        d: Feature dimension (excluding label dimension).
        activation: Callable or None for the attn matrix transformation.
        act_str: String for the chosen activation name.
    """
    def __init__(self, n_layer, n_head, d, var, activation=None, leaky_alpha=0.5):
        super().__init__()
        # allparam shape: (n_layer, n_head, 2, d, d)
        self.register_parameter(
            'allparam',
            nn.Parameter(torch.zeros(n_layer, n_head, 2, d, d))
        )
        with torch.no_grad():
            self.allparam.normal_(0, var)

        self.n_layer = n_layer
        self.n_head = n_head
        self.d = d

        # Choose activation
        if activation is None:
            self.activation = None
            self.act_str = "linear"
        else:
            act = activation.lower()
            if act == "relu":
                self.activation = F.relu
                self.act_str = "relu"
            elif act == "leakyrelu":
                self.activation = lambda x: F.leaky_relu(x, negative_slope=leaky_alpha)
                self.act_str = f"leakyrelu{leaky_alpha}"
            elif act == "softmax":
                self.activation = lambda x: F.softmax(x, dim=2)
                self.act_str = "softmax"
            else:
                raise ValueError(f"Unknown activation: {activation}")

    def forward(self, Z):
        """
        Forward pass over tokens (B, N+1, d+1).

        The last token is typically (x_test, 0).
        At each layer, we sum attention from n_head heads, then add it (residual).

        Args:
            Z: (B, N+1, d+1) tokens.

        Returns:
            (B, N+1, d+1) tokens after all attention layers.
        """
        for layer_idx in range(self.n_layer):
            Z_prev = Z
            residues = 0
            for head_idx in range(self.n_head):
                P_ij = self.allparam[layer_idx, head_idx, 0]
                Q_ij = self.allparam[layer_idx, head_idx, 1]
                residues += attention(P_ij, Q_ij, Z_prev, activation=self.activation)
            Z = Z_prev + residues
        return Z

    def zero_p(self):
        """
        Zero out the P matrices in allparam (the "value" projection),
        typically for theoretical constraints on certain dims.
        """
        with torch.no_grad():
            for layer_idx in range(self.n_layer):
                for head_idx in range(self.n_head):
                    self.allparam[layer_idx, head_idx, 0].zero_()


def transformer_predict(model, Z):
    """
    Convenience function: run model(Z) and extract test prediction as -output[:, -1, -1].

    Args:
        model: The Transformer.
        Z: (B, N+1, d+1).

    Returns:
        (B,) test predictions (by convention: negative of final token's final coordinate).
    """
    out = model(Z)
    return -out[:, -1, -1]