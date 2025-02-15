"""
Contains:
1) Functions to generate synthetic linear-regression data (isotropic or non-isotropic) with optional additive Gaussian noise.
2) A helper for creating random orthogonal matrices via the Haar measure.
"""


import torch


def generate_linear_regression_data(
    batch_size,
    num_tokens,
    feature_dim,
    sigma=None,
    noise_variance=0.0,
    device=None
):
    """
    Generate a batch of linear-regression prompts with optional non-isotropic features and additive Gaussian noise on training labels.

    If sigma is None, features \(x_i\) are drawn from \( \mathcal{N}(0, I)\); otherwise, \(x_i \sim \mathcal{N}(0, \sigma)\).
    A random \(w_*\) is used to produce the noiseless training labels \(y_i = w_*^\top x_i\).
    Gaussian noise with mean 0 and variance noise_variance is then added to the training labels.
    The test token's label is set to 0 (and remains noise-free) so that it can be used for evaluation.

    Args:
        batch_size (int): Number of prompts to generate.
        num_tokens (int): Number of training tokens (excludes the single test token).
        feature_dim (int): Dimension of each feature vector \(x_i\).
        sigma (torch.Tensor or None): Covariance matrix for \(x_i\); if None, isotropic data is used.
        noise_variance (float): Variance of the Gaussian noise added to training labels (default 0.0 means no noise).
        device (torch.device or None): Device on which to place the generated data.

    Returns:
        tuple:
            Z (torch.Tensor): Prompt tensor of shape (batch_size, num_tokens+1, feature_dim+1), where training tokens are
                              \((x_i, y_i^\text{noisy})\) and the test token is \((x_\text{test}, 0)\).
            y_test (torch.Tensor): True test labels (without noise), shape (batch_size,).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate features.
    if sigma is None:
        # Isotropic case: x ~ N(0, I)
        X = torch.randn(batch_size, num_tokens + 1, feature_dim, device=device)
    else:
        # Non-isotropic case: x ~ N(0, sigma)
        mean = torch.zeros(feature_dim, device=device)
        mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=sigma)
        X = mvn.sample((batch_size, num_tokens + 1))

    # Sample w_* for each prompt.
    w_star = torch.randn(batch_size, feature_dim, device=device)
    # Compute noiseless training labels for the first num_tokens tokens.
    y_train = torch.einsum('bi,bni->bn', w_star, X[:, :num_tokens]).unsqueeze(-1)

    # Add Gaussian noise (if noise_variance > 0) to the training labels.
    if noise_variance > 0:
        noise_std = noise_variance ** 0.5
        noise = torch.randn_like(y_train) * noise_std
        y_train = y_train + noise

    # Construct training tokens as concatenation of features and (possibly noisy) labels.
    Z_train = torch.cat([X[:, :num_tokens], y_train], dim=2)

    # For the test token, the label is set to 0.
    test_label_zero = torch.zeros(batch_size, 1, 1, device=device)
    Z_test = torch.cat([X[:, num_tokens:], test_label_zero], dim=2)

    # Concatenate training tokens and test token.
    Z_full = torch.cat([Z_train, Z_test], dim=1)
    # Compute the true test labels (without noise) for evaluation.
    y_test = torch.einsum('bi,bni->bn', w_star, X[:, num_tokens:]).squeeze(1)

    return Z_full, y_test


def random_orthogonal(d, device=None, dtype=torch.float32):
    """
    Generate a d x d random orthogonal matrix via the Haar measure.

    Args:
        d (int): Dimension of the matrix.
        device (torch.device or None): Torch device.
        dtype (torch.dtype): Data type of the returned matrix.

    Returns:
        (torch.Tensor): A (d x d) orthogonal matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(d, d, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    diag_R = torch.diag(R)
    signs = torch.sign(diag_R)
    signs[signs == 0] = 1
    Q = Q * signs.unsqueeze(0)
    return Q