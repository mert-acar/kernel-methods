from typing import Callable, Dict, List, Tuple, Any
from itertools import product

from time import time
import numpy as np
import torch
from sklearn.model_selection import KFold

from dataset import generate_dataset
from model import get_kernel_config
from visualize import plot_diagnostics

DEVICE = torch.device("mps")


def kernel_ridge_regression(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    ridge_coeff: float,
    kernel_fn: Callable,
    kernel_params: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Kernel Ridge Regression with flexible kernel parameters.

    Args:
        x_train: Training features
        y_train: Training targets
        x_test: Test features
        ridge_coeff: Ridge regularization coefficient (lambda)
        kernel_fn: Kernel function to use
        kernel_params: Dictionary of kernel-specific parameters
    """
    K = kernel_fn(x_train, x_train, **kernel_params)
    n = K.shape[0]
    alpha = torch.linalg.solve(K + ridge_coeff * torch.eye(n, device=DEVICE), y_train)
    K_test = kernel_fn(x_test, x_train, **kernel_params)
    y_pred = K_test @ alpha
    return y_pred, alpha


def pick_best_params(
    x: torch.Tensor,
    y: torch.Tensor,
    lambdas: List[float],
    kernel_fn: Callable,
    kernel_param_grid: Dict[str, List[Any]],
    n_folds: int = 5,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Cross-validation to find best hyperparameters for kernel ridge regression.

    Args:
        x: Input features
        y: Target values
        lambdas: List of ridge coefficients to try
        kernel_fn: Kernel function
        kernel_param_grid: Dictionary of kernel parameter names to lists of values
        n_folds: Number of cross-validation folds

    Returns:
        best_score: Best cross-validation score
        best_lambda: Best ridge coefficient
        best_kernel_params: Best kernel parameters
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_score = float("inf")
    best_lambda = lambdas[0]
    best_kernel_params = {}

    param_names = list(kernel_param_grid.keys())
    if param_names:
        param_values = [kernel_param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
    else:
        param_combinations = [()]

    for lam in lambdas:
        for param_combo in param_combinations:
            kernel_params = {
                name: value for name, value in zip(param_names, param_combo)
            }
            fold_scores = []
            for train_idx, val_idx in kf.split(x):
                x_tr, x_val = x[train_idx], x[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                y_pred, _ = kernel_ridge_regression(
                    x_tr, y_tr, x_val, lam, kernel_fn, kernel_params
                )
                rmse = torch.sqrt(torch.mean((y_pred - y_val) ** 2))
                fold_scores.append(rmse.item())

            cv_score = np.mean(fold_scores)
            if cv_score < best_score:
                best_score = cv_score
                best_lambda = lam
                best_kernel_params = kernel_params.copy()

    return best_score, best_lambda, best_kernel_params


def main(
    verbose: bool = False,
    seed: int = 9001,
    n: int = 365,
    noise_scale: float = 4,
    train_portion: float = 0.8,
    lam: float = 0.01,
    n_splits: int = -1,
    kernel: str = "rbf",
    diagnostics: bool = False,
):
    """
    Train kernel ridge regression with cross-validation.

    Args:
        verbose: Print detailed progress information
        seed: Random seed for reproducibility
        n: Number of data points to generate
        noise_scale: Scale of noise to add to data
        train_portion: Fraction of data to use for training
        lab: lambda parameter of ridge regularization
        n_splits: Number of CV folds (-1 to skip CV, use defaults)
        kernel: Kernel type ('rbf', 'poly', 'linear')
        diagnostics: Whether to plot diagnostic visualizations
    """
    kernel_config = get_kernel_config(kernel)
    kernel_fn = kernel_config.kernel_fn

    if verbose:
        print(f"[INFO] Using {kernel_config.name} kernel")

    x, y = generate_dataset(seed=seed, n=n, noise_scale=noise_scale)
    indices = np.random.permutation(n)
    train_size = int(n * train_portion)
    train_idx, test_idx = indices[:train_size], indices[train_size:]
    x_train = torch.tensor(x[train_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    x_test = torch.tensor(x[test_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    y_test = torch.tensor(y[test_idx], dtype=torch.float32).reshape(-1, 1).to(DEVICE)
    if verbose:
        print(
            f"[INFO] Starting training with {len(x_train)} training and {len(x_test)} test data"
        )

    kernel_params = kernel_config.default_params.copy()

    if n_splits > 1:
        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
        if verbose:
            print(f"[INFO] Starting cross validation with {n_splits} splits...")
            param_space_size = len(lambdas)
            for param_values in kernel_config.param_grid.values():
                param_space_size *= len(param_values)
            print(f"[INFO] Searching {param_space_size} parameter combinations...")

        tick = time()
        best_score, lam, kernel_params = pick_best_params(
            x_train, y_train, lambdas, kernel_fn, kernel_config.param_grid, n_splits
        )

        if verbose:
            print(f"[INFO] Cross validation is done in {time() - tick:.4f} seconds!")
            params_str = " | ".join([f"{k} = {v}" for k, v in kernel_params.items()])
            print(
                f"[INFO] Best parameters: λ = {lam} | {params_str} | rMSE = {best_score:.2f}"
            )

    # Training
    if verbose:
        print("[INFO] Training starting...")

    tick = time()
    y_pred, alpha = kernel_ridge_regression(
        x_train, y_train, x_test, lam, kernel_fn, kernel_params
    )
    if verbose:
        print(f"[INFO] Training is done in {time() - tick:.4f} seconds!")

    # Evaluation
    rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2))
    print(f"Test RMSE: {rmse.item():.3f}")
    if diagnostics:
        plot_diagnostics(
            x_train, y_train, x_test, y_test, y_pred, alpha, kernel_params, kernel_fn
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
