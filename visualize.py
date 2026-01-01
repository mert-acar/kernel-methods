from typing import Callable, Dict, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """Convert torch tensors to numpy arrays."""
    return tuple(t.cpu().numpy() for t in tensors)  # type: ignore


def _plot_predictions(
    ax: plt.Axes,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Plot training data, test data, and predictions."""
    ax.scatter(x_train, y_train, alpha=0.5, label="Train")
    ax.scatter(x_test, y_test, alpha=0.5, label="Test")
    ax.scatter(x_test, y_pred, alpha=0.5, label="Predictions")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Energy (MWh)")
    ax.grid()
    ax.legend()


def _plot_alpha_coefficients(
    ax: plt.Axes, x_train: np.ndarray, alpha: np.ndarray
) -> None:
    """Plot alpha coefficients vs input features."""
    ax.scatter(x_train, alpha, alpha=0.6)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Alpha coefficient")
    ax.set_title("Alpha vs Temperature")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)


def _plot_influential_points(
    ax: plt.Axes,
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    n_points: int = 20,
) -> None:
    """Highlight most influential training points based on |alpha|."""
    influential_idx = np.argsort(np.abs(np.squeeze(alpha)))[::-1][:n_points]
    ax.scatter(x_train, y_train, alpha=0.3, label="All points")
    ax.scatter(
        x_train[influential_idx],
        y_train[influential_idx],
        c="red",
        s=100,
        alpha=0.6,
        label="High |alpha|",
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Energy (MWh)")
    ax.legend()
    ax.set_title("Most Influential Points")


def _plot_contributions(
    ax: plt.Axes,
    x_train: np.ndarray,
    x_test: np.ndarray,
    alpha: np.ndarray,
    kernel_fn: Callable,
    kernel_params: Dict[str, Any],
) -> None:
    """Plot contribution of each training point to a specific test prediction."""
    test_point_idx = 50 if len(x_test) > 50 else len(x_test) // 2
    x_test_point = x_test[test_point_idx : test_point_idx + 1]

    k_values = kernel_fn(
        torch.tensor(x_test_point), torch.tensor(x_train), **kernel_params
    ).numpy()
    contributions = np.squeeze(alpha) * np.squeeze(k_values)

    ax.scatter(x_train, contributions, alpha=0.6)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Contribution to prediction")
    ax.set_title(f"Contributions for x={np.squeeze(x_test_point):.1f}°C")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)


def plot_diagnostics(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: torch.Tensor,
    kernel_params: Dict[str, Any],
    kernel_fn: Callable,
) -> None:
    """Plot diagnostic visualizations for kernel ridge regression."""
    x_train_np, y_train_np, x_test_np, y_test_np, y_pred_np, alpha_np = _to_numpy(
        x_train, y_train, x_test, y_test, y_pred, alpha
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 6))

    _plot_predictions(
        axes[0, 0], x_train_np, y_train_np, x_test_np, y_test_np, y_pred_np
    )
    _plot_alpha_coefficients(axes[0, 1], x_train_np, alpha_np)
    _plot_influential_points(axes[1, 0], x_train_np, y_train_np, alpha_np)
    _plot_contributions(
        axes[1, 1], x_train_np, x_test_np, alpha_np, kernel_fn, kernel_params
    )

    plt.tight_layout()
    plt.show()
