import torch
from typing import Callable, Dict, List, Any
from dataclasses import dataclass


@dataclass
class KernelConfig:
    """Configuration for a kernel function including its hyperparameters."""

    name: str
    kernel_fn: Callable
    default_params: Dict[str, Any]
    param_grid: Dict[str, List[Any]]  # hyperparameters


def rbf_kernel(
    x1: torch.Tensor, x2: torch.Tensor, gamma: float = 0.1, **kwargs
) -> torch.Tensor:
    """
    RBF (Gaussian) kernel: k(x,x') = exp(-gamma * ||x - x'||^2)

    Compute pairwise squared distances:
    ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2*x_i^T*x_j
    """
    x1_norm = (x1**2).sum(1).view(-1, 1)  # [d, 1]
    x2_norm = (x2**2).sum(1).view(1, -1)  # [1, d]
    dist_sq = x1_norm + x2_norm - (2 * (x1 @ x2.T))
    return torch.exp(-gamma * dist_sq)


def polynomial_kernel(
    x1: torch.Tensor,
    x2: torch.Tensor,
    degree: int = 2,
    coef: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    """
    Polynomial kernel: k(x,x') = (x^T x' + c)^d
    """
    return (x1 @ x2.T + coef) ** degree


def linear_kernel(x1: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Linear kernel: k(x,x') = x^T x'
    """
    return x1 @ x2.T


KERNEL_CONFIGS = {
    "rbf": KernelConfig(
        name="rbf",
        kernel_fn=rbf_kernel,
        param_grid={"gamma": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]},
        default_params={"gamma": 0.1},
    ),
    "poly": KernelConfig(
        name="poly",
        kernel_fn=polynomial_kernel,
        param_grid={
            "degree": [2, 3, 4, 5],
            "coef": [0.0, 0.1, 1.0, 10.0],
        },
        default_params={"degree": 2, "coef": 1.0},
    ),
    "linear": KernelConfig(
        name="linear",
        kernel_fn=linear_kernel,
        param_grid={},  # Linear kernel has no hyperparameters
        default_params={},
    ),
}


def get_kernel_config(kernel: str) -> KernelConfig:
    """Get kernel configuration by name."""
    if kernel not in KERNEL_CONFIGS:
        available = ", ".join(KERNEL_CONFIGS.keys())
        raise NotImplementedError(
            f"Kernel '{kernel}' is not implemented. Available: {available}"
        )
    return KERNEL_CONFIGS[kernel]
