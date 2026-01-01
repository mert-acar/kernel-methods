# Kernel Methods

Clean implementation of kernel ridge regression with cross-validation. Built this to experiment with different kernel functions without dealing with sklearn's abstractions

## What it does

Implements kernel ridge regression from scratch using PyTorch. Solves the dual problem directly:

$$
\alpha = (\textbf{K}_{\text{train}} + \lambda\textbf{I})^{-1} y_{\text{train}}
$$
$$
\hat{y} = \textbf{K}_{\text{test}}\alpha
$$

where K is the kernel matrix computed over training data.

## Kernels

Three kernel functions implemented:

- **RBF/Gaussian**: $`k(x,x') = e^{(-\gamma\|x - x'\|^{2})}`$ - good for smooth nonlinear patterns
- **Polynomial**: $`k(x,x') = (x^{T} x' + c)^{d}`$ - works when you know the degree of nonlinearity
- **Linear**: $`k(x,x') = x^T x'`$ - baseline, equivalent to ridge regression in original space

## Cross-validation

Grid search over hyperparameters using k-fold CV. For RBF this searches over gamma values, for polynomial it searches over degree and coefficient combinations. Ridge coefficient (lambda) is always tuned.

## Dataset

Synthetic energy consumption data as a function of temperature. Quadratic relationship with additive Gaussian noise. The underlying function models how energy usage increases when temp deviates from optimal (heating/cooling).

## Notes

- Uses MPS backend for Apple Silicon. Change `DEVICE` in train.py if you're on CUDA/CPU.
- Kernel matrix inversion is $`O(n^{3})`$, gets slow above ~10k samples. Consider [Nystrom approximation](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations) or random features for large-scale problems.
- No bias term in the model - kernel ridge regression typically doesn't need one if you center your data or use an appropriate kernel.

