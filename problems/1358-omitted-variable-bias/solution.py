import numpy as np


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept.

    Args:
        X: (n, p) design matrix without an intercept column.
        y: (n,) target.

    Returns:
        (p + 1,) coefficients, intercept first.
    """
    n, p = X.shape

    # Add intercept column
    X_b = np.column_stack((np.ones(n), X))

    # OLS: β = (XᵀX)^(-1)Xᵀy
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    return beta


def omitted_variable_bias(
    X: np.ndarray,
    y: np.ndarray,
    omit_idx: int
) -> tuple:
    """Return (full_kept, short, bias) for a two-column X."""

    # -------------------------
    # 1. Full model
    # y = β₀ + β₁x₁ + β₂x₂
    # -------------------------
    full = ols(X, y)

    # -------------------------
    # 2. Remove the omitted variable
    # -------------------------
    X_short = np.delete(X, omit_idx, axis=1)

    # -------------------------
    # 3. Short model
    # y = γ₀ + γ₁x_kept
    # -------------------------
    short = ols(X_short, y)

    # -------------------------
    # 4. Find coefficient of
    #    the variable we kept
    # -------------------------

    # X has two variables.
    # If we omit 0, we keep 1.
    # If we omit 1, we keep 0.
    kept_idx = 1 - omit_idx

    # +1 because index 0 is the intercept
    full_kept = full[kept_idx + 1]

    # short = [intercept, kept_variable]
    short_kept = short[1]

    # -------------------------
    # 5. OVB
    # -------------------------
    bias = short_kept - full_kept

    return full_kept, short_kept, bias