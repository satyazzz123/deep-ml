
import torch


def _betacf(a, b, x, max_iter=200, eps=3e-14):
    """Continued fraction for the incomplete beta function."""
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / (
            (qam + m2) * (a + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / (
            (a + m2) * (qap + m2)
        )
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return h


def _regularized_beta(x, a, b):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        torch.lgamma(torch.tensor(a + b, dtype=torch.float64))
        - torch.lgamma(torch.tensor(a, dtype=torch.float64))
        - torch.lgamma(torch.tensor(b, dtype=torch.float64))
        + a * torch.log(torch.tensor(x, dtype=torch.float64))
        + b * torch.log(torch.tensor(1.0 - x, dtype=torch.float64))
    )
    bt = torch.exp(log_bt).item()

    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a

    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def two_sided_p_value(t_statistic, df):
    """Two-sided p-value for a Student's t statistic."""
    t = abs(float(torch.as_tensor(t_statistic).item()))
    df = float(torch.as_tensor(df).item())

    x = df / (df + t * t)
    return _regularized_beta(x, df / 2.0, 0.5)


def two_sample_t_test(sample1, sample2, alpha: float = 0.05) -> dict:
    """
    Welch's independent two-sample t-test using PyTorch.

    Returns t_statistic, p_value, degrees_of_freedom,
    reject_null, and cohens_d.
    """
    sample1 = torch.as_tensor(sample1, dtype=torch.float64)
    sample2 = torch.as_tensor(sample2, dtype=torch.float64)

    if sample1.ndim != 1 or sample2.ndim != 1:
        raise ValueError("Samples must be one-dimensional.")
    if len(sample1) < 2 or len(sample2) < 2:
        raise ValueError("Each sample must contain at least 2 observations.")
    if not torch.isfinite(sample1).all() or not torch.isfinite(sample2).all():
        raise ValueError("Samples must contain only finite values.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sample1.mean(), sample2.mean()
    var1, var2 = sample1.var(correction=1), sample2.var(correction=1)

    # Welch's standard error and t-statistic
    se_squared = var1 / n1 + var2 / n2
    if se_squared.item() == 0.0:
        raise ValueError(
            "Both samples have zero variance; the t-statistic is undefined."
        )

    se = torch.sqrt(se_squared)
    t_statistic = (mean1 - mean2) / se

    # Welch-Satterthwaite degrees of freedom
    df_denominator = (
        (var1 / n1) ** 2 / (n1 - 1)
        + (var2 / n2) ** 2 / (n2 - 1)
    )
    degrees_of_freedom = se_squared**2 / df_denominator

    # Two-sided p-value from the actual t-statistic and degrees of freedom
    p_value = two_sided_p_value(t_statistic, degrees_of_freedom)

    # Conventional pooled-SD Cohen's d
    pooled_variance = (
        (n1 - 1) * var1 + (n2 - 1) * var2
    ) / (n1 + n2 - 2)

    if pooled_variance.item() > 0.0:
        cohens_d = (mean2 - mean1) / torch.sqrt(pooled_variance)
        cohens_d_result = torch.abs(cohens_d).item()
    else:
        cohens_d_result = float("nan")

    return {
        "t_statistic": t_statistic.item(),
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom.item(),
        "reject_null": p_value < alpha,
        "cohens_d": cohens_d_result,
    }