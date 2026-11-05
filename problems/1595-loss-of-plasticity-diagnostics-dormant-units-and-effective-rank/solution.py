import numpy as np


def dormant_ratio(H, tau):
    mean_activations = np.abs(H).mean(axis=0)

    normalizer = mean_activations.mean()

    if normalizer == 0:
        scores = np.zeros_like(mean_activations)
    else:
        scores = mean_activations / normalizer

    dormant_indices = np.where(scores <= tau)[0]

    ratio = len(dormant_indices) / H.shape[1]

    return float(ratio), dormant_indices.tolist()


def effective_rank(H, eps=1e-12):
    Hc = H - H.mean(axis=0, keepdims=True)

    singular_values = np.linalg.svd(
        Hc,
        compute_uv=False
    )

    singular_values = singular_values[
        singular_values > eps
    ]

    if len(singular_values) == 0:
        return 0.0

    p = singular_values / singular_values.sum()

    entropy = -np.sum(p * np.log(p))

    return float(np.exp(entropy))


def plasticity_report(H, W, tau, decimals=4):
    dr, _ = dormant_ratio(H, tau)

    er = effective_rank(H)

    wn = np.linalg.norm(W, ord="fro")

    return {
        "dormant_ratio": round(dr, decimals),
        "effective_rank": round(er, decimals),
        "weight_norm": round(float(wn), decimals)
    }
