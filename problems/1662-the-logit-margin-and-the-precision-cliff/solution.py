import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    z=np.asarray(z)
    other_logits=np.delete(z,t)
    max_elem_removed=other_logits-np.max(other_logits)
    return float(z[t]-(np.max(other_logits)+np.log(np.sum(np.exp(max_elem_removed)))))






def precision_cliff(dtype):
    dt = np.dtype(dtype).type

    def saturated(m):
        # Compute exp in float64, then cast it.
        exp_term = dt(np.exp(-m))

        # Perform the addition in the requested dtype.
        denominator = dt(dt(1.0) + exp_term)

        # Take the reciprocal in the requested dtype.
        probability = np.reciprocal(denominator)

        # Did the target probability round to exactly 1?
        return probability == dt(1.0)

    low = 0.0
    high = 100.0

    while high - low >= 1e-9:
        mid = (low + high) / 2.0

        if saturated(mid):
            high = mid
        else:
            low = mid

    return round(high, 4)

