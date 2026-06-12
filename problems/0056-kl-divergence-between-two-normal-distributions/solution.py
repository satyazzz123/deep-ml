import torch
import math

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
    value = math.log(sigma_q / sigma_p)

    value += (
        sigma_p**2 +
        (mu_p - mu_q)**2
    ) / (2 * sigma_q**2)

    value -= 0.5

    return torch.tensor(value)