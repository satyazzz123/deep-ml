import torch
from typing import Tuple

def newton_schulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7
) -> torch.Tensor:

    G = torch.as_tensor(G, dtype=torch.float32)

    a = 3.4445
    b = -4.7750
    c = 2.0315

    transposed = False

    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True

    for _ in range(steps):
        A = G @ G.T
        G = a * G + (b * A + c * (A @ A)) @ G

    if transposed:
        G = G.T

    return G
def normalization(A: torch.Tensor, eps: float = 1e-7):
    frob_norm = torch.sqrt(torch.sum(A ** 2))
    A_normalized = A / (frob_norm + eps)

    return A_normalized, frob_norm

    

def muon_step(
    theta: torch.Tensor,
    B: torch.Tensor,
    grad: torch.Tensor,
    eta: float,
    mu: float,
    ns_steps: int = 5,
    eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:

    theta = torch.as_tensor(theta, dtype=torch.float32)
    B = torch.as_tensor(B, dtype=torch.float32)
    grad = torch.as_tensor(grad, dtype=torch.float32)

    # 1. Momentum
    B_new = mu * B + grad

    # 2. Frobenius normalization
    normalized_B, frob_norm = normalization(B_new, eps)

    # 3. Newton-Schulz orthogonalization
    O = newton_schulz5(
        normalized_B,
        steps=ns_steps,
        eps=eps
    )

    # 4. Scaling
    M, N = B_new.shape
    scale = (M * N)**0.5 / (frob_norm + eps)

    # 5. Parameter update
    theta_new = theta - eta * scale * O

    return theta_new, B_new