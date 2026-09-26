import torch
import torch.nn.functional as F


def unet_time_embedding(
    timesteps: list,
    embed_dim: int,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    max_period: int = 10000,
) -> torch.Tensor:

    W1 = torch.as_tensor(W1, dtype=torch.float32)
    b1 = torch.as_tensor(b1, dtype=torch.float32)
    W2 = torch.as_tensor(W2, dtype=torch.float32)
    b2 = torch.as_tensor(b2, dtype=torch.float32)

    timesteps = torch.as_tensor(timesteps, dtype=torch.float32)

    # Number of sine/cosine frequencies
    half_dim = embed_dim // 2

    # [half_dim]
    frequencies = max_period ** (
        -torch.arange(half_dim, dtype=torch.float32) / half_dim
    )

    # [B, 1] * [1, half_dim] -> [B, half_dim]
    angles = timesteps.unsqueeze(1) * frequencies.unsqueeze(0)

    # [B, half_dim]
    sin = torch.sin(angles)
    cos = torch.cos(angles)

    # [B, embed_dim]
    sinusoidal = torch.cat([sin, cos], dim=1)

    # MLP
    # [B, embed_dim] @ [embed_dim, hidden_dim]
    h = F.silu(sinusoidal @ W1 + b1)

    # [B, hidden_dim] @ [hidden_dim, output_dim]
    output = h @ W2 + b2

    return output