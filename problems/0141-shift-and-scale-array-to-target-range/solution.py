import torch

def convert_range(values: torch.Tensor, c: float, d: float) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.float32)

    a = torch.min(values)
    b = torch.max(values)

    c = torch.as_tensor(c, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)

    return c + ((values - a) / (b - a)) * (d - c)