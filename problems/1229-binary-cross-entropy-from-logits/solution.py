import torch
import torch.nn.functional as F
def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    x = torch.tensor(logits)
    y = torch.tensor(targets)

    loss = (
    torch.clamp(x, min=0)
    - x * y
    + F.softplus(-torch.abs(x))
    
    )
    return torch.mean(loss).item()
