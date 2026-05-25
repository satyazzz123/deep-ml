import torch
import torch.nn.functional as F
from typing import Optional

def to_categorical(x: torch.Tensor,
                   n_col: Optional[int] = None) -> torch.Tensor:

    x = torch.as_tensor(x, dtype=torch.long)

    if n_col is None:
        n_col = torch.max(x).item() + 1

    return F.one_hot(x, num_classes=n_col).float()