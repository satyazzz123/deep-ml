import torch

def diff(a: torch.Tensor) -> torch.Tensor:
    """First-order difference; out[0]=a[0], out[i]=a[i]-a[i-1] (no torch.diff)."""
    # Your code here
    a=torch.as_tensor(a,dtype=torch.float32)
    i=len(a)
    return torch.concat((a[0].unsqueeze(0),a[1:]-a[:i-1]))
    