import torch

def GeLU(x: torch.Tensor) -> torch.Tensor:
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)

    sigma_gate=0.5*(1+torch.erf(x/2**0.5))

    return torch.round((x*sigma_gate),decimals=4)