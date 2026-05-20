import torch
from typing import Union

def make_diagonal(x: Union[torch.Tensor, list, "np.ndarray"]) -> torch.Tensor:
    """Return a diagonal matrix whose diagonal elements are the 1-D values in `x`.
    If `x` is not a torch tensor it will be converted automatically.
    
    Hint: `torch.diag_embed` makes this very short!
    """
    # âï¸  Your code here
    diag=[]
    for i in range(len(x)):
        row=[]
        for j in range(len(x)):
            if i==j:
                row.append(x[i])
            else:
                row.append(0)
        diag.append(row)
    return torch.tensor(diag,dtype=torch.float32)

        
    
