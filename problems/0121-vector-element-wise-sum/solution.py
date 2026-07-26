import torch

def vector_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | int:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    a=torch.as_tensor(a,dtype=torch.float16)
    b=torch.as_tensor(b,dtype=torch.float16)
    if len(a)==0 or len(b)==0 or len(a)!=len(b):
        return -1
    return a+b