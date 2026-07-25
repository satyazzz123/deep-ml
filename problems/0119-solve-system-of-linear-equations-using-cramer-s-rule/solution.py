import torch

def cramers_rule(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solve system of linear equations Ax = b using Cramer's Rule.
    Returns solution vector x or -1 if no unique solution exists.
    """
    A=torch.as_tensor(A,dtype=torch.float32)
    b=torch.as_tensor(b,dtype=torch.float32)
    A_det=torch.linalg.det(A)
    det_val=[]
    if A_det==0:
      return -1
    
    for i in range(len(A)):
      A_clone=A.clone()
      A_clone[:,i]=b
      det_val.append(torch.linalg.det(A_clone)/A_det)
    return torch.tensor(det_val)

