import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    
    if len(T_t)!=len(T_t[0]) and len(S_t)!=len(S_t[0]):
        return torch.tensor(-1)

    shape_A=A_t.size()
    shape_T=T_t.size()
    shape_S=S_t.size()
    if shape_T[-1]!=shape_A[0] and shape_A[-1]!=shape_S[0]:
        return torch.tensor(-1)

    inverse_T=torch.linalg.inv(T_t)

    result = inverse_T @ A_t @ S_t

    return result

    
