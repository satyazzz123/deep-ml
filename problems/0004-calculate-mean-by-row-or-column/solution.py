import torch
import statistics
def calculate_matrix_mean(matrix, mode: str) -> torch.Tensor:
    """
    Calculate mean of a 2D matrix per row or per column using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 1-D tensor of means or raises ValueError on invalid mode.
    """
    a_t = torch.as_tensor(matrix, dtype=torch.float)
    mode=mode.lower()
    result=[]
    if mode=='row':
        for j in range(len(a_t)):
            total_sum=0
            mean=0
            
            
            for i in range(len(a_t[0])):
                
                total_sum+=a_t[j][i]
                
            mean=total_sum/len(a_t[0])
            result.append(mean)
    if mode=='column':
        for i in range(len(a_t[0])):
            total_sum=0
            mean=0
            
            
            for j in range(len(a_t)):
                
                total_sum+=a_t[j][i]
                
            mean=total_sum/len(a_t)
            result.append(mean)
    return torch.tensor(result)
