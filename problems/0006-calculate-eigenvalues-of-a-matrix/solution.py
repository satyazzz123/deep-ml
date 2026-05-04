import torch
import math

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:

    # Your implementation here
    matrix=torch.as_tensor(matrix,dtype=torch.float)
    trace=[]
    for j in range(len(matrix)):
        for i in range(len(matrix[0])):
            if j==i:
                trace.append(matrix[j][i])
    trace_tensor=torch.tensor(trace)
    final_trace=torch.sum(trace_tensor).tolist()
    determinant = torch.det(matrix).item()
    
    underscore_determinant=math.sqrt(((final_trace)**2)-4*1*determinant)
    eigen_value_1= (final_trace+underscore_determinant)/2
    eigen_value_2= (final_trace-underscore_determinant)/2
    eigen_values=[eigen_value_1,eigen_value_2]
    return torch.tensor(eigen_values)

    
            
    
