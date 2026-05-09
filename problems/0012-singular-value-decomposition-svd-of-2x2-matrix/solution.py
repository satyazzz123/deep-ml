import torch
import math
import numpy as np

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float)
    trace = []
    for j in range(len(matrix)):
        trace.append(matrix[j][j])
                
    trace_tensor = torch.tensor(trace)
    final_trace = torch.sum(trace_tensor).item()
    determinant = torch.det(matrix).item()
    
    # Quadratic formula: det(M - λI) = λ² - trace(M)λ + det(M) = 0
    discriminant = (final_trace**2) - 4 * determinant
    # Ensure we don't sqrt a tiny negative due to float precision
    underscore_determinant = math.sqrt(max(0, discriminant))
    
    eigen_value_1 = (final_trace + underscore_determinant) / 2
    eigen_value_2 = (final_trace - underscore_determinant) / 2
    return torch.tensor([eigen_value_1, eigen_value_2])

def get_single_eigenvector(M):
    """
    Finds one non-zero vector v such that Mv = 0 for a 2x2 singular matrix.
    If M = [[a, b], [c, d]], then v = [b, -a] or [-d, c].
    """
    a, b = M[0][0].item(), M[0][1].item()
    c, d = M[1][0].item(), M[1][1].item()
    
    # Use the row that has larger values for better numerical stability
    if abs(a) + abs(b) > abs(c) + abs(d):
        v = torch.tensor([b, -a])
    else:
        v = torch.tensor([d, -c])
        
    # Normalize the vector (make its length 1)
    norm = torch.norm(v)
    if norm > 1e-9:
        v = v / norm
    return v

def svd_2x2_singular_values(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = torch.as_tensor(A, dtype=torch.float)
    A_T = A.transpose(0, 1)
    
    # 1. Compute A^T A
    stretch_strength = A_T @ A
    
    # 2. Get Eigenvalues and Singular Values
    eigen_values = calculate_eigenvalues(stretch_strength)
    # Sort eigenvalues descending
    eigen_values, _ = torch.sort(eigen_values, descending=True)
    
    singular_values = torch.sqrt(torch.clamp(eigen_values, min=0))
    
    # 3. Find V (eigenvectors of A^T A)
    v_list = []
    identity_matrix = torch.eye(2)
    for i in range(2):
        # M = (A^T A - λI)
        M = stretch_strength - eigen_values[i] * identity_matrix
        v = get_single_eigenvector(M)
        v_list.append(v)
    
    # V is the matrix where each column is an eigenvector
    V = torch.stack(v_list, dim=1)
    
    # 4. Find U (using u_i = A * v_i / σ_i)
    u_list = []
    for i in range(2):
        if singular_values[i] > 1e-9:
            u = (A @ V[:, i]) / singular_values[i]
        else:
            # Handle zero singular value by finding a vector orthogonal to the first u
            if i == 1:
                u = torch.tensor([-u_list[0][1], u_list[0][0]])
            else:
                u = torch.tensor([1.0, 0.0]) # Fallback
        u_list.append(u)
        
    U = torch.stack(u_list, dim=1)
    
    # SVD convention returns V transposed
    Vt = V.transpose(0, 1)
    
    return U, singular_values, Vt

# Test the function
