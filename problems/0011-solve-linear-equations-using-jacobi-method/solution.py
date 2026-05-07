import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    shape=A_t.size()
    b_t = torch.as_tensor(b, dtype=torch.float)
    D_t=torch.zeros(shape)
    R_t=A_t.clone().detach()
    for i in range(len(A_t[0])):
      D_t[i][i]+=R_t[i][i]
    
    R_t=R_t-D_t
    D_t_inverse=torch.linalg.inv(D_t)

    # val= D_t_inverse(b_t-R_t*previous_value)  
    previous_value=torch.zeros(shape[1]) 
    for i in range(n):
      previous_value=torch.matmul(D_t_inverse,b_t-torch.matmul(R_t,previous_value))
      previous_value=torch.round(previous_value,decimals=4)

    return previous_value
