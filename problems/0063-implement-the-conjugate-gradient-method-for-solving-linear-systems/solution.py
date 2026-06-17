import torch

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):

    A = torch.as_tensor(A, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)

    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = torch.as_tensor(x0, dtype=torch.float64)

    r_i = b - A @ x
    p_i = r_i.clone()

    for i in range(n):

        alpha = torch.dot(r_i, r_i) / torch.dot(p_i, A @ p_i)

        x = x + alpha * p_i

        r_new = r_i - alpha * (A @ p_i)

        if torch.norm(r_new) < tol:
            break

        beta = torch.dot(r_new, r_new) / torch.dot(r_i, r_i)

        p_i = r_new + beta * p_i

        r_i = r_new

    return x