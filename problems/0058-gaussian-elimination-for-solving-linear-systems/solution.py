import torch

def gaussian_elimination(A, b):
    A = torch.as_tensor(A, dtype=torch.float64).clone()
    b = torch.as_tensor(b, dtype=torch.float64).clone()

    n = A.shape[0]

    # Forward elimination
    for i in range(n):

        # Partial pivoting
        pivot_row = i + torch.argmax(torch.abs(A[i:, i])).item()

        if abs(A[pivot_row, i]) < 1e-12:
            raise ValueError("Singular matrix")

        if pivot_row != i:
            temp = A[i].clone()
            A[i] = A[pivot_row]
            A[pivot_row] = temp

            temp = b[i].clone()
            b[i] = b[pivot_row]
            b[pivot_row] = temp

        # Eliminate below pivot
        for j in range(i + 1, n):

            factor = A[j, i] / A[i, i]

            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = torch.zeros(n, dtype=torch.float64)

    for i in range(n - 1, -1, -1):

        s = torch.dot(A[i, i + 1:], x[i + 1:])

        x[i] = (b[i] - s) / A[i, i]

    return x