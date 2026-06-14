import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    """
    Solve Ax = b using the Gauss-Seidel method.

    Parameters
    ----------
    A : array-like
        Coefficient matrix (square).
    b : array-like
        Right-hand side vector.
    n : int
        Number of iterations.
    x_ini : array-like, optional
        Initial guess. If None, uses zeros.

    Returns
    -------
    numpy.ndarray
        Approximate solution after n iterations.
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    m = len(b)

    if x_ini is None:
        x = np.zeros(m, dtype=float)
    else:
        x = np.array(x_ini, dtype=float)

    for _ in range(n):

        for i in range(m):

            sigma = 0.0

            for j in range(m):
                if j != i:
                    sigma += A[i][j] * x[j]

            x[i] = (b[i] - sigma) / A[i][i]

    return x