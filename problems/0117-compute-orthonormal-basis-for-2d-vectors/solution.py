import torch

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[torch.Tensor]:
    """
    Compute an orthonormal basis for the subspace spanned by a list of vectors
    using the Gram-Schmidt process.

    Args:
        vectors: A list of vectors (e.g., [[1, 2], [3, 4], ...])
        tol: Tolerance for determining linear independence

    Returns:
        A list of orthonormal torch.Tensor vectors that span the same subspace.
    """
    basis = []

    for v in vectors:
        # Convert to a float tensor
        w = torch.tensor(v, dtype=torch.float32)

        # Remove projections onto all previously computed basis vectors
        for u in basis:
            projection = torch.dot(w, u) * u
            w = w - projection

        # Skip if the vector is linearly dependent
        norm = torch.norm(w)
        if norm > tol:
            # Normalize
            u = w / norm
            basis.append(u)

    return basis