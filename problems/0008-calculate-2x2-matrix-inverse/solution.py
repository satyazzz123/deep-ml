import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    m = torch.as_tensor(matrix, dtype=torch.float)
    size = m.shape[0]

    # Check square
    if m.shape[0] != m.shape[1]:
        return None

    # Check determinant
    if torch.isclose(torch.det(m), torch.tensor(0.0)):
        return None

    identity = torch.eye(size)
    aug = torch.cat((m, identity), dim=1)

    for i in range(size):

        # Pivot handling (swap if zero)
        if aug[i][i] == 0:
            for j in range(i + 1, size):
                if aug[j][i] != 0:
                    aug[[i, j]] = aug[[j, i]]
                    break
            else:
                return None

        # Normalize pivot row
        pivot = aug[i][i]
        aug[i] = aug[i] / pivot

        # Eliminate other rows
        for k in range(size):
            if k != i:
                factor = aug[k][i]
                aug[k] = aug[k] - factor * aug[i]

    # Extract inverse
    inverse = aug[:, size:]

    return inverse