import torch

def rref(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(matrix, dtype=torch.float32).clone()

    num_rows, num_cols = matrix.shape

    current_row = 0

    for current_col in range(num_cols):

        # Stop if we've processed all rows
        if current_row >= num_rows:
            break

        # Find a non-zero pivot in this column
        pivot_row = None

        for r in range(current_row, num_rows):
            if matrix[r, current_col] != 0:
                pivot_row = r
                break

        # No pivot found in this column
        if pivot_row is None:
            continue

        # Swap pivot row into position
        if pivot_row != current_row:
            matrix[[current_row, pivot_row]] = matrix[[pivot_row, current_row]]

        # Get pivot value
        pivot = matrix[current_row, current_col]

        # Make pivot = 1
        matrix[current_row] = matrix[current_row] / pivot

        # Make all other entries in pivot column = 0
        for r in range(num_rows):
            if r == current_row:
                continue

            matrix[r] -= matrix[r, current_col] * matrix[current_row]

        # Move to next row
        current_row += 1

    return matrix