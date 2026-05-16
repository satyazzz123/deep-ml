import torch


def calculate_covariance_matrix(matrix):
    matrix = torch.as_tensor(matrix, dtype=torch.float32)

    # Mean center
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=True)

    standardized_matrix = (matrix - mean) / std


    n_samples = standardized_matrix.shape[0]

    # Covariance matrix
    covariance_matrix = (
        standardized_matrix.T @ standardized_matrix
    ) / (n_samples - 1)

    return covariance_matrix


def pca(data, k):
    """
    Perform PCA and return top-k principal components.

    Args:
        data: shape (n_samples, n_features)
        k: number of principal components

    Returns:
        Tensor of shape (n_features, k)
    """

    # Convert to tensor
    data = torch.as_tensor(data, dtype=torch.float32)

    # Step 1: Compute covariance matrix
    covariance_matrix = calculate_covariance_matrix(data)

    # Step 2: Eigen decomposition
    # eigh is used because covariance matrix is symmetric
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Step 3: Sort eigenvalues/eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Step 4: Select top-k eigenvectors
    principal_components = eigenvectors[:, :k]

    # Step 5: Apply sign convention
    for i in range(k):

        vector = principal_components[:, i]

        # Find first element with abs > 1e-10
        for j in range(len(vector)):

            if abs(vector[j]) > 1e-10:

                # Flip sign if negative
                if vector[j] < 0:
                    principal_components[:, i] *= -1

                break

    # Round to 4 decimals
    principal_components = torch.round(
        principal_components * 10000
    ) / 10000

    return principal_components