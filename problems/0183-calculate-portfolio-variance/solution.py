import torch

def calculate_portfolio_variance(cov_matrix: list[list[float]], weights: list[float]) -> float:
    """
    Calculate the variance of a portfolio using PyTorch.
    """
    # Convert inputs to tensors
    cov_matrix = torch.tensor(cov_matrix, dtype=torch.float)
    weights = torch.tensor(weights, dtype=torch.float)
    return (weights.T@cov_matrix@weights).item()
    