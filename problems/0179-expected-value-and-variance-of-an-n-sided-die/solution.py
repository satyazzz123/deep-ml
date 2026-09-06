import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    expected_value=(n+1)/2
    variance=((n+1)*(2*n+1)/6)-expected_value**2
    return (expected_value,variance)