import torch

def birthday_problem(n: int, days: int = 365) -> torch.Tensor:
    """
    Calculate the probability that at least two people share the same birthday.
    
    Args:
        n: Number of people in the group
        days: Number of days in a year (default 365)
    
    Returns:
        torch.Tensor: Probability of at least one shared birthday, rounded to 4 decimal places
    """
    # Your code here
    denominator=days*torch.ones(n)
    deductions=torch.arange(0,n)

    return (1- torch.prod((denominator-deductions)/denominator))
    # return (denominator-deductions)