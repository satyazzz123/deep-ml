import torch
import math
def chi_square_probability(x: float, k: int) -> float:
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    # your code here
    numerator=(x**((0.5*k-1)))*math.exp(-1*x*0.5)
    denominator=2**(0.5*k)*math.gamma(k/2)
    probability=numerator/denominator
    return round(probability, 3)