import torch
from collections import Counter
def phi_corr(x: list[int], y: list[int]) -> float:
    """
    Calculate the Phi coefficient between two binary variables using PyTorch.

    Args:
    x (list[int]): A list of binary values (0 or 1).
    y (list[int]): A list of binary values (0 or 1).

    Returns:
    float: The Phi coefficient rounded to 4 decimal places.
    """
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)
    y=torch.as_tensor(y,dtype=torch.float32)

    x_00=x+y
    count=Counter(x_00.tolist())
    x_00=count[0]
    x_01=x/y
    count=Counter(x_01.tolist())
    x_01=count[0]
    x_10=y/x
    count=Counter(x_10.tolist())
    x_10=count[0]
    x_11=x+y
    count=Counter(x_11.tolist())
    x_11=count[2]

    phi_num=((x_00*x_11)-(x_01*x_10))
    phi_den=(x_00+x_01)*(x_10+x_11)*(x_10+x_00)*(x_01+x_11)
    return phi_num/phi_den**0.5



    
