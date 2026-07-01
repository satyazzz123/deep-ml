import torch
import math
def normal_pdf(x: torch.Tensor, mean: torch.Tensor, std_dev: torch.Tensor) -> float:
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated (torch.Tensor scalar).
    :param mean: The mean (mu) of the distribution (torch.Tensor scalar).
    :param std_dev: The standard deviation (sigma) of the distribution (torch.Tensor scalar).
    :return: The PDF value rounded to 5 decimal places.
    """
    # Your code here
    x=torch.as_tensor(x,dtype=torch.float32)
    mean=torch.as_tensor(mean,dtype=torch.float32)
    std_dev=torch.as_tensor(std_dev,dtype=torch.float32)


    pdf=(1/(std_dev*math.sqrt(2*math.pi)))*torch.exp(-(x-mean)**2/(2*(std_dev**2)))
    return round(pdf.item(),5)