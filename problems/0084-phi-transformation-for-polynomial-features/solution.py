import torch

def phi_transform(data: list[float], degree: int) -> torch.Tensor:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    Returns:
        torch.Tensor: A 2D tensor where each row represents the transformed features of a data point,
                      containing powers from 0 to degree.
    """
    # Your code here
    if isinstance(data,torch.Tensor):
      data=data.tolist()
    if data==[] or degree <0:
      return torch.tensor([])
    polynomial_features=[]
    for i in data:
      # print(i)
      temp=[]
      for j in range(degree+1):
        temp.append(i**j)
      polynomial_features.append(temp)



    return torch.tensor(polynomial_features)
