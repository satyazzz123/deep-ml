import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    # Your implementation here
    X=torch.as_tensor(X,dtype=torch.float32)
    k=[]
    for i in range(len(X[0])):
     k.append(X[:,i].tolist())
    features=torch.tensor(k)
    
    means=torch.mean(features,dim=1)
    means=means.reshape(-1,1)
    features=features-means 
    
    standard_deviation=torch.std(features,dim=1)

    correlation=torch.zeros(features.shape[0],features.shape[0])
    for i in range(len(features)):
      for j in range(len(features)):
        correlation[i][j]+=(torch.dot(features[i],features[j])/(features.shape[-1]-1))/(standard_deviation[i]*standard_deviation[j])


    return correlation
    
    
