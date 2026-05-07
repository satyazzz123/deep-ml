import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    features=v_t.size()
    mean=[]
    final_covariance_tensor=[]
    covariance=[]
    for j in range(len(v_t)):
        row_mean=torch.mean(v_t[j]).item()
        mean.append(row_mean)

    for j in range(len(v_t)):
      variance=[]
      for i in range(len(v_t[0])):
        variance.append((v_t[j][i]-mean[j]))
      covariance.append(variance)
    covariance_tensor=torch.tensor(covariance)
    for j in range(len(covariance)):
      for i in range(len(covariance)):
        final_covariance_tensor.append((torch.dot(covariance_tensor[j],covariance_tensor[i]))/(len(covariance[0])-1))

    final_covariance_tensor=torch.reshape(torch.tensor(final_covariance_tensor),(features[0],features[0]))
    return final_covariance_tensor

            

