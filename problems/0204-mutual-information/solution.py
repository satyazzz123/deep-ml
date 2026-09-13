import torch

def mutual_information(joint_prob: torch.Tensor) -> float:
    """
    Compute the mutual information between two random variables.
    
    Args:
        joint_prob: 2D joint probability distribution P(X,Y) as a torch.Tensor
    
    Returns:
        Mutual information I(X;Y)
    """
    # Your code here
    joint_prob=torch.as_tensor(joint_prob,dtype=torch.float32)
    x_marginals=torch.sum(joint_prob,dim=-1)
    y_marginals=torch.sum(joint_prob,dim=0)
    marginals=torch.stack((x_marginals,y_marginals),dim=1)
    sum=0
    for i in range(len(joint_prob)):
      for j in range(len(joint_prob[0])):
        if joint_prob[i][j]>0:
          sum+=joint_prob[i,j]*torch.log(joint_prob[i][j]/(x_marginals[i]*y_marginals[j]+1e-7))

    return sum.item()