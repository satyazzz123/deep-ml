import torch

def apply_weight_decay(parameters: list[torch.Tensor], gradients: list[torch.Tensor], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[torch.Tensor]:
    """
    Apply weight decay (L2 regularization) to parameters.
    
    Args:
        parameters: List of parameter tensors
        gradients: List of gradient tensors
        lr: Learning rate
        weight_decay: Weight decay factor
        apply_to_all: Boolean list indicating which parameter groups get weight decay
    
    Returns:
        Updated parameters
    """
    # Your code here
    for i in range(len(apply_to_all)):
      if apply_to_all[i]:
        parameters[i]=parameters[i]*(1-lr*weight_decay)-lr*gradients[i]
      else:
         parameters[i]=parameters[i]-lr*gradients[i]

    return parameters