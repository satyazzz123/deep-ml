import torch
import torch.nn.functional as F
class MixedPrecision:
    def __init__(self, loss_scale: float = 1024.0):
        # Initialize loss scaling factor
        self.loss_scale=loss_scale

    
    def forward(self, weights: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        # Perform forward pass with float16, return scaled loss as Python float
        w=torch.as_tensor(weights,dtype=torch.float16)
        x=torch.as_tensor(inputs,dtype=torch.float16)
        y=torch.as_tensor(targets,dtype=torch.float16).reshape(-1,1)
        y_pred=x@w
        loss=F.mse_loss(y_pred.reshape(-1,1),y)

        return (self.loss_scale*loss).item()

    
    def backward(self, gradients: torch.Tensor) -> torch.Tensor:
        # Unscale gradients and check for overflow, return as float32
        gradients=torch.as_tensor(gradients,dtype=torch.float32)
        gradients = torch.nan_to_num(
        gradients,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
        )

        return gradients/self.loss_scale