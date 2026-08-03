import torch
import math
def compute_cross_entropy_loss(predicted_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-15) -> float:
    """Compute average cross-entropy loss for multi-class classification.
    
    Args:
        predicted_probs: Tensor of predicted probabilities (batch_size, num_classes)
        true_labels: One-hot encoded true labels (batch_size, num_classes)
        epsilon: Small value for numerical stability
    
    Returns:
        Average cross-entropy loss as a float
    """
    predicted_probs=torch.as_tensor(predicted_probs,dtype=torch.float32)
    predicted_probs = torch.clamp(predicted_probs, min=epsilon)
    true_labels=torch.as_tensor(true_labels,dtype=torch.float32)
    log_loss=torch.log(predicted_probs)
    log_loss[log_loss==-math.inf]=123456
    cross_entropy=log_loss*true_labels

    return (torch.sum(cross_entropy*-1,dim=(0,1))/predicted_probs.shape[0]).item()