import torch
import torch.nn as nn
import torch.nn.functional as F

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    features=torch.as_tensor(features,dtype=torch.float32)
    initial_weights =torch.tensor(initial_weights ,dtype=torch.float32, requires_grad=True)
    initial_bias =torch.tensor( initial_bias,dtype=torch.float32, requires_grad=True)
    labels  =torch.as_tensor( labels ,dtype=torch.float32)
    criterion = nn.MSELoss()
    loss_values=[]
    for _ in range(epochs):
      value=features@initial_weights.T + initial_bias
      activation=F.sigmoid(value)
      
      loss=criterion(activation,labels)
      loss_values.append(loss.item())
      loss.backward()

      with torch.no_grad():
        initial_weights-=learning_rate*initial_weights.grad
        initial_bias-=learning_rate*initial_bias.grad
        initial_weights.grad.zero_()
        initial_bias.grad.zero_()
    return (
    [round(x,4) for x in initial_weights.tolist()],
    round(initial_bias.item(),4),
    [round(x,4) for x in loss_values]
    )




    

