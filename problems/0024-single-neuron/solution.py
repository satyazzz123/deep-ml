import torch
import torch.nn.functional as F

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> tuple[list[float], float]:
    # Ensure all inputs are float tensors to avoid MSE errors
    features_t = torch.as_tensor(features, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.float32)
    weights_t = torch.as_tensor(weights, dtype=torch.float32)
    
    # 1. Linear combination (dot product + bias)
    # Using @ for matrix-vector multiplication
    val = features_t @ weights_t.T + bias
    
    # 2. Sigmoid activation
    activation_t = torch.sigmoid(val)
    
    # 3. MSE Loss
    loss_t = F.mse_loss(activation_t, labels_t)
    
    # 4. Rounding and Type Conversion
    # Use a list comprehension for the probabilities list
    activation = [round(x, 4) for x in activation_t.tolist()]
    
    # Use .item() for the single loss value
    loss = round(loss_t.item(), 4)
    
    return activation, loss