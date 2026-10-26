import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    x1=torch.tensor(x,dtype=torch.float32,requires_grad=True)
    x2=torch.tensor(x,dtype=torch.float32,requires_grad=True)
    x3=torch.tensor(x,dtype=torch.float32,requires_grad=True)
    derivatives={}
    relu_out=torch.relu(x1)
    tanh_out=torch.tanh(x2)
    sigmoid_out=torch.sigmoid(x3)
    relu_out.backward()
    tanh_out.backward()
    sigmoid_out.backward()
    derivatives["relu"]=x1.grad.item()
    derivatives["tanh"]=x2.grad.item()

    derivatives["sigmoid"]=x3.grad.item()
    return derivatives