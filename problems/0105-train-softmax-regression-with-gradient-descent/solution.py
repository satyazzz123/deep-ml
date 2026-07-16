import torch
import torch.nn as nn
import torch.nn.functional as F

def train_softmaxreg(X: torch.Tensor, y: torch.Tensor, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.long)  
    
    ones_part = torch.ones(x.shape[0], 1)
    x = torch.concat((ones_part, x), dim=1)
    
    unique_labels = torch.unique(y).numel()
    
    
    w = torch.zeros(unique_labels, x.shape[-1], requires_grad=True)
    
    
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    one_hot = F.one_hot(y, num_classes=unique_labels).float()
    
    loss_values = []
    softmax_layer = nn.Softmax(dim=-1)
    
    for _ in range(iterations):
        logits = x @ w.T
        # probabilities=softmax_layer(logits)
        
        loss = loss_fn(logits, one_hot) 
        loss_values.append(loss.item())
        
        loss.backward()
        
        
        with torch.no_grad():
            w -= learning_rate * w.grad
        w.grad.zero_()
        
    return (w.detach().tolist(),loss_values)
    