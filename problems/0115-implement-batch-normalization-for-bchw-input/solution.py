import torch 

def batch_normalization(
    X,
    gamma,
    beta,
    
    
    epsilon=1e-5,
   
):
    momentum=0.1
    X=torch.as_tensor(X,dtype=torch.float32)
    gamma=torch.as_tensor(gamma,dtype=torch.float32)
    beta=torch.as_tensor(beta,dtype=torch.float32)
    training=True
    B, C, H, W = X.shape
    running_mean=None
    running_var=None
    if running_mean is None:
        running_mean = torch.zeros(C,dtype=torch.float32)

    if running_var is None:
        running_var = torch.ones(C,dtype=torch.float32)

    if training:
        # Batch statistics
        mean = torch.mean(X, dim=(0, 2, 3), keepdims=True)
        var = torch.var(X, dim=(0, 2, 3), keepdims=True,correction=0)

        # Update running statistics
        running_mean = (
            (1 - momentum) * running_mean
            + momentum * mean.reshape(C)
        )

        running_var = (
            (1 - momentum) * running_var
            + momentum * var.reshape(C)
        )
    else:
        mean = running_mean.reshape(1, C, 1, 1)
        var = running_var.reshape(1, C, 1, 1)

    # Normalize
    X_hat = (X - mean) / torch.sqrt(var + epsilon)

    # Scale and shift
    output = gamma * X_hat + beta

    return output