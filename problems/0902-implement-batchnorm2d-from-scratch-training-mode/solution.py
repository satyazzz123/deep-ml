import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # x: (B, C, H, W)

    x = torch.as_tensor(x, dtype=torch.float32)

    mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)

    variance = torch.var(
        x,
        dim=(0, 2, 3),
        unbiased=False,
        keepdim=True
    )

    x_hat = (x - mean) / torch.sqrt(variance + eps)
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    return x_hat * gamma + beta
