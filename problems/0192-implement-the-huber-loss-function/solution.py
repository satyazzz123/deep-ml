import torch

def huber_loss(y_true, y_pred, delta=1.0) -> float:
    y_true = torch.as_tensor(y_true, dtype=torch.float32)
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32)

    error = y_true - y_pred
    abs_error = torch.abs(error)

    loss = torch.where(
        abs_error <= delta,
        0.5 * error**2,
        delta * (abs_error - 0.5 * delta)
    )

    return torch.mean(loss).item()