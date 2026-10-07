import torch

def early_stopping(
    val_losses: torch.Tensor,
    patience: int = 5,
    min_delta: float = 0.0
) -> torch.Tensor:

    val_losses = torch.as_tensor(val_losses, dtype=torch.float32)

    counter = 0
    mask = [False]

    for i in range(1, len(val_losses)):

        improvement = val_losses[i - 1] - val_losses[i]

        if improvement > min_delta + 1e-6:
            counter = 0
            mask.append(False)

        else:
            counter += 1

            if counter >= patience:
                mask.append(True)
                counter = 0
            else:
                mask.append(False)

    return torch.tensor(mask, dtype=torch.bool)