import torch
from typing import Tuple

def early_stopping(
    val_losses: torch.Tensor,
    patience: int,
    min_delta: float
) -> Tuple[int, int]:

    val_losses = torch.as_tensor(val_losses, dtype=torch.float32)

    best_loss_value = val_losses[0]
    best_loss_idx = 0

    bad_epochs = 0
    stopping_epoch = len(val_losses)-1  # if we never early-stop

    for i in range(1, len(val_losses)):

        delta = best_loss_value - val_losses[i]

        # Meaningful improvement
        if delta >= min_delta:
            best_loss_value = val_losses[i]
            best_loss_idx = i
            bad_epochs = 0

        # No meaningful improvement
        else:
            bad_epochs += 1

        # Patience exhausted
        if bad_epochs >= patience:
            stopping_epoch = i 
            break

    # +1 because tensor indices are 0-based but epochs are 1-based
    best_epoch = best_loss_idx 

    return stopping_epoch, best_epoch