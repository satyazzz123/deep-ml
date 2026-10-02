import torch


def smooth_labels(
    y_true: torch.Tensor,
    num_classes: int,
    epsilon: float
) -> torch.Tensor:

    y_true = torch.as_tensor(y_true, dtype=torch.long)

    # One-hot encoding
    labels = torch.eye(num_classes)[y_true]

    # Smoothed probabilities
    off_value = epsilon / num_classes
    on_value = 1 - epsilon + off_value

    smoothed = torch.where(
        labels == 1,
        on_value,
        off_value
    )

    return smoothed


def label_smoothing_cross_entropy(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
    epsilon: float = 0.1,
    round_decimals: int = None
) -> float:

    logits = torch.as_tensor(logits, dtype=torch.float32)

    # Stable log-softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # Smoothed targets
    smoothed = smooth_labels(
        y_true,
        num_classes,
        epsilon
    )

    # Cross entropy for every sample
    losses = -(smoothed * log_probs).sum(dim=-1)

    # Mean over batch
    loss = losses.mean()

    if round_decimals is not None:
        loss = torch.round(
            loss * (10 ** round_decimals)
        ) / (10 ** round_decimals)

    return loss.item()