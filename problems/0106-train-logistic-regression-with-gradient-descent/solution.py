import torch
import torch.nn as nn

def train_logreg(
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float,
    iterations: int
) -> tuple[list[float], list[float]]:
    """
    Train logistic regression using gradient descent with BCE loss.
    """

    # Convert inputs
    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    # Add bias column
    bias = torch.ones((x.shape[0], 1))
    x = torch.cat((bias, x), dim=1)

    # Initialize weights
    w = torch.zeros((x.shape[1], 1), requires_grad=True)

    # BCE loss (sum reduction as required)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

    losses = []

    for _ in range(iterations):

        # Forward
        logits = x @ w
        loss = loss_fn(logits, y)

        # Save loss
        losses.append(round(loss.item(), 4))

        # Backward
        loss.backward()

        # Gradient descent
        with torch.no_grad():
            w -= learning_rate * w.grad

        # Clear gradients
        w.grad.zero_()

    # Convert coefficients to Python list
    coefficients = [round(v, 4) for v in w.squeeze().tolist()]

    return coefficients, losses