import torch

def checkpoint_forward(funcs, input_arr) -> torch.Tensor:
    """
    Applies a list of functions in sequence to the input tensor, simulating gradient checkpointing by not storing intermediates.

    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: The output after applying all functions, same shape as output of last function.
    """
    x=torch.as_tensor(input_arr)
    for i in funcs:
      x=i(x)
    return x