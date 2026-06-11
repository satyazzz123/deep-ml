import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    # 1. Convert everything to PyTorch tensors safely
    input_sequence = torch.as_tensor(input_sequence, dtype=torch.float32)
    h_t = torch.as_tensor(initial_hidden_state, dtype=torch.float32) # Use h_t to track state
    Wx = torch.as_tensor(Wx, dtype=torch.float32)
    Wh = torch.as_tensor(Wh, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)

    # 2. Iterate through each time step in the sequence
    for x_t in input_sequence:
        # Compute the next hidden state
        # Formula: h_t = tanh(x_t @ Wx.T + h_t @ Wh.T + b)
        h_t = torch.tanh(x_t @ Wx.T + h_t @ Wh.T + b)
        
    # 3. Round to 4 decimal places as requested by the docstring
    return torch.round(h_t, decimals=4)