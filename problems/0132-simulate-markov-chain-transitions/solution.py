import torch

def simulate_markov_chain(transition_matrix: torch.Tensor, initial_state: int, num_steps: int) -> torch.Tensor:
    """Simulates a Markov Chain given a transition matrix, initial state, and number of steps.
    
    Parameters:
    transition_matrix : 2D torch.Tensor, transition probabilities where each row sums to 1.
    initial_state : int, starting state index.
    num_steps : int, number of steps to simulate.
    
    Returns:
    torch.Tensor, tensor of state indices over time, including the initial state.
    """
    transition_matrix=torch.as_tensor(transition_matrix,dtype=torch.float32)
    torch.manual_seed(42)
    current_state=initial_state
    final_state=[current_state]
    for i in range(num_steps):
      probs=transition_matrix[current_state]
      x=torch.multinomial(probs,1)
      current_state=x.item()
      final_state.append(current_state)
    return torch.tensor(final_state)
      

    

