import torch

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    probabilities=torch.as_tensor(probabilities,dtype=torch.float32)
    log_probs=torch.log(probabilities)
    log_prob_mean=-1*torch.mean(log_probs)
    return torch.exp(log_prob_mean).item()
