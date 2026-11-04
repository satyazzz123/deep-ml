import torch

def thanksgiving_dish_predictor(preference_scores: torch.Tensor) -> torch.Tensor:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.
    
    Args:
        preference_scores: Tensor of preference scores for each dish
        
    Returns:
        Tensor of probabilities for each dish
    """
    # Your code here
    preference_scores=torch.as_tensor(preference_scores,dtype=torch.float32)

    scores=torch.softmax(preference_scores,dim=-1)
    return scores