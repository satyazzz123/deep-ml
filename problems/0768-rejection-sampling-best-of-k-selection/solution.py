import numpy as np
def rejection_sampling_best_of_k(candidates, scores):
    """
    Select the highest-scoring candidate per prompt.

    Args:
        candidates: list of N lists, each containing K candidate outputs.
        scores: list of N lists, each containing K reward scores.

    Returns:
        List of N selected candidates.
    """
   
    indx=[score.index(max(score)) for score in scores]
    
    result=[]
    for i in range(len(indx)):
      result.append(candidates[i][indx[i]])
    return result


