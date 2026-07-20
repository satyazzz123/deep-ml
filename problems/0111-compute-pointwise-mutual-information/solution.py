import torch
import math
def compute_pmi(joint_counts: int, total_counts_x: int, total_counts_y: int, total_samples: int) -> torch.Tensor:
    """
    Compute Pointwise Mutual Information (PMI) given the joint occurrence count of two events,
    their individual counts, and the total number of samples.
    """
    p_x=total_counts_x/total_samples
    p_y=total_counts_y/total_samples
    p_x_y=joint_counts/total_samples
    val=(math.log2(p_x_y/(p_x*p_y)))
    val=torch.tensor(val)
    return val
