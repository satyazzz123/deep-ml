import numpy as np

def simulate_clt(num_samples: int, sample_size: int, distribution: str = 'uniform') -> float:
    """
    Compute the mean of sample means to demonstrate the sampling distribution.

    Args:
        num_samples: Number of independent samples to draw
        sample_size: Size of each sample
        distribution: 'uniform' (0,1) or 'exponential' (scale=1)

    Returns:
        Mean of the sample means (float)
    """
    # Your code here
    sum=0
    # np.random.seed(42)
    for _ in range(num_samples):
      if(distribution=="uniform"):
        distribution_val=np.random.uniform(0, 1, sample_size)
        

      else:
        distribution_val=np.random.exponential(1, sample_size)
      population_mean=np.mean(distribution_val).item()
      sum+=population_mean
    return round(sum/num_samples,4)
