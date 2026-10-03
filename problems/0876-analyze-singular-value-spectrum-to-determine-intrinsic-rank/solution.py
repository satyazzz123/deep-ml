import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
  is_zero_matrix = not np.any(delta_W)
  if is_zero_matrix:
    return 0
  U,S,Vh = np.linalg.svd(delta_W, full_matrices=True, compute_uv=True, hermitian=False)
  # S=np.array(S)
  squared_singular=S**2
  cumulative_sum=np.sum(squared_singular)
  for i in range(1,len(squared_singular)+1):
    indx=i
    sum=np.sum(squared_singular[:i])
    
    if (sum/cumulative_sum)>=energy_threshold:
      
      break

  return i
	