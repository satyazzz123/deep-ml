import numpy as np
def compute_throughput(intervals):
    """
    Compute step-wise, cumulative, and overall tokens-per-second.

    Args:
        intervals: list of (tokens_processed, elapsed_seconds) tuples.

    Returns:
        dict with keys 'step_tps', 'cumulative_tps', 'overall_tps'.
    """
    intervals=np.asarray(intervals)
    if len(intervals)==0:
       return {
        "step_tps":[],
        "cumulative_tps":[],
        "overall_tps":0
        }
    z=np.where(intervals[:,1]<=0)
    if len(z[0])>0:
      raise ValueError("Your custom error message goes here")
    
    
      
    step_tps=(intervals[:,0]/intervals[:,1]).tolist()

    cumulative_tps=[]
    for i in range(1,len(intervals)+1):
      sum=np.sum(intervals[:i,:],axis=0)
      tps=np.round(sum[0]/sum[1],decimals=2)
      cumulative_tps.append(tps.item())


    return {
        "step_tps":step_tps,
        "cumulative_tps":cumulative_tps,
        "overall_tps":cumulative_tps[-1]
    }

    
