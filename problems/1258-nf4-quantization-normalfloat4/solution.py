
import numpy as np

NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)

def nf4_quantize(x: np.ndarray):
    """
    NF4 block quant. scale=absmax; y=x/scale; q=argmin |y-NF4|; x_hat=scale*NF4[q].
    If absmax==0: q all 7, scale 1.0, x_hat 0.
    Ties: smallest index (np.argmin).
    Returns q (int), scale (float), x_hat (float array)
    """
    # Your code here
    
    x_abs=np.abs(x)
    scale=np.max(x_abs)
    if scale==0:
      q_i=7*np.ones_like(x, dtype='int8')
      scale=1.0
      x_hat=np.zeros_like(x)
      return(q_i,scale,x_hat)
    normalized=x/scale
    q_i_mod=np.abs(normalized.reshape(-1,1)-NF4)
    q_i=np.argmin(q_i_mod,keepdims=True,axis=-1)
    w,h=q_i.shape
    q_i=q_i.reshape(w*h)
    x_hat=[]
    
    for i in q_i:
      x_hat.append((scale*NF4[i]).item())
    return (np.array(q_i),scale,np.array(x_hat))
