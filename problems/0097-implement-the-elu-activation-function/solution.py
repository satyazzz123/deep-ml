import math
def elu(x: float, alpha: float = 1.0) -> float:
  if x>0:
    return x
  val=alpha*(math.exp(x)-1)
  return round(val,4)
	