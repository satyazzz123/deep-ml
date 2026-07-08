import torch
import math
PI = 3.14159
def power_grid_forecast(consumption_data: torch.Tensor) -> int:
    """
    Forecast power consumption for day 15 using linear regression.
    
    Steps:
    1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    2) Perform linear regression on the detrended data using torch.linalg.
    3) Predict day 15's base consumption.
    4) Add the day 15 fluctuation back.
    5) Round, then add a 5% safety margin (rounded up).
    6) Return the final integer.
    
    Args:
        consumption_data: torch.Tensor of shape (10,) with daily consumption values
    Returns:
        int: Forecasted consumption with safety margin
    """
    consumption_data=torch.as_tensor(consumption_data,dtype=torch.float32)
    updated_consumption_data=[]
    for i in range(1,len(consumption_data)+1):
      fluctuation=10*math.sin((2*PI*i)/(10))
      updated_consumption_data.append(consumption_data[i-1].item()-fluctuation)
    n=len(consumption_data)
    numerator_first_part=0
    
    denominator=0
    x_i=0
    y_i=0
    x_i_square=0
    for i in range(1,len(updated_consumption_data)+1):
      numerator_first_part+=i*updated_consumption_data[i-1]
      x_i+=i
      y_i+=updated_consumption_data[i-1]
      x_i_square+=i**2
    numerator_second_part=x_i*y_i
    numerator=n*numerator_first_part-numerator_second_part
    denominator=n*x_i_square-x_i**2
    m=numerator/denominator
    b=(y_i-m*x_i)/n

    base=m*15+b
    pred=base+10*math.sin((2*PI*15)/(10))
    final=math.ceil(1.05 * round(pred))
    
    


      
    return final
      