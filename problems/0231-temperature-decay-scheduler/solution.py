import torch
import math

def temperature_decay_pytorch(
    schedule_type: str,
    initial_temp: float,
    current_step: int,
    total_steps: int,
    final_temp: float = 0.01,
    decay_rate: float = 0.95) -> torch.Tensor:

    
    temp=initial_temp

    if schedule_type=="linear":
      temp=initial_temp-(initial_temp-final_temp)*(current_step/total_steps)
    if schedule_type=="exponential":
      temp=max(initial_temp*decay_rate**current_step,final_temp)
    if schedule_type=="cosine":
      temp=final_temp+0.5*(initial_temp-final_temp)*(1+math.cos((math.pi*current_step)/total_steps))

    return torch.tensor(temp)
    