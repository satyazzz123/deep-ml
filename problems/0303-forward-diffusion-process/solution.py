import torch

def forward_diffusion(x_0: torch.Tensor, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: torch.Tensor) -> torch.Tensor:
    """
    Apply forward diffusion process to add noise to input data.
    
    Args:
        x_0: Original input data (torch.Tensor)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: Noise tensor (same shape as x_0)
    
    Returns:
        Noisy sample x_t as torch.Tensor
    """
    beta_scheduler=[beta_start+((i-1)/(num_timesteps-1+1e-7))*(beta_end-beta_start) for i in range(1,num_timesteps+1)]
    beta_scheduler=torch.tensor(beta_scheduler)
    alpha=1-beta_scheduler

    alpha_dash=[]

    for i in range(1,len(alpha)+1):
      alpha_dash.append(torch.prod(alpha[:i]))
    
    alpha_dash=torch.stack(alpha_dash)
    t=t-1

    x_t=x_0*alpha_dash[t]**0.5+noise*(1-alpha_dash[t])**0.5 
    return x_t
