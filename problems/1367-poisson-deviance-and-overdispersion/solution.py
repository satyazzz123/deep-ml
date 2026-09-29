import numpy as np


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance, using the convention 0 * log(0) = 0."""
    # Your code here
    y=y+1e-10
    mu=mu+1e-10
    D=2*(y*np.log( y/mu )-(y-mu))
    return np.sum(D,axis=-1).item()


def dispersion_ratio(y: np.ndarray, mu: np.ndarray, n_params: int) -> float:
    """Pearson chi-square divided by (n - n_params)."""
    # Your code here
    n_obs=len(y)
    df=n_obs-n_params
    
    sigma=1/df * np.sum((y-mu)**2/mu)

    return sigma
