import jax
import jax.numpy as jnp

def df_dx(x):
    def f(x):
        return x**3 + 2*x

    df = jax.grad(f)
    return float(df(x))