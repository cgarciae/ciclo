from functools import partial
import jax
import jax.numpy as jnp

print(jax.local_devices())

d = jnp.array([1, 2, 3])
t = (d, d)

# t = jax.tree_map(jnp.array, t)


@partial(jax.jit, donate_argnums=(0,))
def f(x):
    return jax.tree_map(lambda x: x + 1, x)


jnp.copy

t = f(t)

print(t)
