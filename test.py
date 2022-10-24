from typing import Any, Dict
import jax
import jax.numpy as jnp


x = jnp.array([1, 2, 3])


@jax.vmap
def g(x):
    return x + 1


@jax.jit
def f(x):
    x = g(x)
    return x * 2


class A(Dict[str, Any]):
    pass


tree = A({"a": 1, "b": 2})

print(jax.tree_map(lambda x: x + 1, tree))
