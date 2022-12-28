import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

import ciclo
from ciclo import managed
from flax import struct
from typing import Any


class GaussianDiffusion(struct.PyTreeNode):
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_bars: jnp.ndarray


class State(managed.ManagedState):
    key: jax.random.KeyArray
    ema_params: Any
    process: GaussianDiffusion
    config: int = struct.field(pytree_node=False)
    loss_fn: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, *, apply_fn, params, tx, key, process, config, loss_fn, **kwargs):
        ema_params = jax.tree_map(jnp.copy, params)
        return super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            ema_params=ema_params,
            key=key,
            process=process,
            config=config,
            loss_fn=loss_fn,
            **kwargs
        )


class TestManaged:
    def test_data_parallel_donate(self):
        process = GaussianDiffusion(
            betas=jnp.ones(10), alphas=jnp.ones(10), alpha_bars=jnp.ones(10)
        )
        module = nn.Dense(features=1)
        variables = module.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))

        state = State.create(
            apply_fn=module.apply,
            params=variables["params"],
            tx=optax.adam(1e-3),
            key=jax.random.PRNGKey(0),
            process=process,
            config=1,
            loss_fn=2,
            strategy="data_parallel_donate",
        )

        @managed.train_step
        def train_step(state, batch):
            loss = sum(jnp.mean(x) for x in jax.tree_util.tree_leaves(state.params))
            logs = ciclo.logs()
            logs.add_loss("loss", loss)
            return logs, state

        logs, state = train_step(state, None)
