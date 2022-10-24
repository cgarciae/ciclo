from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass, replace
from functools import partial
import importlib.util
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import jax
import jax.numpy as jnp
from einop import einop
from flax import jax_utils
from flax.training import train_state
from typing_extensions import Protocol, runtime_checkable
from ciclo.api import GeneralCallback, LogsLike, Metric


class Dataclass(Protocol):
    __dataclass_fields__: Dict


@runtime_checkable
class HasKey(Protocol):
    key: jax.random.PRNGKeyArray


StrategyConstructor = Callable[[], "Strategy"]
A = TypeVar("A")
S = TypeVar("S", bound=Dataclass)
ME = TypeVar("ME", bound=Metric)


_REGISTRY: Dict[str, StrategyConstructor] = {}


# ----------------------------------------------------------------------------
# utils
# ----------------------------------------------------------------------------


@overload
def register_strategy(
    name: str,
) -> Callable[[StrategyConstructor], StrategyConstructor]:
    ...


@overload
def register_strategy(
    name: str,
    *,
    constructor: StrategyConstructor,
) -> None:
    ...


def register_strategy(
    name: str,
    *,
    constructor: Optional[StrategyConstructor] = None,
) -> Optional[Callable[[StrategyConstructor], StrategyConstructor]]:
    """
    Register a strategy class.
    """

    def _register(constructor: StrategyConstructor):
        if name in _REGISTRY:
            raise ValueError(f"Strategy {name} already registered")

        _REGISTRY[name] = constructor

    if constructor is None:

        def decorator(
            constructor: StrategyConstructor,
        ) -> StrategyConstructor:
            _register(constructor)
            return constructor

        return decorator
    else:
        _register(constructor)


def get_strategy(name: str) -> "Strategy":
    """
    Get a strategy class.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Strategy {name} not registered")

    return _REGISTRY[name]()


# ----------------------------------------------------------------------------
# register strategies
# ----------------------------------------------------------------------------

register_strategy(
    name="eager",
    constructor=lambda: Eager(),
)
register_strategy(
    name="jit",
    constructor=lambda: JIT(donate_args=False),
)
register_strategy(
    name="jit_donate",
    constructor=lambda: JIT(donate_args=True),
)
register_strategy(
    name="data_parallel",
    constructor=lambda: DataParallel(donate_args=False),
)
register_strategy(
    name="data_parallel_donate",
    constructor=lambda: DataParallel(donate_args=True),
)


# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------


class Strategy(ABC):
    def from_host(self, state: S) -> S:
        return state

    def to_host(self, state: S) -> S:
        return state

    def lift_batch(self, data: A) -> A:
        return data

    def lift_key(self, key: jax.random.PRNGKeyArray) -> jax.random.PRNGKeyArray:
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size

    def handle_gatherable(self, outputs: A) -> A:
        return outputs

    def handle_metric(self, metric: ME) -> ME:
        return metric

    def handle_grads(
        self,
        grads: Any,
    ) -> Any:
        return grads

    def handle_batch_stats(
        self,
        batch_stats: Any,
    ) -> Any:
        return batch_stats

    def handle_averageable(self, x: A) -> A:
        return x

    @abstractmethod
    def __call__(self, callback: GeneralCallback) -> GeneralCallback:
        ...


@dataclass(eq=True, frozen=True)
class Eager(Strategy):
    def __call__(self, callback: GeneralCallback) -> GeneralCallback:
        return callback


@dataclass(eq=True, frozen=True)
class JIT(Strategy):
    donate_args: bool = False

    def __call__(self, callback: GeneralCallback) -> GeneralCallback:
        return jax.jit(
            callback,
            donate_argnums=0 if self.donate_args else (),
            static_argnums=3,
        )


@dataclass(eq=True, frozen=True)
class DataParallel(Strategy):
    axis_name: str = "device"
    donate_args: bool = False

    def from_host(self, state: S) -> S:
        key = state.key if isinstance(state, HasKey) else None
        state = jax_utils.replicate(state)
        if key is not None:
            devices = jax_utils._pmap_device_order()
            key = jax.random.split(key, jax.local_device_count())
            key = jax.device_put_sharded(list(key), devices)
            state = replace(state, key=key)
        return state

    def to_host(self, state: S) -> S:
        return jax_utils.unreplicate(state)

    def lift_batch(self, data: A) -> A:
        data = jax.tree_map(
            lambda x: einop(
                x,
                "(device batch) ... -> device batch ...",
                device=jax.local_device_count(),
            ),
            data,
        )
        return data

    def handle_gatherable(self, x: A) -> A:
        x = jax.lax.all_gather(x, axis_name=self.axis_name, tiled=True)
        return x

    def lift_key(self, key: jax.random.PRNGKeyArray) -> jax.random.PRNGKeyArray:
        return jax.random.split(key, jax.local_device_count())

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size * jax.local_device_count()

    def handle_metric(
        self,
        metric: ME,
    ) -> ME:
        # metrics = jax.lax.stop_gradient(metrics)
        metric = jax.lax.all_gather(metric, axis_name=self.axis_name)
        metric = metric.reduce()
        return metric

    def handle_grads(self, grads: A) -> A:
        return jax.lax.pmean(grads, axis_name=self.axis_name)

    def handle_batch_stats(self, batch_stats: A) -> A:
        return jax.lax.pmean(batch_stats, axis_name=self.axis_name)

    def handle_averageable(self, x: A) -> A:
        return jax.lax.pmean(x, axis_name=self.axis_name)

    def __call__(self, callback: GeneralCallback) -> GeneralCallback:
        return jax.pmap(
            callback,
            axis_name=self.axis_name,
            donate_argnums=0 if self.donate_args else (),
            in_axes=(0, 0, None, None),
            out_axes=(None, 0),
            static_broadcasted_argnums=3,
        )
