from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, TypeVar, Union, overload

import jax
import jax.numpy as jnp
from einop import einop
from flax import jax_utils
from typing_extensions import Protocol, runtime_checkable

from ciclo.loops.loop import GeneralCallback
from ciclo.types import CluMetric, MetricLike


class Dataclass(Protocol):
    __dataclass_fields__: Dict


@runtime_checkable
class HasKey(Protocol):
    key: jax.random.KeyArray


StrategyConstructor = Callable[[], "Strategy"]
A = TypeVar("A")
S = TypeVar("S", bound=Dataclass)
Metric = Any


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

    def lift_key(self, key: jax.random.KeyArray) -> jax.random.KeyArray:
        return key

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size

    def handle_metric(self, metric: Metric) -> Metric:
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

    def lower_tileable(self, logs: A) -> A:
        return jax.device_get(logs)

    def lower_sharded(self, logs: A) -> A:
        return jax.device_get(logs)

    def lower_averageable(self, logs: A) -> A:
        return jax.device_get(logs)

    def lower_replicated(self, logs: A) -> A:
        return jax.device_get(logs)

    @abstractmethod
    def __call__(self, callback: GeneralCallback[S]) -> GeneralCallback[S]:
        ...


@dataclass(eq=True, frozen=True)
class Eager(Strategy):
    def __call__(self, callback: GeneralCallback[S]) -> GeneralCallback[S]:
        return callback


@dataclass(eq=True, frozen=True)
class JIT(Strategy):
    donate_args: bool = False

    def __call__(self, callback: GeneralCallback[S]) -> GeneralCallback[S]:
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

    def lift_key(self, key: jax.random.KeyArray) -> jax.random.KeyArray:
        return jax.random.split(key, jax.local_device_count())

    def lift_batch_size(self, batch_size: int) -> int:
        return batch_size * jax.local_device_count()

    def handle_metric(
        self,
        metric: Metric,
    ) -> Metric:
        # metrics = jax.lax.stop_gradient(metrics)
        metric = jax.lax.all_gather(metric, axis_name=self.axis_name)
        if isinstance(metric, (CluMetric, MetricLike)):
            metric = metric.reduce()
        else:
            raise ValueError(f"Unknown metric type {type(metric)}")
        return metric

    def handle_grads(self, grads: A) -> A:
        return jax.lax.pmean(grads, axis_name=self.axis_name)

    def handle_batch_stats(self, batch_stats: A) -> A:
        return jax.lax.pmean(batch_stats, axis_name=self.axis_name)

    def lower_averageable(self, logs: A) -> A:
        return jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), logs)

    def lower_tileable(self, logs: A) -> A:
        return jax.tree_util.tree_map(
            lambda x: einop(x, "device batch ... -> (device batch) ..."), logs
        )

    def lower_sharded(self, logs: A) -> A:
        return jax.device_get(logs)

    def lower_replicated(self, logs: A) -> A:
        return jax.tree_util.tree_map(lambda x: x[0], logs)

    def __call__(self, callback: GeneralCallback[S]) -> GeneralCallback[S]:
        return jax.pmap(
            callback,
            axis_name=self.axis_name,
            donate_argnums=0 if self.donate_args else (),
            in_axes=(0, 0, None, None),
            out_axes=(0, 0),
            static_broadcasted_argnums=3,
        )
