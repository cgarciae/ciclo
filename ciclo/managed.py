from functools import partial
import functools
import importlib.util
import inspect
import threading
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import jax
from flax import struct
from flax.core import tracers
from flax.training import train_state
from typing_extensions import Protocol, runtime_checkable
from ciclo.api import (
    Broadcasts,
    Callback,
    CallbackBase,
    CallbackOutput,
    Elapsed,
    LogsLike,
    State,
    Statics,
    inject,
    register_adapter,
    Batch,
    Metric,
)

from ciclo.strategies import GeneralCallback, Strategy, get_strategy


@runtime_checkable
class HasStrategy(Protocol):
    strategy: Strategy


@runtime_checkable
class HasBatchStats(Protocol):
    batch_stats: Any


Loss = jax.Array

S = TypeVar("S", bound="ManagedState")
L = TypeVar("L", bound="LossFn")
A = TypeVar("A")
B = TypeVar("B")


class ManagedState(train_state.TrainState):
    """
    A train state that manages the strategy.
    """

    strategy: "Strategy" = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls, *, apply_fn, params, tx, strategy: Union[Strategy, str] = "jit", **kwargs
    ) -> "ManagedState":
        state = super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            strategy=get_strategy("eager"),
            **kwargs,
        )
        return state.with_strategy(strategy)

    def with_strategy(self, strategy: Union[Strategy, str]) -> "ManagedState":
        new_strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
        current_strategy = self.strategy
        if new_strategy == current_strategy:
            return self
        state = current_strategy.to_host(self)
        state = new_strategy.from_host(state)
        return state.replace(strategy=new_strategy)


@dataclass
class ManagedStep(CallbackBase[S]):
    strategy_callbacks: Dict[Strategy, GeneralCallback[S]]
    default_strategy: Strategy
    step_fn: GeneralCallback[S]

    def __call__(
        self, state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
    ) -> CallbackOutput[S]:

        if isinstance(state, HasStrategy):
            strategy = state.strategy
            assert isinstance(strategy, Strategy)
        else:
            strategy = self.default_strategy

        if strategy not in self.strategy_callbacks:
            self.strategy_callbacks[strategy] = strategy(self.get_callback(strategy))

        callback = self.strategy_callbacks[strategy]

        batch = strategy.lift_batch(batch)
        return callback(state, batch, broadcasts, statics)

    def get_callback(self, strategy: Strategy) -> GeneralCallback:
        def lifted_postprocess(
            state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
        ) -> CallbackOutput[S]:
            step_fn = self.step_callback(strategy)
            step_output = step_fn(state, batch, broadcasts, statics)

            if step_output is None:
                return

            logs = step_output[0]
            state = step_output[1] if step_output[1] is not None else state

            if logs is None:
                return logs, state

            for collection in logs.keys():
                for key, value in logs[collection].items():
                    if collection == "stateful_metrics":
                        if isinstance(value, Metric):
                            metric: Metric = getattr(state, key)
                            value = strategy.handle_metric(value)
                            metric = metric.merge(value)
                            state = state.replace(**{key: metric})
                            metric_value = metric.compute()
                            if isinstance(metric_value, Mapping):
                                logs[collection].update(metric_value)
                            else:
                                logs[collection][key] = metric_value
                        else:
                            logs[collection][key] = strategy.handle_averageable(value)
                    elif collection in ("losses", "metrics"):
                        logs[collection][key] = strategy.handle_averageable(value)
                    else:
                        logs[collection][key] = strategy.handle_gatherable(value)
            return logs, state

        return lifted_postprocess

    def step_callback(self, strategy: Strategy) -> GeneralCallback:
        def regular_step_callback(
            state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
        ) -> CallbackOutput[S]:
            return inject(self.step_fn, state, batch, broadcasts, statics)

        return regular_step_callback


@dataclass
class ManagedTrainStep(ManagedStep[S]):
    def step_callback(self, strategy: Strategy) -> GeneralCallback[S]:
        def train_step_callback(
            state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
        ) -> CallbackOutput[S]:
            def loss_fn(params):
                _state = state.replace(params=params)
                step_output = inject(self.step_fn, _state, batch, broadcasts, statics)

                if step_output is None:
                    raise ValueError(
                        "callback must return a (logs, state), but got None"
                    )

                logs = step_output[0]
                _state = step_output[1] if step_output[1] is not None else _state

                if logs is None:
                    raise ValueError(
                        "callback must return a logs dictionary, but got None"
                    )

                if "losses" not in logs:
                    raise ValueError(
                        f"callback must return dictorionary with a 'losses' key, but got {logs.keys()}"
                    )

                if len(logs["losses"]) == 0:
                    raise ValueError(
                        "'losses' collection is empty, you must provide at least one entry "
                        "in the 'losses' collection"
                    )

                loss = 0.0
                for k, v in logs["losses"].items():
                    if v.shape != ():
                        raise ValueError(
                            f"Loss {k} should be a scalar, but has shape {v.shape}"
                        )
                    loss += v

                return loss, (logs, _state)

            (_, (logs, state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            grads = strategy.handle_grads(grads)
            state = state.apply_gradients(grads=grads)

            if isinstance(state, HasBatchStats):
                batch_stats = strategy.handle_batch_stats(state.batch_stats)
                state = state.replace(batch_stats=batch_stats)

            return logs, state

        return train_step_callback


def train_step(
    step_fn: Callable[..., CallbackOutput[S]],
    strategy: Union[Strategy, str] = "jit",
) -> ManagedTrainStep[S]:
    strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
    return ManagedTrainStep(
        strategy_callbacks={},
        default_strategy=strategy,
        step_fn=step_fn,
    )


def step(
    step_fn: Callable[..., CallbackOutput[S]],
    strategy: Union[Strategy, str] = "jit",
) -> ManagedStep[S]:
    strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
    return ManagedStep(
        step_fn=step_fn,
        strategy_callbacks={},
        default_strategy=strategy,
    )


@partial(register_adapter, cls=ManagedStep)
def managed_adapter(f: ManagedStep):
    @functools.wraps(f)
    def callback(state: State, batch: Batch, elapsed: Elapsed, loop: Any):
        return f(state, batch, elapsed, None)

    return callback
