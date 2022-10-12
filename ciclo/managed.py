import importlib.util as importlib_util
import inspect
import threading
from abc import abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import jax
from flax import struct
from flax.core import tracers
from flax.training import train_state
from typing_extensions import Protocol, runtime_checkable

from ciclo.strategies import Callback, Strategy, get_strategy

if importlib_util.find_spec("clu"):
    from clu.metrics import Metric
else:
    locals()["Metric"] = type("Metric", (), {})


@runtime_checkable
class HasStrategy(Protocol):
    strategy: Strategy


@runtime_checkable
class HasBatchStats(Protocol):
    batch_stats: Any


Loss = jax.Array
State = Any
Batch = Any
Broadcasts = Any
Statics = Any
Logs = Dict[str, Any]
S = TypeVar("S", bound="ManagedState")
L = TypeVar("L", bound="LossFn")
A = TypeVar("A")
B = TypeVar("B")


class LossFn(Generic[S], Protocol):
    def __call__(self, state: S, batch: Batch, *args) -> Tuple[Loss, S]:
        ...


class StepFn(Generic[S], Protocol):
    def __call__(
        self, state: S, batch: Batch, *args
    ) -> Optional[Tuple[Optional[Logs], Optional[S]]]:
        ...


@dataclass
class _ManagedContext:
    logs: Logs
    trace_level: float
    managed_state_copy: "ManagedState"


class _Context(threading.local):
    def __init__(self) -> None:
        self.managed_stack: List[_ManagedContext] = []


_CONTEXT = _Context()


@contextmanager
def _managed_context(state: "ManagedState"):
    trace_level = tracers.trace_level(tracers.current_trace())
    context = _ManagedContext({}, trace_level, state)
    _CONTEXT.managed_stack.append(context)
    try:
        yield context
    finally:
        _CONTEXT.managed_stack.pop()


def _call_managed(
    strategy: Strategy,
    loss_fn: LossFn[S],
    state: S,
    batch: Batch,
    broadcasts: Broadcasts,
    statics: Statics,
) -> Tuple[Loss, Tuple[Logs, S]]:
    with _managed_context(state) as managed_context:
        loss, state = loss_fn(state, batch, broadcasts, statics)
        logs = {}
        for key, value in managed_context.logs.items():
            if isinstance(value, Metric):
                metric: Metric = getattr(state, key)
                metric_updates = strategy.handle_metrics(value)
                metric = metric.merge(metric_updates)
                state = state.replace(**{key: metric})
                metric_value = metric.compute()
                if isinstance(metric_value, Dict):
                    logs.update(metric_value)
                else:
                    logs[key] = metric_value
            else:
                logs[key] = value

        if "loss" not in logs:
            logs["loss"] = loss

    return loss, (logs, state)


def log(
    key: str,
    value: Any,
):
    if not _CONTEXT.managed_stack:
        raise ValueError("no log context available")

    managed_context = _CONTEXT.managed_stack[-1]
    state_copy = managed_context.managed_state_copy

    trace_level = tracers.trace_level(tracers.current_trace())
    if trace_level != managed_context.trace_level:
        raise ValueError("log must be called from the same trace level")

    if isinstance(value, Metric):
        if not hasattr(state_copy, key):
            raise ValueError(
                f"Metric field '{key}' not found in state '{type(state_copy).__name__}'"
            )
        elif not isinstance(getattr(state_copy, key), Metric):
            raise ValueError(
                f"field '{key}' is not a Metric in state '{type(state_copy).__name__}'"
            )

    managed_context.logs[key] = value


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


class Managed(Generic[S]):
    @abstractmethod
    def __call__(
        self, state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
    ) -> Any:
        ...


@dataclass
class ManagedStepBase(Managed[S]):
    strategy_callbacks: Dict[Strategy, Callback[S]]
    default_strategy: Strategy

    def __call__(
        self, state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
    ) -> Tuple[Logs, S]:

        if isinstance(state, HasStrategy):
            strategy = state.strategy
            assert isinstance(strategy, Strategy)
        else:
            strategy = self.default_strategy

        if strategy not in self.strategy_callbacks:
            self.strategy_callbacks[strategy] = strategy(self.get_callback(strategy))

        callback = self.strategy_callbacks[strategy]

        batch = strategy.lift_batch(batch)
        logs, state = callback(state, batch, broadcasts, statics)
        return logs, state

    @abstractmethod
    def get_callback(self, strategy: Strategy) -> Callback:
        ...


@dataclass
class ManagedStep(ManagedStepBase[S]):
    step_fn: Callback[S]

    def get_callback(self, strategy: Strategy) -> Callback[S]:
        return self.step_fn


@dataclass
class ManagedEvalStep(ManagedStepBase[S]):
    loss_fn: LossFn[S]

    def __call__(
        self, state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
    ) -> Tuple[Logs, S]:

        if isinstance(state, HasStrategy):
            strategy = state.strategy
            assert isinstance(strategy, Strategy)
        else:
            strategy = self.default_strategy

        if strategy not in self.strategy_callbacks:
            self.strategy_callbacks[strategy] = strategy(self.get_callback(strategy))

        callback = self.strategy_callbacks[strategy]

        batch = strategy.lift_batch(batch)
        logs, state = callback(state, batch, broadcasts, statics)
        return logs, state

    def get_callback(self, strategy: Strategy) -> Callback:
        def callback(
            state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
        ) -> Tuple[Logs, S]:

            loss, (logs, state) = _call_managed(
                strategy=strategy,
                loss_fn=self.loss_fn,
                state=state,
                batch=batch,
                broadcasts=broadcasts,
                statics=statics,
            )

            logs = strategy.handle_logs(logs)

            return logs, state

        return callback


@dataclass
class ManagedTrainStep(ManagedStepBase[S]):
    loss_fn: LossFn[S]

    def get_callback(self, strategy: Strategy) -> Callback:
        def callback(
            state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics
        ) -> Tuple[Logs, S]:
            def loss_fn(params):
                return _call_managed(
                    strategy=strategy,
                    loss_fn=self.loss_fn,
                    state=state.replace(params=params),
                    batch=batch,
                    broadcasts=broadcasts,
                    statics=statics,
                )

            (loss, (logs, state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            grads = strategy.handle_grads(grads)
            state = state.apply_gradients(grads=grads)
            if isinstance(state, HasBatchStats):
                batch_stats = strategy.handle_batch_stats(state.batch_stats)
                state = state.replace(batch_stats=batch_stats)

            logs = strategy.handle_logs(logs)

            return logs, state

        return callback


def train_step(
    loss_fn: LossFn[S],
    strategy: Union[Strategy, str] = "jit",
) -> ManagedTrainStep[S]:
    strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
    n_args = len(inspect.getfullargspec(loss_fn).args)

    if n_args < 2 or n_args > 4:
        raise ValueError(
            f"loss_fn must have 2, 3 or 4 arguments, but got {n_args} arguments"
        )

    def _loss_fn(state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics):
        args = (state, batch, broadcasts, statics)
        return loss_fn(*args[:n_args])

    return ManagedTrainStep(
        strategy_callbacks={},
        default_strategy=strategy,
        loss_fn=_loss_fn,
    )


def eval_step(
    loss_fn: LossFn[S],
    strategy: Union[Strategy, str] = "jit",
) -> ManagedEvalStep[S]:
    strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
    n_args = len(inspect.getfullargspec(loss_fn).args)

    if n_args < 2 or n_args > 4:
        raise ValueError(
            f"loss_fn must have 2, 3 or 4 arguments, but got {n_args} arguments"
        )

    def _loss_fn(state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics):
        args = (state, batch, broadcasts, statics)
        return loss_fn(*args[:n_args])

    return ManagedEvalStep(
        loss_fn=_loss_fn,
        strategy_callbacks={},
        default_strategy=strategy,
    )


def step(
    step_fn: StepFn[S],
    strategy: Union[Strategy, str] = "jit",
) -> ManagedStep[S]:
    strategy = get_strategy(strategy) if isinstance(strategy, str) else strategy
    n_args = len(inspect.getfullargspec(step_fn).args)

    if n_args < 2 or n_args > 4:
        raise ValueError(
            f"loss_fn must have 2, 3 or 4 arguments, but got {n_args} arguments"
        )

    def _step_fn(state: S, batch: Batch, broadcasts: Broadcasts, statics: Statics):
        args = (state, batch, broadcasts, statics)
        return step_fn(*args[:n_args])

    return ManagedStep(
        step_fn=_step_fn,
        strategy_callbacks={},
        default_strategy=strategy,
    )
