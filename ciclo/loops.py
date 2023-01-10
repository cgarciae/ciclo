import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from typing_extensions import Protocol, runtime_checkable

import ciclo
from ciclo.logging import History, Logs
from ciclo.timetracking import Elapsed, Period
from ciclo.types import (
    A,
    B,
    Batch,
    Broadcasts,
    InputCallback,
    LogsLike,
    S,
    Schedule,
    Statics,
)

CallbackOutput = Tuple[LogsLike, S]


@runtime_checkable
class LoopCallback(Protocol, Generic[S]):
    def __loop_callback__(self, loop_state: "LoopState[S]") -> CallbackOutput[S]:
        ...


InputTasks = Dict[Schedule, Union[InputCallback, List[InputCallback]]]
ScheduleCallback = Dict[Schedule, List[LoopCallback[S]]]
CallbackAdapter = Callable[[Any], LoopCallback[S]]

FunctionCallbackOutputs = Union[
    Tuple[Optional[LogsLike], Optional[S]], LogsLike, S, None
]
GeneralCallback = Callable[[S, Batch, Broadcasts, Statics], FunctionCallbackOutputs[S]]


@dataclass
class LoopState(Generic[S]):
    state: S
    batch: Batch
    history: History
    elapsed: Elapsed
    logs: Logs
    accumulated_logs: Logs
    metadata: Any
    stop_iteration: bool = False


LoopOutput = Tuple[S, History, Elapsed]


def to_standard_outputs(
    outputs: FunctionCallbackOutputs[S], current_state: S
) -> CallbackOutput[S]:
    logs: LogsLike
    state: S
    if outputs is None:
        logs = {}
        state = current_state

    elif type(outputs) is tuple:
        if len(outputs) != 2:
            raise ValueError(
                f"Invalid output from callback function: {outputs}, must be a tuple of length 2"
            )
        logs = outputs[0] or {}
        state = outputs[1] or current_state
    elif isinstance(outputs, Dict):
        logs = outputs
        state = current_state
    else:
        logs = {}
        state = outputs

    return logs, state


class LoopElement:
    def keys(self) -> List[Schedule]:
        return [ciclo.every(1)]

    def __getitem__(self: A, key: Schedule) -> List[A]:
        return [self]


class LoopCallbackBase(LoopCallback[S], LoopElement):
    @abstractmethod
    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        ...


@dataclass(frozen=True)
class LoopFunctionCallback(LoopCallbackBase[S]):
    f: Callable[..., FunctionCallbackOutputs[S]]

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        outputs = ciclo.inject(self.f)(
            loop_state.state, loop_state.batch, loop_state.elapsed, loop_state
        )
        return to_standard_outputs(outputs, loop_state.state)


def loop(
    state: S,
    dataset: Iterable[B],
    tasks: InputTasks,
    *,
    stop: Union[Period, int, None] = None,
    on_start: Union[InputCallback, List[InputCallback], None] = None,
    on_end: Union[InputCallback, List[InputCallback], None] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
) -> LoopOutput[S]:
    if isinstance(stop, int):
        stop_period = Period.create(steps=stop)
    else:
        stop_period = stop

    if history is None:
        history = History()

    loop_state = LoopState(
        state=state,
        batch=None,
        history=history,
        elapsed=elapsed or Elapsed.create(),
        logs=Logs(),
        accumulated_logs=Logs(),
        metadata=metadata,
    )

    if on_start is None:
        on_start = []
    elif not isinstance(on_start, list):
        on_start = [on_start]

    if on_end is None:
        on_end = []
    elif not isinstance(on_end, list):
        on_end = [on_end]

    tasks_ = [
        (
            schedule,
            [
                get_loop_callback(f)
                for f in (callbacks if isinstance(callbacks, list) else [callbacks])
            ],
        )
        for schedule, callbacks in tasks.items()
    ]

    try:
        for i, (elapsed, batch) in enumerate(ciclo.elapse(dataset, initial=elapsed)):
            loop_state.elapsed = elapsed
            loop_state.batch = batch

            # call on_start on first batch
            if i == 0:
                loop_state.logs = Logs()
                for callback in on_start:
                    callback = get_loop_callback(callback)
                    _make_call(loop_state, callback)

            loop_state.logs = Logs()
            for i, (schedule, callbacks) in enumerate(tasks_):
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback)
                        if loop_state.stop_iteration:
                            break
                if loop_state.stop_iteration:
                    break

            if loop_state.logs:
                loop_state.history.commit(elapsed, loop_state.logs)

            if loop_state.stop_iteration or (
                stop_period and loop_state.elapsed >= stop_period
            ):
                break

        # call on_end on last batch
        loop_state.logs = Logs()
        for callback in on_end:
            callback = get_loop_callback(callback)
            _make_call(loop_state, callback)

    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            print("\nStopping loop...")
        else:
            raise

    return loop_state.state, loop_state.history, loop_state.elapsed


def _make_call(loop_state: LoopState[S], callback: LoopCallback[S]):
    try:
        loop_state.elapsed = loop_state.elapsed.update_time()
        logs, state = callback.__loop_callback__(loop_state)
        loop_state.logs.merge(logs)
        loop_state.accumulated_logs.merge(logs)
        loop_state.state = state
    except BaseException as e:
        raise type(e)(f"Error in callback {callback}: {e}") from e


# ---------------------------------------
# Registry
# ---------------------------------------
_REGISTRY: Dict[Type, CallbackAdapter] = {}


def register_adapter(adapter: CallbackAdapter, cls: Type):
    if cls in _REGISTRY:
        raise ValueError(f"Adapter for {cls} already registered")

    _REGISTRY[cls] = adapter


def get_adapter(cls: Type) -> Optional[CallbackAdapter]:
    super_classes = inspect.getmro(cls)
    for super_class in super_classes:
        if super_class in _REGISTRY:
            return _REGISTRY[super_class]
    return None


def get_loop_callback(f: Any) -> LoopCallback:
    adaper = get_adapter(type(f))

    if adaper is not None:
        return adaper(f)
    elif isinstance(f, LoopCallbackBase):
        return f
    elif callable(f):
        return LoopFunctionCallback(f)
    else:
        raise ValueError(f"Invalid callback: {f}")
