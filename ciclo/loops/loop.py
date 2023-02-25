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
from ciclo.schedules import ScheduleLike, to_schedule
from ciclo.timetracking import Elapsed, PeriodLike, elapse
from ciclo.types import A, B, Batch, Broadcasts, S, Schedule, Statics

CallbackOutput = Tuple[Logs, S]


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


@runtime_checkable
class LoopCallback(Protocol, Generic[S]):
    def __loop_callback__(self, loop_state: "LoopState[S]") -> CallbackOutput[S]:
        ...


class LoopElement:
    def keys(self) -> List[Schedule]:
        return [ciclo.every(1)]

    def __getitem__(self: A, key: Schedule) -> List[A]:
        return [self]


class LoopCallbackBase(LoopCallback[S], LoopElement):
    @abstractmethod
    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        ...


LoopCallbackLike = Any
InputTasks = Dict[ScheduleLike, Union[LoopCallbackLike, List[LoopCallbackLike]]]
ScheduleCallback = Dict[Schedule, List[LoopCallback[S]]]
CallbackAdapter = Callable[[Any], LoopCallback[S]]
FunctionCallbackOutputs = Union[Tuple[Optional[Logs], Optional[S]], Logs, S, None]
GeneralCallback = Callable[[S, Batch, Broadcasts, Statics], FunctionCallbackOutputs[S]]
LoopOutput = Tuple[S, History, Elapsed]


def to_standard_outputs(
    outputs: FunctionCallbackOutputs[S], current_state: S
) -> CallbackOutput[S]:
    logs: Logs
    state: S
    if outputs is None:
        logs = Logs()
        state = current_state
    elif type(outputs) is tuple:
        if len(outputs) != 2:
            raise ValueError(
                f"Invalid output from callback function: {outputs}, must be a tuple of length 2"
            )
        logs = outputs[0] or Logs()
        state = outputs[1] or current_state
    elif isinstance(outputs, Logs):
        logs = outputs
        state = current_state
    else:
        logs = Logs()
        state = outputs

    return logs, state


@dataclass(frozen=True)
class LoopFunctionCallback(LoopCallbackBase[S]):
    f: Callable[..., FunctionCallbackOutputs[S]]

    def __loop_callback__(self, loop_state: LoopState[S]) -> CallbackOutput[S]:
        outputs = ciclo.inject(self.f)(
            loop_state.state, loop_state.batch, loop_state.elapsed, loop_state
        )
        return to_standard_outputs(outputs, loop_state.state)


def _make_call(loop_state: LoopState[S], callback: LoopCallback[S]):
    # try:
    loop_state.elapsed = loop_state.elapsed.update_time()
    logs, state = callback.__loop_callback__(loop_state)
    loop_state.logs.merge(logs)
    loop_state.accumulated_logs.merge(logs)
    loop_state.state = state
    # except BaseException as e:
    #     raise type(e)(f"Error in callback {callback}: {e}") from e


# -------------------------------------
# loops
# -------------------------------------


def loop(
    state: S,
    dataset: Iterable[B],
    tasks: InputTasks,
    *,
    stop: Optional[PeriodLike] = None,
    on_start: Union[LoopCallbackLike, List[LoopCallbackLike], None] = None,
    on_end: Union[LoopCallbackLike, List[LoopCallbackLike], None] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    batch_size_fn: Optional[Callable[[List[Tuple[int, ...]]], int]] = None,
    metadata: Optional[Any] = None,
) -> LoopOutput[S]:
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

    schedule_callbacks = [
        (
            to_schedule(schedule),
            [
                get_loop_callback(f)
                for f in (callbacks if isinstance(callbacks, list) else [callbacks])
            ],
        )
        for schedule, callbacks in tasks.items()
    ]
    # prone empty tasks
    schedule_callbacks = [x for x in schedule_callbacks if len(x[1]) > 0]

    try:
        for i, (elapsed, batch) in enumerate(
            elapse(dataset, initial=elapsed, stop=stop, batch_size_fn=batch_size_fn)
        ):
            loop_state.elapsed = elapsed
            loop_state.batch = batch

            # call on_start on first batch
            if i == 0:
                loop_state.logs = Logs()
                for callback in on_start:
                    callback = get_loop_callback(callback)
                    _make_call(loop_state, callback)

            loop_state.logs = Logs()
            for i, (schedule, callbacks) in enumerate(schedule_callbacks):
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback)
                        if loop_state.stop_iteration:
                            break
                if loop_state.stop_iteration:
                    break

            if loop_state.logs:
                loop_state.history.commit(elapsed, loop_state.logs)

            if loop_state.stop_iteration:
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
