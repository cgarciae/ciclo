import functools
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import util as importlib_util
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import jax
from flax.struct import PyTreeNode
from pkbar import Kbar
from tqdm import tqdm

# ---------------------------------------
# types
# ---------------------------------------
State = Any
Batch = Any
Broadcasts = Any
Statics = Any
Logs = Dict[str, Any]
Outputs = Dict[str, Any]
LogHistory = List[Logs]
OutputHistory = List[Outputs]
InputCallable = Any
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)

Schedule = Callable[["Elapsed"], bool]
Callback = Callable[
    [S, Batch, "Elapsed", "LoopState"],
    Union[
        Tuple[Optional[Outputs], Optional[Logs], Optional[S]],
        Tuple[Optional[Logs], Optional[S]],
        None,
    ],
]
GeneralCallback = Callable[
    [S, Batch, Broadcasts, Statics],
    Union[
        Tuple[Optional[Outputs], Optional[Logs], Optional[S]],
        Tuple[Optional[Logs], Optional[S]],
        None,
    ],
]
InputTasks = Dict[Schedule, List[InputCallable]]
ScheduleCallback = Dict[Schedule, List[Callback]]
CallbackAdapter = Callable[[Any], Callback]

# ---------------------------------------
# Registry
# ---------------------------------------
_REGISTRY: Dict[Type, CallbackAdapter] = {}


def register_adapter(adapter: CallbackAdapter, cls: Type):
    if cls in _REGISTRY:
        raise ValueError(f"Adapter for {cls} already registered")

    _REGISTRY[cls] = adapter


def _get_adapter(cls: Type) -> Optional[CallbackAdapter]:
    super_classes = inspect.getmro(cls)
    for super_class in super_classes:
        if super_class in _REGISTRY:
            return _REGISTRY[super_class]
    return None


def get_callback(f: Any) -> Callback:
    adaper = _get_adapter(type(f))

    if adaper is not None:
        callback = adaper(f)
    elif callable(f):
        callback = f
    else:
        raise ValueError(f"Invalid callback: {f}")

    @functools.wraps(callable)
    def callback_wrapper(
        state: State, batch: Batch, broadcasts: Broadcasts, statics: Statics
    ):
        # maybe inject logs and history
        n_args = len(inspect.getfullargspec(callback).args)
        args = (state, batch, broadcasts, statics)
        return callback(*args[:n_args])

    return callback_wrapper


# ---------------------------------------
# elapsed
# ---------------------------------------


class Elapsed(PyTreeNode):
    steps: int
    samples: int
    date: float
    _date_start: float

    @property
    def time(self) -> float:
        return self.date - self._date_start

    @classmethod
    def create(cls, steps: int = 0, samples: int = 0) -> "Elapsed":
        now = datetime.now().timestamp()
        return cls(steps=steps, samples=samples, _date_start=now, date=now)

    def update_steps(self) -> "Elapsed":
        return self.replace(
            steps=self.steps + 1,
            date=datetime.now().timestamp(),
        )

    def update_samples(self, batch_size: int) -> "Elapsed":
        return self.replace(
            samples=self.samples + batch_size,
            date=datetime.now().timestamp(),
        )

    def update_time(self) -> "Elapsed":
        return self.replace(date=datetime.now().timestamp())


class Period:
    def __init__(
        self,
        steps: Union[int, None] = None,
        samples: Union[int, None] = None,
        time: Union[timedelta, float, int, None] = None,
        date: Union[datetime, float, None] = None,
    ):
        if all(x is None for x in [steps, samples, time, date]):
            raise ValueError("At least one duration parameter must be specified.")

        self.steps = steps
        self.samples = samples
        self.time = time.total_seconds() if isinstance(time, timedelta) else time
        self.date = date.timestamp() if isinstance(date, datetime) else date

    def __repr__(self) -> str:
        params_repr = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items() if v is not None
        )
        return f"Duration({params_repr})"

    def __ge__(self, other: Elapsed) -> bool:
        if self.steps is not None and other.steps >= self.steps:
            return True
        if self.samples is not None and other.samples >= self.samples:
            return True
        if self.time is not None and other.time >= self.time:
            return True
        if self.date is not None and other.date >= self.date:
            return True
        return False


# ---------------------------------------
# loops
# ---------------------------------------
def get_batch_size(batch: Batch) -> int:
    def get_size(sizes, x):
        sizes.add(x.shape[0])
        return sizes

    sizes = jax.tree_util.tree_reduce(get_size, batch, set())
    if len(sizes) != 1:
        raise ValueError("Batch size must be the same for all elements in the batch.")
    return sizes.pop()


@dataclass
class LoopState:
    log_history: LogHistory
    output_history: OutputHistory
    elapsed: Elapsed
    state: Optional[State] = None
    step_logs: Optional[Logs] = None
    accumulated_logs: Optional[Logs] = None


def loop(
    state: S,
    dataset,
    tasks: InputTasks,
    *,
    stop: Union[Period, int, None] = None,
    on_start: Optional[List[InputCallable]] = None,
    on_end: Optional[List[InputCallable]] = None,
    loop_state: Optional[LoopState] = None,
) -> Tuple[LoopState, S]:

    if loop_state is None:
        loop_state = LoopState(
            log_history=[],
            output_history=[],
            elapsed=Elapsed.create(),
        )

    tasks_ = {
        schedule: [get_callback(f) for f in callbacks]
        for schedule, callbacks in tasks.items()
    }

    if isinstance(stop, int):
        stop = Period(steps=stop)
    try:
        loop_state.state = state
        batch = None
        loop_state.accumulated_logs = {}

        for i, batch in enumerate(dataset):
            batch_size = get_batch_size(batch)
            loop_state.elapsed = loop_state.elapsed.update_samples(batch_size)

            # call on_start on first batch
            if i == 0 and on_start is not None:
                loop_state.step_logs = {}
                for callback in on_start:
                    callback = get_callback(callback)
                    _make_call(loop_state, callback, batch)

            loop_state.step_logs = {}
            for schedule, callbacks in tasks_.items():
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback, batch)

            loop_state.elapsed = loop_state.elapsed.update_steps()
            if loop_state.step_logs:
                loop_state.step_logs["elapsed"] = loop_state.elapsed
                loop_state.log_history.append(loop_state.step_logs)

            if stop is not None and stop >= loop_state.elapsed:
                break

        # call on_end on last batch
        if on_end is not None:
            loop_state.step_logs = {}
            for callback in on_end:
                callback = get_callback(callback)
                _make_call(loop_state, callback, batch)

        return loop_state, loop_state.state
    finally:
        loop_state.state = None
        loop_state.step_logs = None
        loop_state.accumulated_logs = None


def _make_call(loop_state: LoopState, callback: Callback, batch: Batch):
    assert loop_state.step_logs is not None
    assert loop_state.accumulated_logs is not None
    assert loop_state.elapsed is not None

    loop_state.elapsed = loop_state.elapsed.update_time()
    callback_outputs = callback(loop_state.state, batch, loop_state.elapsed, loop_state)
    if callback_outputs is not None:
        if len(callback_outputs) == 3:
            outputs, logs, state = callback_outputs
        else:
            logs, state = callback_outputs
            outputs = None
        if outputs is not None:
            loop_state.output_history.append(outputs)
        if logs is not None:
            loop_state.step_logs.update(logs)
            loop_state.accumulated_logs.update(logs)
        if state is not None:
            loop_state.state = state


# ---------------------------------------
# schedules
# ---------------------------------------


class every:
    def __init__(
        self,
        steps: Union[int, None] = None,
        samples: Union[int, None] = None,
        time: Union[timedelta, float, int, None] = None,
    ) -> None:
        self.period = Period(steps=steps, samples=samples, time=time)
        self.last_samples: int = 0
        self.last_time: float = datetime.now().timestamp()

    def __call__(self, elapsed: Elapsed) -> bool:

        if self.period.steps is not None:
            return elapsed.steps > 0 and elapsed.steps % self.period.steps == 0

        if self.period.samples is not None:
            if elapsed.samples - self.last_samples >= self.period.samples:
                self.last_samples = elapsed.samples
                return True

        if self.period.time is not None:
            if elapsed.date - self.last_time >= self.period.time:
                self.last_time = elapsed.date
                return True

        return False


# ---------------------------------------
# callbacks
# ---------------------------------------


class inner_loop:
    @overload
    def __init__(
        self,
        name_or_loop_fn: str,
        maybe_loop_fn: Callable[[State], Tuple[LoopState, State]],
        *,
        output_state: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        name_or_loop_fn: Callable[[State], Tuple[LoopState, State]],
        *,
        output_state: bool = False,
    ):
        ...

    def __init__(
        self,
        name_or_loop_fn: Union[str, Callable[[State], Tuple[LoopState, State]]],
        maybe_loop_fn: Optional[Callable[[State], Tuple[LoopState, State]]] = None,
        *,
        output_state: bool = False,
    ):
        if isinstance(name_or_loop_fn, str):
            assert maybe_loop_fn is not None
            self.name = name_or_loop_fn
            self.loop_fn = maybe_loop_fn
        else:
            assert maybe_loop_fn is None
            self.name = None
            self.loop_fn = name_or_loop_fn
        self.output_state = output_state

    def __call__(self, state, batch, elapsed, outer_loop_state):
        loop_state, state = self.loop_fn(state)
        logs = loop_state.log_history[-1] if len(loop_state.log_history) > 0 else {}
        logs = {
            k + f"_{self.name}" if self.name else k: v
            for k, v in logs.items()
            if not isinstance(v, Elapsed)
        }

        return logs, state if self.output_state else None


class tqdm_bar:
    def __init__(
        self,
        total: Union[Period, int, None] = None,
        desc=None,
        leave=True,
        file=None,
        ncols=None,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        lock_args=None,
        nrows=None,
        colour=None,
        delay=0,
        gui=False,
        **kwargs,
    ):

        if isinstance(total, int):
            total = Period(steps=total)

        if total is not None:
            if total.steps is not None:
                bar_total = total.steps
                unit = "steps"
            elif total.samples is not None:
                bar_total = total.samples
                unit = "samples"
            elif total.time is not None:
                bar_total = total.time
                unit = "s"
                unit_scale = True
            elif total.date is not None:
                total.time = total.date - datetime.now().timestamp()
                bar_total = total.time
                unit = "s"
                unit_scale = True
            else:
                raise ValueError("Invalid total")
        else:
            bar_total = None
            unit = "it"

        self.total = total
        self.prev_step: Optional[int] = None
        self.prev_samples: Optional[int] = None
        self.prev_time: Optional[float] = None
        self.bar_total = bar_total
        self.bar = tqdm(
            desc=desc,
            total=bar_total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )

    def __call__(self, state, batch, elapsed: Elapsed, loop):

        if self.total is None or self.total.steps is not None:
            if self.prev_step is None:
                self.prev_step = elapsed.steps - 1
            self.bar.update(elapsed.steps - self.prev_step)
            self.prev_step = elapsed.steps
        elif self.total.samples is not None:
            if self.prev_samples is None:
                self.prev_samples = elapsed.samples - get_batch_size(batch)
            self.bar.update(elapsed.samples - self.prev_samples)
            self.prev_samples = elapsed.samples
        elif self.total.time is not None:
            if self.prev_time is None:
                self.prev_time = elapsed._date_start
            self.bar.update(elapsed.date - self.prev_time)
            self.prev_time = elapsed.date
        else:
            raise ValueError("Invalid total")


class keras_bar:
    def __init__(
        self,
        total: Union[Period, int, None] = None,
        epoch=None,
        num_epochs=None,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        always_stateful=False,
        unit_name="step",
    ):
        if isinstance(total, int):
            total = Period(steps=total)

        if total is not None:
            if total.steps is not None:
                bar_total = total.steps
                unit_name = "step"
            elif total.samples is not None:
                bar_total = total.samples
                unit_name = "sample"
            elif total.time is not None:
                bar_total = total.time
                unit_name = "s"
                unit_scale = True
            elif total.date is not None:
                total.time = total.date - datetime.now().timestamp()
                bar_total = total.time
                unit_name = "s"
                unit_scale = True
            else:
                raise ValueError("Invalid total")
        else:
            bar_total = None
            unit_name = "it"

        self.total = total
        self.prev_step: Optional[int] = None
        self.prev_samples: Optional[int] = None
        self.prev_time: Optional[float] = None
        self.bar_total = bar_total
        self.bar = Kbar(
            bar_total,
            epoch=epoch,
            num_epochs=num_epochs,
            width=width,
            verbose=verbose,
            interval=interval,
            stateful_metrics=stateful_metrics,
            always_stateful=always_stateful,
            unit_name=unit_name,
        )

    def __call__(self, state, batch, elapsed: Elapsed, loop_state: LoopState):
        if self.total is None or self.total.steps is not None:
            update = elapsed.steps
        elif self.total.samples is not None:
            update = elapsed.samples
        elif self.total.time is not None:
            update = elapsed.time
        else:
            raise ValueError("Invalid total")

        self.bar.update(
            update,
            values=[
                (k, v)
                for k, v in loop_state.step_logs.items()
                if not isinstance(v, Elapsed)
            ],
        )


# ---------------------------------------
# utils
# ---------------------------------------


def at(
    steps: Optional[int] = None,
    samples: Optional[int] = None,
    time: Optional[float] = None,
    date: Optional[float] = None,
) -> Period:
    return Period(steps=steps, samples=samples, time=time, date=date)


# -------------------------------------------
# Adapters
# -------------------------------------------

if importlib_util.find_spec("clu") is not None:
    from clu.periodic_actions import PeriodicAction

    @functools.partial(register_adapter, cls=PeriodicAction)
    def periodic_action_adapter(f: PeriodicAction):
        @functools.wraps(f)
        def callback(
            state: State, batch: Batch, elapsed: Elapsed, loop_state: LoopState
        ):
            f(elapsed.steps, t=elapsed.date)

        return callback
