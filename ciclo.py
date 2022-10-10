from dataclasses import dataclass, field
from datetime import datetime, timedelta
import functools
from re import U
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)
import inspect
from clu.periodic_actions import PeriodicAction
from pkbar import Kbar
from tqdm import tqdm
from flax.struct import PyTreeNode
import jax


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


State = Any
Batch = Any
Step = int
Logs = Dict[str, Any]
LogHistory = List[Logs]

Schedule = Callable[[Elapsed], bool]
Callback = Callable[
    [State, Batch, Elapsed, "Loop"], Optional[Tuple[Optional[Logs], Optional[State]]]
]
InputCallable = Union[Callable, PeriodicAction]
ScheduleCallbacks = Dict[Schedule, Union[InputCallable, List[InputCallable]]]


def create_callback(f: InputCallable) -> Callback:

    if isinstance(f, PeriodicAction):

        @functools.wraps(f)
        def wrapper(state: State, batch: Batch, elapsed: Elapsed, loop: Loop):
            f(elapsed.steps)

    else:
        sig = inspect.signature(f)
        params = sig.parameters

        @functools.wraps(f)
        def wrapper(state: State, batch: Batch, elapsed: Elapsed, loop: Loop):
            # maybe inject logs and history
            args = [state, batch, elapsed]
            kwargs = {}
            if "loop" in params:
                kwargs["loop"] = loop

            return f(*args, **kwargs)

    return wrapper


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


class Loop:
    def __init__(
        self,
        initial_steps: int = 0,
        initial_samples: int = 0,
        history: Optional[LogHistory] = None,
    ):
        self.state = None
        self.initial_steps = initial_steps
        self.initial_samples = initial_samples
        self.logs: Optional[Logs] = None
        self.history = history or []

    def run(
        self,
        state: State,
        dataset,
        schedule_callbacks: ScheduleCallbacks,
        *,
        stop: Union[Period, int, None] = None,
    ) -> State:

        elapsed = Elapsed.create(steps=self.initial_steps, samples=self.initial_samples)

        if isinstance(stop, int):
            stop = Period(steps=stop)
        try:
            self.state = state
            schedule_callbacks_: Dict[Schedule, List[Callback]] = {
                schedule: [
                    create_callback(callback)
                    for callback in (
                        callbacks if isinstance(callbacks, list) else [callbacks]
                    )
                ]
                for schedule, callbacks in schedule_callbacks.items()
            }

            for i, batch in enumerate(dataset):
                batch_size = get_batch_size(batch)
                elapsed = elapsed.update_samples(batch_size)

                self.logs = {}

                for schedule, callbacks in schedule_callbacks_.items():
                    if schedule(elapsed):
                        for callback in callbacks:
                            elapsed = elapsed.update_time()
                            output = callback(self.state, batch, elapsed, self)
                            if output is not None:
                                out_logs, out_state = output
                                if out_state is not None:
                                    self.state = out_state
                                if out_logs is not None:
                                    self.logs.update(out_logs)

                elapsed = elapsed.update_steps()
                if self.logs:
                    self.logs["elapsed"] = elapsed
                    self.history.append(self.logs)

                if stop is not None and stop >= elapsed:
                    break

            return self.state
        finally:
            self.state = None
            self.logs = None


def loop(
    state: State,
    dataset,
    schedule_callbacks: ScheduleCallbacks,
    *,
    stop: Union[Period, int, None] = None,
    initial_steps: int = 0,
    initial_samples: int = 0,
    history: Optional[LogHistory] = None,
) -> Tuple[State, Loop]:
    loop = Loop(
        initial_steps=initial_steps, initial_samples=initial_samples, history=history
    )
    state = loop.run(state, dataset, schedule_callbacks, stop=stop)
    return state, loop


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
        maybe_loop_fn: Callable[[State], Tuple[State, Loop]],
        *,
        reset_fn: Optional[Callable[[State], State]] = None,
    ):
        ...

    @overload
    def __init__(
        self,
        name_or_loop_fn: Callable[[State], Tuple[State, Loop]],
        *,
        reset_fn: Optional[Callable[[State], State]] = None,
    ):
        ...

    def __init__(
        self,
        name_or_loop_fn: Union[str, Callable[[State], Tuple[State, Loop]]],
        maybe_loop_fn: Optional[Callable[[State], Tuple[State, Loop]]] = None,
        *,
        reset_fn: Optional[Callable[[State], State]] = None,
    ):
        if isinstance(name_or_loop_fn, str):
            assert maybe_loop_fn is not None
            self.name = name_or_loop_fn
            self.loop_fn = maybe_loop_fn
        else:
            assert maybe_loop_fn is None
            self.name = None
            self.loop_fn = name_or_loop_fn
        self.reset_fn = reset_fn

    def __call__(self, state, batch, elapsed: Elapsed, loop):
        if self.reset_fn is not None:
            state = self.reset_fn(state)
        state, inner_loop = self.loop_fn(state)
        logs = inner_loop.history[-1]
        logs = {
            k + f"_{self.name}" if self.name else k: v
            for k, v in logs.items()
            if not isinstance(v, Elapsed)
        }
        if self.reset_fn is not None:
            state = self.reset_fn(state)
        return logs, state


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
                unit_name = "steps"
            elif total.samples is not None:
                bar_total = total.samples
                unit_name = "samples"
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

    def __call__(self, state, batch, elapsed: Elapsed, loop: Loop):
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
            values=[(k, v) for k, v in loop.logs.items() if not isinstance(v, Elapsed)],
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
