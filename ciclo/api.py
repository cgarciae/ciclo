import functools
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta
from importlib import util as importlib_util
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    NamedTuple,
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
OutputHistory = List[Outputs]
InputCallable = Any
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)

Schedule = Callable[["Elapsed"], bool]
Callback = Callable[
    [S, Batch, "Elapsed", "LoopState"],
    Optional[Tuple[Optional[Logs], Optional[S]]],
]
GeneralCallback = Callable[
    [S, Batch, Broadcasts, Statics],
    Optional[Tuple[Optional[Logs], Optional[S]]],
]
InputTasks = Dict[Schedule, List[InputCallable]]
ScheduleCallback = Dict[Schedule, List[Callback]]
CallbackAdapter = Callable[[Any], Callback]


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

    def update(self, batch_size: int) -> "Elapsed":
        return self.replace(
            steps=self.steps + 1,
            samples=self.samples + batch_size,
            date=datetime.now().timestamp(),
        )

    def update_time(self) -> "Elapsed":
        return self.replace(date=datetime.now().timestamp())

    def __sub__(self, other: "Elapsed") -> "Elapsed":
        return Elapsed(
            steps=self.steps - other.steps,
            samples=self.samples - other.samples,
            date=self.date,
            _date_start=other._date_start,
        )


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


class History(List[Logs]):
    def get_all(self, key: str) -> List[Tuple[Elapsed, Any]]:
        return [(logs["elapsed"], logs[key]) for logs in self if key in logs]


@dataclass
class LoopState(Generic[S]):
    state: S
    history: History
    elapsed: Elapsed
    step_logs: Logs
    accumulated_logs: Logs
    stop_iteration: bool = False


LoopOutput = Tuple[S, History, Elapsed]


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
# utils
# ---------------------------------------


def at(
    steps: Optional[int] = None,
    samples: Optional[int] = None,
    time: Optional[float] = None,
    date: Optional[float] = None,
) -> Period:
    return Period(steps=steps, samples=samples, time=time, date=date)


def get_batch_size(batch: Batch) -> int:
    def get_size(sizes, x):
        sizes.add(x.shape[0])
        return sizes

    sizes = jax.tree_util.tree_reduce(get_size, batch, set())
    if len(sizes) != 1:
        raise ValueError("Batch size must be the same for all elements in the batch.")
    return sizes.pop()


def is_scalar(x):
    if isinstance(x, (int, float, bool)):
        return True
    elif hasattr(x, "shape"):
        return x.shape == () or x.shape == (1,)
    else:
        return False


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
