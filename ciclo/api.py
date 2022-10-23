from abc import ABC, abstractmethod
import functools
import inspect
from dataclasses import dataclass
from datetime import datetime, timedelta
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

import ciclo

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
InputCallable = Callable
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)

Schedule = Callable[["Elapsed"], bool]
CallbackOutput = Optional[Tuple[Optional[Logs], Optional[S]]]
Callback = Callable[[S, Batch, "Elapsed", "LoopState"], CallbackOutput[S]]
GeneralCallback = Callable[[S, Batch, Broadcasts, Statics], CallbackOutput[S]]
InputTasks = Dict[Schedule, Union[InputCallable, List[InputCallable]]]
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

    def _compare(
        self, other: "Period", predicate: Callable[[float, float], bool]
    ) -> bool:
        if other.steps is not None and predicate(self.steps, other.steps):
            return True
        if other.samples is not None and predicate(self.samples, other.samples):
            return True
        if other.time is not None and predicate(self.time, other.time):
            return True
        if other.date is not None and predicate(self.date, other.date):
            return True
        return False

    def __ge__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a >= b)

    def __gt__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a > b)

    def __le__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a <= b)

    def __lt__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a < b)

    def __eq__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other: "Period") -> bool:
        return self._compare(other, lambda a, b: a != b)


@dataclass(frozen=True)
class Period:
    steps: Union[int, None]
    samples: Union[int, None]
    time: Union[timedelta, float, int, None]
    date: Union[datetime, float, None]

    @classmethod
    def create(
        cls,
        steps: Union[int, None] = None,
        samples: Union[int, None] = None,
        time: Union[timedelta, float, int, None] = None,
        date: Union[datetime, float, None] = None,
    ):
        if all(x is None for x in [steps, samples, time, date]):
            raise ValueError("At least one duration parameter must be specified.")

        return cls(
            steps=steps,
            samples=samples,
            time=time.total_seconds() if isinstance(time, timedelta) else time,
            date=date.timestamp() if isinstance(date, datetime) else date,
        )

    def __repr__(self) -> str:
        params_repr = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items() if v is not None
        )
        return f"Period({params_repr})"


class History(List[Logs]):
    @overload
    def __getitem__(self, index: int) -> Logs:
        ...

    @overload
    def __getitem__(self, index: str) -> List[Any]:
        ...

    @overload
    def __getitem__(self, index: Tuple[str, ...]) -> Tuple[List[Any], ...]:
        ...

    def __getitem__(
        self, index: Union[int, str, Tuple[str, ...]]
    ) -> Union[Logs, List[Any], Tuple[List[Any], ...]]:
        if isinstance(index, int):
            return super().__getitem__(index)

        if isinstance(index, str):
            keys = (index,)
        else:
            keys = index

        outputs = tuple([] for _ in keys)
        for logs in self:
            if all(self._has_key(logs, key) for key in keys):
                for i, key in enumerate(keys):
                    outputs[i].append(self._get_key(logs, key))

        return outputs if len(keys) > 1 else outputs[0]

    @staticmethod
    def _has_key(logs: Logs, key: str) -> Any:
        return key in logs or ("elapsed" in logs and hasattr(logs["elapsed"], key))

    @staticmethod
    def _get_key(logs: Logs, key: str) -> Any:
        if key in logs:
            return logs[key]
        elif "elapsed" in logs and hasattr(logs["elapsed"], key):
            return getattr(logs["elapsed"], key)
        else:
            raise KeyError(f"Key {key} not found in logs.")


@dataclass
class LoopState(Generic[S]):
    state: S
    history: History
    elapsed: Elapsed
    step_logs: Logs
    accumulated_logs: Logs
    metadata: Any
    stop_iteration: bool = False


LoopOutput = Tuple[S, History, Elapsed]


class CallbackBase(ABC, Generic[S]):
    def keys(self) -> List[Schedule]:
        return [ciclo.every(1)]

    def __getitem__(self, key: Schedule) -> List[Callback]:
        return [self]

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Optional[Tuple[Optional[Logs], Optional[S]]]:
        ...


@dataclass(frozen=True)
class FunctionCallback(CallbackBase[S]):
    f: Callable

    def __call__(
        self, state: S, batch, broadcasts, statics
    ) -> Optional[Tuple[Optional[Logs], Optional[S]]]:
        n_args = len(inspect.getfullargspec(self.f).args)
        args = (state, batch, broadcasts, statics)
        return self.f(*args[:n_args])


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
        _callback = adaper(f)
    elif isinstance(f, CallbackBase):
        return f
    elif callable(f):
        _callback = FunctionCallback(f)
    else:
        raise ValueError(f"Invalid callback: {f}")

    return _callback
