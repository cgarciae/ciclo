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
    Iterable,
    List,
    Mapping,
    MutableMapping,
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
LogsLike = Dict[str, Mapping[str, Any]]
InputCallable = Callable
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)

Schedule = Callable[["Elapsed"], bool]
CallbackOutput = Optional[Tuple[Optional[LogsLike], Optional[S]]]
Callback = Callable[[S, Batch, "Elapsed", "LoopState"], CallbackOutput[S]]
GeneralCallback = Callable[[S, Batch, Broadcasts, Statics], CallbackOutput[S]]
InputTasks = Dict[Schedule, Union[InputCallable, List[InputCallable]]]
ScheduleCallback = Dict[Schedule, List[Callback]]
CallbackAdapter = Callable[[Any], Callback]


class Elapsed(PyTreeNode, Mapping[str, Any]):
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

    # Mapping
    def __iter__(self) -> Iterable[str]:
        return self.__dict__.keys()

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __len__(self) -> int:
        return len(self.__dict__)


@dataclass(frozen=True)
class Period:
    steps: Union[int, None]
    samples: Union[int, None]
    time: Union[float, int, None]
    date: Union[float, None]

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


class Logs(Dict[str, Union[Dict[str, Any], Elapsed]]):
    def subkey_value(self, key: str) -> Any:
        path = self.subkey_path(key)
        if path is None:
            raise KeyError(f"Key {key} not found in logs.")
        return self.path_value(path)

    def subkey_path(self, key: str) -> Optional[Tuple[str, str]]:
        path = key.split(".")

        if len(path) == 1:
            key = path[0]
            collection = self.subkey_collection(key)
            if collection is None:
                return None
        elif len(path) == 2:
            collection, key = path
        else:
            raise ValueError(f"Got more than 2 levels of nesting in key '{key}'")

        return collection, key

    def subkey_collection(self, key: str) -> Optional[str]:
        collections = [col for col in self if key in self[col]]

        if len(collections) == 0:
            return None
        elif len(collections) == 1:
            return collections[0]
        else:
            raise ValueError(
                f"Found multiple collections for key {key}: {collections}. "
                "Use `collection.key` syntax."
            )

    def path_value(self, path: Tuple[str, str]) -> Any:
        collection, key = path
        return self[collection][key]

    def merge(self, collection_updates: LogsLike):
        for name, updates in collection_updates.items():
            if not isinstance(updates, Mapping):
                raise ValueError(
                    f"Invalide value '{updates}' for collection '{name}', value must be a Mapping"
                )
            if name in self:
                collection = self[name]
                if isinstance(collection, Dict):
                    collection.update(updates)
                elif isinstance(collection, Elapsed):
                    if not isinstance(updates, Elapsed):
                        raise ValueError(
                            f"Cannot merge Elapsed with {type(updates)} for collection '{name}'"
                        )
                    self[name] = updates
                else:
                    raise ValueError(
                        f"Cannot update non-MutableMapping collection '{name}' of type '{type(collection).__name__}'"
                    )
            else:
                if isinstance(updates, Dict):
                    self[name] = updates.copy()
                elif isinstance(updates, Elapsed):
                    self[name] = updates
                else:
                    self[name] = dict(updates)


class History(List[Logs]):
    @overload
    def collect(self, key: str) -> List[Any]:
        ...

    @overload
    def collect(self, key: str, *keys: str) -> Tuple[List[Any], ...]:
        ...

    def collect(
        self, key: str, *keys: str
    ) -> Union[Logs, List[Any], Tuple[List[Any], ...]]:
        keys = (key,) + keys
        outputs = tuple([] for _ in keys)
        for logs in self:
            paths = [logs.subkey_path(key) for key in keys]
            if all(path is not None for path in paths):
                for i, path in enumerate(paths):
                    assert path is not None
                    outputs[i].append(logs.path_value(path))

        return outputs if len(keys) > 1 else outputs[0]


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
    def __call__(
        self, *args, **kwargs
    ) -> Optional[Tuple[Optional[LogsLike], Optional[S]]]:
        ...


@dataclass(frozen=True)
class FunctionCallback(CallbackBase[S]):
    f: Callable

    def __call__(self, state: S, batch, broadcasts, statics) -> CallbackOutput[S]:
        return inject(self.f, state, batch, broadcasts, statics)


def inject(f, state: S, batch, broadcasts, statics) -> CallbackOutput[S]:
    n_args = len(inspect.getfullargspec(f).args)
    args = (state, batch, broadcasts, statics)
    return f(*args[:n_args])


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
        return adaper(f)
    elif isinstance(f, CallbackBase):
        return f
    elif callable(f):
        return f
    else:
        raise ValueError(f"Invalid callback: {f}")
