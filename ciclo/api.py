import importlib.util
import inspect
from abc import abstractmethod
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
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.tree_util import register_pytree_node
from typing_extensions import Protocol, runtime_checkable

import ciclo

if importlib.util.find_spec("clu"):
    from clu.metrics import Metric
else:
    locals()["Metric"] = type("Metric", (), {})

# ---------------------------------------
# types
# ---------------------------------------
State = Any
Batch = Any
Broadcasts = Any
Statics = Any
LogPath = Tuple[str, str]
LogsLike = Dict[str, Mapping[str, Any]]
InputCallback = Any
A = TypeVar("A")
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)
Schedule = Callable[["Elapsed"], bool]
CallbackOutput = Tuple[LogsLike, S]


@runtime_checkable
class LoopCallback(Protocol, Generic[S]):
    def __loop_callback__(self, loop_state: "LoopState[S]") -> CallbackOutput[S]:
        ...


FunctionCallbackOutputs = Union[
    Tuple[Optional[LogsLike], Optional[S]], LogsLike, S, None
]
GeneralCallback = Callable[[S, Batch, Broadcasts, Statics], FunctionCallbackOutputs[S]]
InputTasks = Dict[Schedule, Union[InputCallback, List[InputCallback]]]
ScheduleCallback = Dict[Schedule, List[LoopCallback[S]]]
CallbackAdapter = Callable[[Any], LoopCallback[S]]


class Elapsed(struct.PyTreeNode, Mapping[str, Any]):
    steps: int
    samples: int
    date: float
    _date_start: float = struct.field(pytree_node=True, repr=False)

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


class Logs(LogsLike):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # copy mutable values
        for k, v in self.items():
            if isinstance(v, MutableMapping):
                self[k] = dict(v)

    @property
    def updates(self) -> Optional[LogsLike]:
        # raise error if accessed
        raise AttributeError("updates is a write-only attribute")

    @updates.setter
    def updates(self, updates: Optional[LogsLike]) -> None:
        if updates is not None:
            self.merge(updates)

    # ----------------------------------
    # logger behavior
    # ----------------------------------
    def add_entry(self, collection: str, name: str, value: Any) -> "Logs":

        if collection not in self:
            self[collection] = {}

        mapping = self[collection]

        if not isinstance(mapping, MutableMapping):
            raise ValueError(
                f"Invalid collection '{collection}' of type '{type(mapping).__name__}', must be a MutableMapping"
            )

        mapping[name] = value

        return self

    def add_entries(self, collection: str, values: Dict[str, Any]) -> "Logs":

        for name, value in values.items():
            self.add_entry(collection, name, value)

        return self

    def add_metric(self, name: str, value: Any, *, stateful: bool = False) -> "Logs":
        if isinstance(value, Metric):
            stateful = True
        collection = "metrics" if not stateful else "stateful_metrics"
        return self.add_entry(collection, name, value)

    def add_metrics(self, metrics: Dict[str, Any], *, stateful: bool = False):
        for name, value in metrics.items():
            self.add_metric(name, value, stateful=stateful)

    def add_loss(self, name: str, value: Any, *, add_metric: bool = False):
        self.add_entry("losses", name, value)
        if add_metric:
            self.add_metric(name, value)

    def add_losses(self, losses: Dict[str, Any], *, add_metrics: bool = False):
        for name, value in losses.items():
            self.add_loss(name, value, add_metric=add_metrics)

    def add_output(self, name: str, value: Any, *, per_sample: bool = True):
        collection = "per_sample_outputs" if per_sample else "outputs"
        self.add_entry(collection, name, value)

    # ----------------------------------
    # history behavior
    # ----------------------------------

    def entry_value(self, name: str) -> Any:
        path = self.entry_path(name)
        if path is None:
            raise KeyError(f"Key {name} not found in logs.")
        collection, name = path
        return self[collection][name]

    def entry_path(self, name: str) -> Optional[LogPath]:
        path = name.split(".")

        if len(path) == 1:
            name = path[0]
            collection = self.entry_collection(name)
            if collection is None:
                return None
        elif len(path) == 2:
            collection, name = path
        else:
            raise ValueError(f"Got more than 2 levels of nesting in key '{name}'")

        return collection, name

    def entry_collection(self, name: str) -> Optional[str]:
        collections = [col for col in self if name in self[col]]

        if len(collections) == 0:
            return None
        elif len(collections) == 1:
            return collections[0]
        else:
            raise ValueError(
                f"Found multiple collections for name '{name}' : {collections}. "
                "Use `collection.name` syntax."
            )

    def merge(self, collection_updates: LogsLike):
        for collection, updates in collection_updates.items():
            if not isinstance(updates, Mapping):
                raise ValueError(
                    f"Invalide value '{updates}' for collection '{collection}', value must be a Mapping"
                )
            if collection in self:
                entries = self[collection]
                if isinstance(entries, MutableMapping):
                    entries.update(updates)
                elif isinstance(entries, Mapping):
                    if type(entries) != type(updates):
                        raise ValueError(
                            f"Cannot merge collections of different types: {type(entries)} and {type(updates)}"
                        )
                    self[collection] = updates
                else:
                    raise ValueError(
                        f"Invalid collection '{collection}' of type '{type(entries).__name__}', must be a Mapping "
                        "or MutableMapping"
                    )
            else:
                # NOTE: we copy mutable mappings to avoid side effects
                if isinstance(updates, Dict):
                    self[collection] = updates.copy()
                elif isinstance(updates, MutableMapping):
                    self[collection] = dict(updates)
                else:
                    self[collection] = updates


def _logs_tree_flatten(self):
    return (dict(self),), None


def _logs_tree_unflatten(aux_data, children):
    self = Logs(children[0])
    return self


register_pytree_node(Logs, _logs_tree_flatten, _logs_tree_unflatten)

# ----------------------------------
# history
# ----------------------------------


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
            paths = [logs.entry_path(key) for key in keys]
            if all(path is not None for path in paths):
                for i, path in enumerate(paths):
                    assert path is not None
                    collection, key = path
                    outputs[i].append(logs[collection][key])

        return outputs if len(keys) > 1 else outputs[0]

    def commit(self, elapsed: Elapsed, logs: LogsLike):
        # convert JAX arrays to numpy arrays to free memory
        logs = jax.tree_map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, logs
        )
        if not isinstance(logs, Logs):
            logs = Logs(logs)
        logs["elapsed"] = elapsed
        self.append(logs)


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
        outputs = inject(self.f)(
            loop_state.state, loop_state.batch, loop_state.elapsed, loop_state
        )
        return to_standard_outputs(outputs, loop_state.state)


def inject(f: Callable[..., A]) -> Callable[..., A]:
    def _inject(*args) -> A:
        n_args = len(inspect.getfullargspec(f).args)
        if inspect.ismethod(f) or inspect.ismethod(f.__call__):
            n_args -= 1
        return f(*args[:n_args])

    return _inject


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
