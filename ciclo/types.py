import importlib.util
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import ciclo

if importlib.util.find_spec("clu"):
    from clu.metrics import Metric as CluMetric
else:
    locals()["CluMetric"] = type("CluMetric", (), {})

State = Any
Batch = Any
Broadcasts = Any
Statics = Any
LogPath = Tuple[str, str]
A = TypeVar("A")
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)
Schedule = Callable[["ciclo.Elapsed"], bool]


@runtime_checkable
class MetricLike(Protocol):
    @abstractmethod
    def reset(self: A) -> A:
        ...

    @abstractmethod
    def update(self: A, **kwargs) -> A:
        ...

    @abstractmethod
    def batch_updates(self: A, **kwargs) -> A:
        ...

    @abstractmethod
    def merge(self: A, other: A) -> A:
        ...

    @abstractmethod
    def compute(self) -> Any:
        ...

    @abstractmethod
    def reduce(self: A) -> A:
        ...
