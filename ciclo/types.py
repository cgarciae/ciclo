import importlib.util
from typing import Any, Callable, Dict, Mapping, Tuple, TypeVar

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
LogsLike = Dict[str, Mapping[str, Any]]
InputCallback = Any
A = TypeVar("A")
S = TypeVar("S", bound=State)
B = TypeVar("B", bound=Batch)
Schedule = Callable[["ciclo.Elapsed"], bool]
