from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar
from ciclo.api import (
    Batch,
    History,
    LogsLike,
    Elapsed,
    Period,
    Logs,
    B,
)
from ciclo.loops import LoopCallback, LoopFunctionCallback
import jax

F = TypeVar("F", bound=Callable[..., Any])


def logs(*args, **kwargs: Dict[str, Any]) -> Logs:
    return Logs(*args, **kwargs)


def history(logs_list: Optional[List[LogsLike]] = None) -> History:
    if logs_list is None:
        return History()

    return History(
        Logs(logs) if not isinstance(logs, Logs) else logs for logs in logs_list
    )


def at(
    steps: Optional[int] = None,
    *,
    samples: Optional[int] = None,
    time: Optional[float] = None,
    date: Optional[float] = None,
) -> Period:
    return Period.create(steps=steps, samples=samples, time=time, date=date)


def is_scalar(x):
    if isinstance(x, (int, float, bool)):
        return True
    elif hasattr(x, "shape"):
        return x.shape == () or x.shape == (1,)
    else:
        return False


def callback(f) -> LoopFunctionCallback:
    return LoopFunctionCallback(f)
