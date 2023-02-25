import functools
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import jax
import numpy as np

from ciclo.logging import History, Logs
from ciclo.loops.loop import LoopFunctionCallback
from ciclo.timetracking import Period
from ciclo.types import A, B, Batch

F = TypeVar("F", bound=Callable[..., Any])


def logs(*args, **kwargs: Dict[str, Any]) -> Logs:
    return Logs(*args, **kwargs)


def history(logs_list: Optional[List[Logs]] = None) -> History:
    if logs_list is None:
        return History()

    return History(logs_list)


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


def get_batch_size(
    batch: Batch, batch_size_fn: Callable[[List[Tuple[int, ...]]], int]
) -> int:
    def get_shape(x) -> Tuple[int, ...]:
        if not hasattr(x, "shape"):
            return (1,)
        shape = x.shape
        if len(shape) == 0:
            shape = (1,)
        return shape

    shapes = [get_shape(x) for x in jax.tree_util.tree_leaves(batch)]

    if len(shapes) == 0:
        return 1

    return batch_size_fn(shapes)


def max_first_axis(shapes: List[Tuple[int, ...]]) -> int:
    return max(s[0] for s in shapes)


def inject(f: Callable[..., A]) -> Callable[..., A]:
    @functools.wraps(f)
    def _inject(*args) -> A:
        n_args = len(inspect.getfullargspec(f).args)
        if inspect.ismethod(f) or inspect.ismethod(f.__call__):
            n_args -= 1
        return f(*args[:n_args])

    return _inject
