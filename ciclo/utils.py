import functools
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import jax

from ciclo.logging import Elapsed, History, Logs
from ciclo.loops import LoopFunctionCallback
from ciclo.timetracking import Period
from ciclo.types import A, B, Batch, LogsLike

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


def get_batch_size(batch: Batch) -> int:
    def get_size(sizes, x):
        sizes.add(x.shape[0])
        return sizes

    sizes = jax.tree_util.tree_reduce(get_size, batch, set())
    if len(sizes) != 1:
        raise ValueError("Batch size must be the same for all elements in the batch.")
    return sizes.pop()


def elapse(
    dataset: Iterable[B], initial: Optional[Elapsed] = None
) -> Iterable[Tuple[Elapsed, B]]:
    elapsed = initial or Elapsed.create()
    for batch in dataset:
        batch_size = get_batch_size(batch)
        elapsed = elapsed.update(batch_size)
        yield elapsed, batch


def inject(f: Callable[..., A]) -> Callable[..., A]:
    @functools.wraps(f)
    def _inject(*args) -> A:
        n_args = len(inspect.getfullargspec(f).args)
        if inspect.ismethod(f) or inspect.ismethod(f.__call__):
            n_args -= 1
        return f(*args[:n_args])

    return _inject
