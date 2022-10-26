from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar
from ciclo.api import Batch, Callback, Elapsed, FunctionCallback, Period, Logs, B
import jax

F = TypeVar("F", bound=Callable[..., Any])


def logs(**kwargs: Dict[str, Any]) -> Logs:
    # copy internal dicts to avoid side effects
    return Logs({k: v.copy() for k, v in kwargs.items()})


def at(
    steps: Optional[int] = None,
    *,
    samples: Optional[int] = None,
    time: Optional[float] = None,
    date: Optional[float] = None,
) -> Period:
    return Period.create(steps=steps, samples=samples, time=time, date=date)


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


def callback(f) -> FunctionCallback:
    return FunctionCallback(f)


def elapse(
    dataset: Iterable[B], initial: Optional[Elapsed] = None
) -> Iterable[Tuple[Elapsed, B]]:
    elapsed = initial or Elapsed.create()
    for batch in dataset:
        batch_size = get_batch_size(batch)
        elapsed = elapsed.update(batch_size)
        yield elapsed, batch
