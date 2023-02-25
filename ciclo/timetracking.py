import dataclasses
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from flax import struct

from ciclo.types import B


class Elapsed(struct.PyTreeNode):
    steps: int
    samples: int
    date: float
    date_start: float = struct.field(pytree_node=True, repr=False)

    @property
    def time(self) -> float:
        return self.date - self.date_start

    @classmethod
    def create(cls, steps: int = 0, samples: int = 0) -> "Elapsed":
        now = datetime.now().timestamp()
        return cls(steps=steps, samples=samples, date_start=now, date=now)

    def update(self, batch_size: int) -> "Elapsed":
        return self.replace(
            steps=self.steps + 1,
            samples=self.samples + batch_size,
            date=datetime.now().timestamp(),
        )

    def update_time(self) -> "Elapsed":
        return self.replace(date=datetime.now().timestamp())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "samples": self.samples,
            "date": self.date,
            "date_start": self.date_start,
            "time": self.time,
        }

    def __sub__(self, other: "Elapsed") -> "Elapsed":
        return Elapsed(
            steps=self.steps - other.steps,
            samples=self.samples - other.samples,
            date=self.date,
            date_start=other.date_start,
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


@dataclasses.dataclass(frozen=True)
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


PeriodLike = Union[Period, int]


def to_period(period: PeriodLike) -> Period:
    if isinstance(period, int):
        period = Period.create(steps=period)
    return period


def elapse(
    dataset: Iterable[B],
    initial: Optional[Elapsed] = None,
    stop: Optional[PeriodLike] = None,
    batch_size_fn: Optional[Callable[[List[Tuple[int, ...]]], int]] = None,
) -> Iterable[Tuple[Elapsed, B]]:
    from ciclo.utils import get_batch_size, max_first_axis

    if batch_size_fn is None:
        batch_size_fn = max_first_axis

    if stop is not None:
        stop = to_period(stop)

    elapsed = initial or Elapsed.create()
    for batch in dataset:
        yield elapsed, batch
        batch_size = get_batch_size(batch, batch_size_fn=batch_size_fn)
        elapsed = elapsed.update(batch_size)

        if stop and elapsed >= stop:
            break
