import dataclasses
from datetime import datetime, timedelta
from typing import Any, Callable, Iterable, Mapping, Union

from flax import struct


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


__all__ = ["Elapsed", "Period"]
