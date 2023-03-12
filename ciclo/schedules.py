import abc
import dataclasses
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from ciclo.timetracking import Elapsed, Period
from ciclo.types import A, Schedule

ScheduleLike = Schedule


@runtime_checkable
class Copyable(Protocol):
    def copy(self: A) -> A:
        ...


# ---------------------------------------
# base
# ---------------------------------------


class ScheduleBase(abc.ABC, Copyable):
    @abc.abstractmethod
    def __call__(self, elapsed: Elapsed) -> bool:
        pass

    @abc.abstractmethod
    def copy(self: A) -> A:
        ...

    def __and__(self, other: Schedule) -> "And":
        return And.create(self, other)

    def __or__(self, other: Schedule) -> "Or":
        return Or.create(self, other)

    def __not__(self) -> "Not":
        return Not.create(self)


# ---------------------------------------
# logical
# ---------------------------------------


@dataclass(unsafe_hash=True)
class And(ScheduleBase):
    schedules: Tuple[Schedule, ...]

    @classmethod
    def create(cls, *schedules: ScheduleLike) -> "And":
        return cls(schedules=tuple(map(to_schedule, schedules)))

    def __call__(self, elapsed: Elapsed) -> bool:
        return all(schedule(elapsed) for schedule in self.schedules)

    def copy(self) -> "And":
        return dataclasses.replace(self)


@dataclass(unsafe_hash=True)
class Or(ScheduleBase):
    schedules: Tuple[Schedule, ...]

    @classmethod
    def create(cls, *schedules: ScheduleLike) -> "Or":
        return Or(schedules=tuple(map(to_schedule, schedules)))

    def __call__(self, elapsed: Elapsed) -> bool:
        return any(schedule(elapsed) for schedule in self.schedules)

    def copy(self) -> "Or":
        return dataclasses.replace(self)


@dataclass(unsafe_hash=True)
class Not(ScheduleBase):
    schedule: Schedule

    @classmethod
    def create(cls, schedule: Schedule) -> "Not":
        return cls(schedule=to_schedule(schedule))

    def __call__(self, elapsed: Elapsed) -> bool:
        return not self.schedule(elapsed)

    def copy(self) -> "Not":
        return dataclasses.replace(self)


@dataclass(unsafe_hash=True)
class Constant(ScheduleBase):
    value: bool

    @classmethod
    def create(cls, value: bool) -> "Constant":
        return cls(value=value)

    def __call__(self, elapsed: Elapsed) -> bool:
        return self.value

    def copy(self) -> "Constant":
        return dataclasses.replace(self)


@dataclass(unsafe_hash=True)
class Lambda(ScheduleBase):
    func: Schedule

    @classmethod
    def create(cls, func: Schedule) -> "Lambda":
        return cls(func=func)

    def __call__(self, elapsed: Elapsed) -> bool:
        return self.func(elapsed)

    def copy(self) -> "Lambda":
        return dataclasses.replace(self)


def always() -> Constant:
    return Constant.create(True)


def never() -> Constant:
    return Constant.create(False)


# ---------------------------------------
# every
# ---------------------------------------


def every(
    steps: Union[int, None] = None,
    *,
    samples: Union[int, None] = None,
    time: Union[timedelta, float, int, None] = None,
) -> "Every":
    return Every(
        period=Period.create(steps=steps, samples=samples, time=time),
        last_samples=0,
        last_time=datetime.now().timestamp(),
    )


@dataclass(unsafe_hash=True)
class Every(ScheduleBase):
    period: Period
    last_samples: int
    last_time: float

    def __call__(self, elapsed: Elapsed) -> bool:
        if self.period.steps is not None:
            steps = elapsed.steps
            return steps >= 0 and steps % self.period.steps == 0

        if self.period.samples is not None:
            if elapsed.samples - self.last_samples >= self.period.samples:
                self.last_samples = elapsed.samples
                return True

        if self.period.time is not None:
            if elapsed.date - self.last_time >= self.period.time:
                self.last_time = elapsed.date
                return True

        return False

    def copy(self) -> "Every":
        return dataclasses.replace(self)


# ---------------------------------------
# after
# ---------------------------------------


def after(
    steps: Union[int, None] = None,
    *,
    samples: Union[int, None] = None,
    time: Union[timedelta, float, int, None] = None,
) -> "After":
    return After(
        period=Period.create(steps=steps, samples=samples, time=time),
    )


@dataclass(unsafe_hash=True)
class After(ScheduleBase):
    period: Period

    def __call__(self, elapsed: Elapsed) -> bool:
        return elapsed >= self.period

    def copy(self) -> "After":
        return dataclasses.replace(self)

    def then(self, schedule: Schedule) -> "Piecewise":
        return piecewise(never(), {self.period: schedule})

    def __rshift__(self, schedule: Schedule) -> "Piecewise":
        return self.then(schedule)

    def every(
        self,
        steps: Optional[int] = None,
        *,
        samples: Optional[int] = None,
        time: Optional[Union[timedelta, float, int]] = None,
    ) -> "Piecewise":
        return self.then(every(steps=steps, samples=samples, time=time))


# ---------------------------------------
# piecewise
# ---------------------------------------


def piecewise(
    schedule: Schedule,
    period_schedules: Dict[Period, Schedule],
) -> "Piecewise":
    return Piecewise(
        schedule=to_schedule(schedule),
        period_schedules=tuple(
            (p, to_schedule(s)) for p, s in period_schedules.items()
        ),
    )


@dataclass(unsafe_hash=True)
class Piecewise(ScheduleBase):
    schedule: Schedule
    period_schedules: Tuple[Tuple[Period, Schedule], ...]
    reference: Optional[Elapsed] = None
    schedule_index: int = -1

    @property
    def current_schedule(self) -> Schedule:
        if self.schedule_index == -1:
            return self.schedule
        else:
            return self.period_schedules[self.schedule_index][1]

    @property
    def current_period(self) -> Period:
        return self.period_schedules[self.schedule_index + 1][0]

    def __call__(self, elapsed: Elapsed) -> bool:
        if self.reference is None:
            self.reference = elapsed

        if self.schedule_index < len(self.period_schedules) - 1:
            if elapsed >= self.current_period:
                self.schedule_index += 1
                self.reference = elapsed

        elapsed = elapsed - self.reference
        return self.current_schedule(elapsed)

    def copy(self) -> "Piecewise":
        return dataclasses.replace(self)


# ---------------------------------------
# utils
# ---------------------------------------


def to_schedule(value: Any) -> Schedule:
    value = maybe_copy(value)

    if isinstance(value, ScheduleBase):
        return value
    elif callable(value):
        return Lambda.create(value)
    # elif isinstance(value, bool):
    #     return Constant.create(value)
    # elif isinstance(value, int):
    #     return every(steps=value)
    else:
        raise ValueError(
            f"Invalid schedule, must be a Schedule, callable, bool or int, got: {value}"
        )


def maybe_copy(value: A) -> A:
    if isinstance(value, Copyable):
        return value.copy()
    else:
        return value
