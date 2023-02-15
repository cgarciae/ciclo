import abc
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Union

from ciclo.timetracking import Elapsed, Period
from ciclo.types import Schedule

ScheduleLike = Schedule

# ---------------------------------------
# base
# ---------------------------------------


class ScheduleBase(abc.ABC):
    @abc.abstractmethod
    def __call__(self, elapsed: Elapsed) -> bool:
        pass

    def __and__(self, other: Schedule) -> Schedule:
        return And.create(self, other)

    def __or__(self, other: Schedule) -> Schedule:
        return Or.create(self, other)

    def __not__(self) -> Schedule:
        return Not.create(self)


# ---------------------------------------
# logical
# ---------------------------------------


@dataclass(unsafe_hash=True)
class And(ScheduleBase):
    schedules: Tuple[Schedule, ...]

    @classmethod
    def create(cls, *schedules: ScheduleLike) -> Schedule:
        return cls(schedules=tuple(map(to_schedule, schedules)))

    def __call__(self, elapsed: Elapsed) -> bool:
        return all(schedule(elapsed) for schedule in self.schedules)


@dataclass(unsafe_hash=True)
class Or(ScheduleBase):
    schedules: Tuple[Schedule, ...]

    @classmethod
    def create(cls, *schedules: ScheduleLike) -> Schedule:
        return Or(schedules=tuple(map(to_schedule, schedules)))

    def __call__(self, elapsed: Elapsed) -> bool:
        return any(schedule(elapsed) for schedule in self.schedules)


@dataclass(unsafe_hash=True)
class Not(ScheduleBase):
    schedule: Schedule

    @classmethod
    def create(cls, schedule: Schedule) -> Schedule:
        return cls(schedule=schedule)

    def __call__(self, elapsed: Elapsed) -> bool:
        return not self.schedule(elapsed)


@dataclass(unsafe_hash=True)
class Constant(ScheduleBase):
    value: bool

    @classmethod
    def create(cls, value: bool) -> "Constant":
        return cls(value=value)

    def __call__(self, elapsed: Elapsed) -> bool:
        return self.value


@dataclass(unsafe_hash=True)
class Lambda(ScheduleBase):
    func: Schedule

    @classmethod
    def create(cls, func: Schedule) -> Schedule:
        return cls(func=func)

    def __call__(self, elapsed: Elapsed) -> bool:
        return self.func(elapsed)


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
    steps_offset: int = 0,
) -> "Every":
    return Every(
        period=Period.create(steps=steps, samples=samples, time=time),
        last_samples=0,
        last_time=datetime.now().timestamp(),
        steps_offset=steps_offset,
    )


@dataclass(unsafe_hash=True)
class Every(ScheduleBase):
    period: Period
    last_samples: int
    last_time: float
    steps_offset: int

    def __call__(self, elapsed: Elapsed) -> bool:
        if self.period.steps is not None:
            steps = elapsed.steps - self.steps_offset
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


# ---------------------------------------
# piecewise
# ---------------------------------------


def piecewise(
    schedule: Schedule,
    period_schedules: Dict[Period, Schedule],
) -> "Piecewise":
    return Piecewise(
        schedule=schedule,
        period_schedules=list(period_schedules.items()),
    )


@dataclass
class Piecewise(ScheduleBase):
    schedule: Schedule
    period_schedules: List[Tuple[Period, Schedule]]

    def __call__(self, elapsed: Elapsed) -> bool:
        if len(self.period_schedules) > 0:
            period, next_schedule = self.period_schedules[0]
            if elapsed >= period:
                self.schedule = next_schedule
                self.period_schedules = self.period_schedules[1:]

        return self.schedule(elapsed)


# ---------------------------------------
# utils
# ---------------------------------------


def to_schedule(value: Any) -> Schedule:
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
