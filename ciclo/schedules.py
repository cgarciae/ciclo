from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

from ciclo.timetracking import Elapsed, Period
from ciclo.types import Schedule

# ---------------------------------------
# every
# ---------------------------------------


def every(
    steps: Union[int, None] = None,
    *,
    samples: Union[int, None] = None,
    time: Union[timedelta, float, int, None] = None,
    steps_offset: int = 1,
) -> Schedule:
    return Every(
        period=Period.create(steps=steps, samples=samples, time=time),
        last_samples=0,
        last_time=datetime.now().timestamp(),
        steps_offset=steps_offset,
    )


@dataclass(unsafe_hash=True)
class Every:
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
) -> Schedule:
    return Piecewise(
        schedule=schedule,
        period_schedules=list(period_schedules.items()),
    )


@dataclass
class Piecewise:
    schedule: Schedule
    period_schedules: List[Tuple[Period, Schedule]]

    def __call__(self, elapsed: Elapsed) -> bool:
        if len(self.period_schedules) > 0:
            period, next_schedule = self.period_schedules[0]
            if elapsed >= period:
                self.schedule = next_schedule
                self.period_schedules = self.period_schedules[1:]

        return self.schedule(elapsed)
