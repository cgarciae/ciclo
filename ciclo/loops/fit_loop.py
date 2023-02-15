from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ciclo import callbacks as callbacks_lib
from ciclo import schedules
from ciclo.logging import History
from ciclo.loops import loop
from ciclo.loops.loop import LoopCallbackLike, LoopOutput
from ciclo.schedules import ScheduleLike
from ciclo.timetracking import Elapsed, Period, PeriodLike
from ciclo.types import B, S

CallbackOrList = Union[LoopCallbackLike, List[LoopCallbackLike]]
FitScheduleLike = Union[ScheduleLike, str]
FitInputTasks = Dict[FitScheduleLike, CallbackOrList]
FitCallback = Any

FIT_CALLBACK_NAMES = [
    "train_step",
    "test_step",
    # "validation_step",
    "reset_step",
    # ----------------
    "on_train_begin",
    "on_train_end",
    # "on_epoch_begin",
    "on_epoch_end",
    # "on_predict_batch_begin",
    # "on_predict_batch_end",
    # "on_predict_begin",
    # "on_predict_end",
    "on_test_batch_begin",
    "on_test_batch_end",
    "on_test_begin",
    "on_test_end",
    "on_train_batch_begin",
    "on_train_batch_end",
]


def fit_loop(
    state: S,
    dataset: Iterable[B],
    tasks: Optional[FitInputTasks] = None,
    *,
    callbacks: Optional[FitCallback] = None,
    eval_dataset: Optional[Callable[[], Iterable[B]]] = None,
    eval_every: Optional[PeriodLike] = None,
    eval_duration: Optional[PeriodLike] = None,
    stop: Optional[PeriodLike] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
) -> LoopOutput[S]:
    if tasks is None:
        tasks = {}

    if isinstance(eval_every, int):
        eval_every = Period.create(steps=eval_every)

    if isinstance(eval_duration, int):
        eval_duration = Period.create(steps=eval_duration)

    additionl_tasks: Dict[ScheduleLike, CallbackOrList] = {}
    named_tasks: Dict[str, CallbackOrList] = {}
    for schedule in list(tasks.keys()):
        if isinstance(schedule, str):
            named_tasks[schedule] = tasks.pop(schedule)
        else:
            additionl_tasks[schedule] = tasks.pop(schedule)

    named_tasks = {
        schedule: callbacks if isinstance(callbacks, list) else [callbacks]
        for schedule, callbacks in named_tasks.items()
    }

    # extract callbacks from state
    for name in FIT_CALLBACK_NAMES:
        if hasattr(state, name):
            # note: here we get an unbounded method
            callback = getattr(type(state), name)
            named_tasks.setdefault(name, []).insert(0, callback)

    # extract callbacks from callbacks
    if callbacks is not None:
        for name in FIT_CALLBACK_NAMES:
            if hasattr(callbacks, name):
                callback = getattr(callbacks, name)
                named_tasks.setdefault(name, []).append(callback)

    train_tasks = {}

    if eval_dataset is not None or eval_every is not None:
        if eval_dataset is None or eval_every is None:
            raise ValueError("eval_interval and eval_dataset must be set together")

        eval_schedule = schedules.every(
            steps=eval_every.steps,
            samples=eval_every.samples,
            time=eval_every.time,
        )

        eval_on_start = named_tasks.pop("on_test_begin", None)
        eval_tasks = {
            schedules.every(1): [
                *named_tasks.pop("on_test_batch_begin", []),
                *named_tasks.pop("test_step", []),
                *named_tasks.pop("on_test_batch_end", []),
            ],
        }
        eval_on_end = named_tasks.pop("on_test_end", None)

        train_tasks[eval_schedule] = [
            *named_tasks.pop("reset_step", []),
            callbacks_lib.inner_loop(
                "test",
                lambda state: loop.loop(
                    state,
                    eval_dataset(),
                    on_start=eval_on_start,
                    tasks=eval_tasks,
                    on_end=eval_on_end,
                    stop=eval_duration,
                ),
            ),
            *named_tasks.pop("on_epoch_end", []),
        ]
    else:
        named_tasks.pop("reset_step", None)
        named_tasks.pop("on_test_begin", None)
        named_tasks.pop("on_test_batch_begin", None)
        named_tasks.pop("test_step", None)
        named_tasks.pop("on_test_batch_end", None)
        named_tasks.pop("on_test_end", None)
        named_tasks.pop("on_epoch_end", None)

    train_tasks[schedules.every(1)] = [
        *named_tasks.pop("on_train_batch_begin", []),
        *named_tasks.pop("train_step", []),
        *named_tasks.pop("on_train_batch_end", []),
    ]
    train_tasks.update(additionl_tasks)

    on_start = named_tasks.pop("on_train_begin", None)
    on_end = named_tasks.pop("on_train_end", None)

    if len(named_tasks) > 0:
        raise ValueError(f"Unknown tasks: {named_tasks}")

    return loop.loop(
        state,
        dataset,
        on_start=on_start,
        tasks=train_tasks,
        on_end=on_end,
        stop=stop,
        history=history,
        elapsed=elapsed,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        metadata=metadata,
    )
