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

PREDICT_NAMES = {
    PREDICT_STEP := "predict_step",
    # ----------------
    ON_PREDICT_BATCH_BEGIN := "on_predict_batch_begin",
    ON_PREDICT_BATCH_END := "on_predict_batch_end",
    ON_PREDICT_BEGIN := "on_predict_begin",
    ON_PREDICT_END := "on_predict_end",
}

EVAL_NAMES = {
    TEST_STEP := "test_step",
    RESET_STEP := "reset_step",
    # ----------------
    ON_TEST_BATCH_BEGIN := "on_test_batch_begin",
    ON_TEST_BATCH_END := "on_test_batch_end",
    ON_TEST_BEGIN := "on_test_begin",
    ON_TEST_END := "on_test_end",
}

FIT_NAMES = {
    TRAIN_STEP := "train_step",
    RESET_STEP := "reset_step",
    # ----------------
    ON_TRAIN_BEGIN := "on_train_begin",
    ON_TRAIN_END := "on_train_end",
    # NOTE: due to a choice in the implementation of the fit_loop
    # there is no ON_EPOCH_BEGIN, instead you can use a combination of
    # ON_TRAIN_BEGIN and ON_EPOCH_END
    # ON_EPOCH_BEGIN := "on_epoch_begin",
    ON_EPOCH_END := "on_epoch_end",
    ON_TRAIN_BATCH_BEGIN := "on_train_batch_begin",
    ON_TRAIN_BATCH_END := "on_train_batch_end",
}


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
    for name in FIT_NAMES | EVAL_NAMES:
        if hasattr(state, name):
            # note: here we get an unbounded method
            callback = getattr(type(state), name)
            named_tasks.setdefault(name, []).insert(0, callback)

    # extract callbacks from callbacks
    if callbacks is not None:
        for name in FIT_NAMES | EVAL_NAMES:
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

        eval_on_start = named_tasks.pop(ON_TEST_BEGIN, None)
        eval_tasks = {
            schedules.every(1): [
                *named_tasks.pop(ON_TEST_BATCH_BEGIN, []),
                *named_tasks.pop(TEST_STEP, []),
                *named_tasks.pop(ON_TEST_BATCH_END, []),
            ],
        }
        eval_on_end = named_tasks.pop(ON_TEST_END, None)

        train_tasks[eval_schedule] = [
            *named_tasks.pop(RESET_STEP, []),
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
            *named_tasks.pop(ON_EPOCH_END, []),
        ]
    else:
        named_tasks.pop(RESET_STEP, None)
        named_tasks.pop(ON_TEST_BEGIN, None)
        named_tasks.pop(ON_TEST_BATCH_BEGIN, None)
        named_tasks.pop(TEST_STEP, None)
        named_tasks.pop(ON_TEST_BATCH_END, None)
        named_tasks.pop(ON_TEST_END, None)
        named_tasks.pop(ON_EPOCH_END, None)

    train_tasks[schedules.every(1)] = [
        *named_tasks.pop(ON_TRAIN_BATCH_BEGIN, []),
        *named_tasks.pop(TRAIN_STEP, []),
        *named_tasks.pop(ON_TRAIN_BATCH_END, []),
    ]
    train_tasks.update(additionl_tasks)

    on_start = named_tasks.pop(ON_TRAIN_BEGIN, None)
    on_end = named_tasks.pop(ON_TRAIN_END, None)

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


def test_loop(
    state: S,
    dataset: Iterable[B],
    tasks: Optional[TestInputTasks] = None,
    *,
    callbacks: Optional[TestCallback] = None,
    stop: Optional[PeriodLike] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
) -> LoopOutput[S]:
    if tasks is None:
        tasks = {}

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
    for name in TEST_NAMES:
        if hasattr(state, name):
            # note: here we get an unbounded method
            callback = getattr(type(state), name)
            named_tasks.setdefault(name, []).insert(0, callback)

    # extract callbacks from callbacks
    if callbacks is not None:
        for name in TEST_NAMES:
            if hasattr(callbacks, name):
                callback = getattr(callbacks, name)
                named_tasks.setdefault(name, []).append(callback)

    test_tasks = {
        schedules.every(1): [
            *named_tasks.pop(ON_TEST_BATCH_BEGIN, []),
            *named_tasks.pop(TEST_STEP, []),
            *named_tasks.pop(ON_TEST_BATCH_END, []),
        ],
    }
    test_tasks.update(additionl_tasks)

    on_start = named_tasks.pop(ON_TEST_BEGIN, None)
    on_end = named_tasks.pop(ON_TEST_END, None)

    if len(named_tasks) > 0:
        raise ValueError(f"Unknown tasks: {named_tasks}")

    return loop.loop(
        state,
        dataset,
        on_start=on_start,
        tasks=test_tasks,
        on_end=on_end,
        stop=stop,
        history=history,
        elapsed=elapsed,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        metadata=metadata,
    )
