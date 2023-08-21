from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

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
    (ON_PREDICT_STEP := "on_predict_step"),
    # also support the simple names
    (PREDICT_STEP := "predict_step"),
    # ----------------
    (ON_PREDICT_BATCH_BEGIN := "on_predict_batch_begin"),
    (ON_PREDICT_BATCH_END := "on_predict_batch_end"),
    (ON_PREDICT_BEGIN := "on_predict_begin"),
    (ON_PREDICT_END := "on_predict_end"),
}

TEST_NAMES = {
    (ON_TEST_STEP := "on_test_step"),
    (ON_RESET_STEP := "on_reset_step"),
    # also support the simple names
    (TEST_STEP := "test_step"),
    (RESET_STEP := "reset_step"),
    # ----------------
    (ON_TEST_BATCH_BEGIN := "on_test_batch_begin"),
    (ON_TEST_BATCH_END := "on_test_batch_end"),
    (ON_TEST_BEGIN := "on_test_begin"),
    (ON_TEST_END := "on_test_end"),
}

# NOTE: due to a choice in the implementation of the fit_loop
# there is no ON_EPOCH_BEGIN, instead you can use a combination
# of ON_TRAIN_BEGIN and ON_EPOCH_END
FIT_NAMES = {
    (ON_TRAIN_STEP := "on_train_step"),
    ON_RESET_STEP,
    # also support the simple names
    (TRAIN_STEP := "train_step"),
    RESET_STEP,
    # ----------------
    (ON_TRAIN_BEGIN := "on_train_begin"),
    (ON_TRAIN_END := "on_train_end"),
    # ON_EPOCH_BEGIN := "on_epoch_begin",
    (ON_EPOCH_END := "on_epoch_end"),
    (ON_TRAIN_BATCH_BEGIN := "on_train_batch_begin"),
    (ON_TRAIN_BATCH_END := "on_train_batch_end"),
}


def train_loop(
    state: S,
    dataset: Iterable[B],
    tasks: Optional[FitInputTasks] = None,
    *,
    callbacks: Optional[List[FitCallback]] = None,
    test_dataset: Optional[Callable[[], Iterable[B]]] = None,
    epoch_duration: Optional[PeriodLike] = None,
    test_duration: Optional[PeriodLike] = None,
    test_name: str = "test",
    stop: Optional[PeriodLike] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
    batch_size_fn: Optional[Callable[[List[Tuple[int, ...]]], int]] = None,
    inner_loop_kwargs: Optional[Dict[str, Any]] = None,
) -> LoopOutput[S]:
    if tasks is None:
        tasks = {}

    if isinstance(epoch_duration, int):
        epoch_duration = Period.create(steps=epoch_duration)

    if isinstance(test_duration, int):
        test_duration = Period.create(steps=test_duration)

    if inner_loop_kwargs is None:
        inner_loop_kwargs = {}

    additionl_tasks: Dict[ScheduleLike, CallbackOrList] = {}
    named_tasks: Dict[str, CallbackOrList] = {}
    for schedule in list(tasks.keys()):
        if isinstance(schedule, str):
            named_tasks[schedule] = tasks.pop(schedule)
        else:
            additionl_tasks[schedule] = tasks.pop(schedule)

    # check that all named tasks are valid
    unknown_names = set(named_tasks.keys()) - FIT_NAMES - TEST_NAMES
    if len(unknown_names) > 0:
        raise ValueError(f"Unknown named tasks: {unknown_names}")

    named_tasks = {
        schedule: callbacks if isinstance(callbacks, list) else [callbacks]
        for schedule, callbacks in named_tasks.items()
    }

    # extract callbacks from state
    for name in FIT_NAMES | TEST_NAMES:
        if hasattr(state, name):
            # note: here we get an unbounded method
            callback = getattr(type(state), name)
            named_tasks.setdefault(name, []).insert(0, callback)

    # extract callbacks from callbacks
    if callbacks is not None:
        for callback in callbacks:
            for name in FIT_NAMES | TEST_NAMES:
                if hasattr(callback, name):
                    callback_method = getattr(callback, name)
                    named_tasks.setdefault(name, []).append(callback_method)

    # extract test named tasks
    named_tasks_test = {}
    for name in TEST_NAMES - FIT_NAMES:
        if name in named_tasks:
            named_tasks_test[name] = named_tasks.pop(name)

    train_tasks = {}

    train_tasks[schedules.every(1)] = [
        *named_tasks.get(ON_TRAIN_BATCH_BEGIN, []),
        *named_tasks.get(TRAIN_STEP, []),
        *named_tasks.get(ON_TRAIN_STEP, []),
    ]

    if epoch_duration is not None:
        test_tasks = []
        test_tasks += named_tasks.pop(RESET_STEP, [])
        test_tasks += named_tasks.pop(ON_RESET_STEP, [])
        if test_dataset is not None:
            test_tasks.append(
                callbacks_lib.inner_loop(
                    test_name,
                    lambda state: test_loop(
                        state,
                        test_dataset(),
                        tasks=named_tasks_test,
                        stop=test_duration,
                        batch_size_fn=batch_size_fn,
                    ),
                    **inner_loop_kwargs,
                )
            )
        test_tasks += named_tasks.pop(ON_EPOCH_END, [])

        eval_schedule = schedules.every(
            steps=epoch_duration.steps,
            samples=epoch_duration.samples,
            time=epoch_duration.time,
        )

        train_tasks[eval_schedule] = test_tasks

    train_tasks[schedules.every(1)] = [
        *named_tasks.get(ON_TRAIN_BATCH_END, []),
    ]

    train_tasks.update(additionl_tasks)

    return loop.loop(
        state,
        dataset,
        tasks=train_tasks,
        on_start=named_tasks.get(ON_TRAIN_BEGIN, None),
        on_end=named_tasks.get(ON_TRAIN_END, None),
        stop=stop,
        history=history,
        elapsed=elapsed,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        metadata=metadata,
        batch_size_fn=batch_size_fn,
    )


def test_loop(
    state: S,
    dataset: Iterable[B],
    tasks: Optional[FitInputTasks] = None,
    *,
    callbacks: Optional[FitCallback] = None,
    stop: Optional[PeriodLike] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
    batch_size_fn: Optional[Callable[[List[Tuple[int, ...]]], int]] = None,
) -> LoopOutput[S]:
    if tasks is None:
        tasks = {}

    # create a copy of tasks to avoid modifying the original
    tasks = tasks.copy()
    additionl_tasks: Dict[ScheduleLike, CallbackOrList] = {}
    named_tasks: Dict[str, CallbackOrList] = {}
    for schedule in list(tasks.keys()):
        if isinstance(schedule, str):
            named_tasks[schedule] = tasks.pop(schedule)
        else:
            additionl_tasks[schedule] = tasks.pop(schedule)

    # check that all named tasks are valid
    unknown_names = set(named_tasks.keys()) - TEST_NAMES
    if len(unknown_names) > 0:
        raise ValueError(f"Unknown named tasks: {unknown_names}")

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
        for callback in callbacks:
            for name in TEST_NAMES:
                if hasattr(callback, name):
                    callback_method = getattr(callback, name)
                    named_tasks.setdefault(name, []).append(callback_method)

    test_tasks = {}
    test_tasks[schedules.every(1)] = [
        *named_tasks.get(ON_TEST_BATCH_BEGIN, []),
        *named_tasks.get(TEST_STEP, []),
        *named_tasks.get(ON_TEST_STEP, []),
        *named_tasks.get(ON_TEST_BATCH_END, []),
    ]
    test_tasks.update(additionl_tasks)

    return loop.loop(
        state,
        dataset,
        tasks=test_tasks,
        on_start=named_tasks.get(ON_TEST_BEGIN, [])
        + named_tasks.get(RESET_STEP, [])
        + named_tasks.get(ON_RESET_STEP, []),
        on_end=named_tasks.get(ON_TEST_END, None),
        stop=stop,
        history=history,
        elapsed=elapsed,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        metadata=metadata,
        batch_size_fn=batch_size_fn,
    )


def predict_loop(
    state: S,
    dataset: Iterable[B],
    tasks: Optional[FitInputTasks] = None,
    *,
    callbacks: Optional[FitCallback] = None,
    stop: Optional[PeriodLike] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
    batch_size_fn: Optional[Callable[[List[Tuple[int, ...]]], int]] = None,
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

    # check that all named tasks are valid
    unknown_names = set(named_tasks.keys()) - PREDICT_NAMES
    if len(unknown_names) > 0:
        raise ValueError(f"Unknown named tasks: {unknown_names}")

    named_tasks = {
        schedule: callbacks if isinstance(callbacks, list) else [callbacks]
        for schedule, callbacks in named_tasks.items()
    }

    # extract callbacks from state
    for name in PREDICT_NAMES:
        if hasattr(state, name):
            # note: here we get an unbounded method
            callback = getattr(type(state), name)
            named_tasks.setdefault(name, []).insert(0, callback)

    # extract callbacks from callbacks
    if callbacks is not None:
        for callback in callbacks:
            for name in PREDICT_NAMES:
                if hasattr(callback, name):
                    callback_method = getattr(callback, name)
                    named_tasks.setdefault(name, []).append(callback_method)

    predict_tasks = {}
    predict_tasks[schedules.every(1)] = [
        *named_tasks.get(ON_PREDICT_BATCH_BEGIN, []),
        *named_tasks.get(PREDICT_STEP, []),
        *named_tasks.get(ON_PREDICT_STEP, []),
        *named_tasks.get(ON_PREDICT_BATCH_END, []),
    ]
    predict_tasks.update(additionl_tasks)

    return loop.loop(
        state,
        dataset,
        tasks=predict_tasks,
        on_start=named_tasks.get(ON_PREDICT_BEGIN, None),
        on_end=named_tasks.get(ON_PREDICT_END, None),
        stop=stop,
        history=history,
        elapsed=elapsed,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        metadata=metadata,
        batch_size_fn=batch_size_fn,
    )
