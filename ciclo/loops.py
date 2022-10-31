from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Union
from ciclo.api import (
    S,
    B,
    Batch,
    LoopCallback,
    Elapsed,
    History,
    InputCallback,
    InputTasks,
    Logs,
    LoopOutput,
    LoopState,
    Period,
    get_loop_callback,
    inject,
)
from ciclo.utils import get_batch_size, elapse


def loop(
    state: S,
    dataset: Iterable[B],
    tasks: InputTasks,
    *,
    stop: Union[Period, int, None] = None,
    on_start: Union[InputCallback, List[InputCallback], None] = None,
    on_end: Union[InputCallback, List[InputCallback], None] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
    metadata: Optional[Any] = None,
) -> LoopOutput[S]:

    if isinstance(stop, int):
        stop_period = Period.create(steps=stop)
    else:
        stop_period = stop

    if history is None:
        history = History()

    loop_state = LoopState(
        state=state,
        batch=None,
        history=history,
        elapsed=elapsed or Elapsed.create(),
        logs=Logs(),
        accumulated_logs=Logs(),
        metadata=metadata,
    )

    if on_start is None:
        on_start = []
    elif not isinstance(on_start, list):
        on_start = [on_start]

    if on_end is None:
        on_end = []
    elif not isinstance(on_end, list):
        on_end = [on_end]

    tasks_ = [
        (
            schedule,
            [
                get_loop_callback(f)
                for f in (callbacks if isinstance(callbacks, list) else [callbacks])
            ],
        )
        for schedule, callbacks in tasks.items()
    ]

    try:

        for i, (elapsed, batch) in enumerate(elapse(dataset, initial=elapsed)):
            loop_state.elapsed = elapsed
            loop_state.batch = batch

            # call on_start on first batch
            if i == 0:
                loop_state.logs = Logs()
                for callback in on_start:
                    callback = get_loop_callback(callback)
                    _make_call(loop_state, callback)

            loop_state.logs = Logs()
            for s, (schedule, callbacks) in enumerate(tasks_):
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback)
                        if loop_state.stop_iteration:
                            break
                if loop_state.stop_iteration:
                    break

            if loop_state.logs:
                loop_state.history.commit(elapsed, loop_state.logs)

            if loop_state.stop_iteration or (
                stop_period and loop_state.elapsed >= stop_period
            ):
                break

        # call on_end on last batch
        loop_state.logs = Logs()
        for callback in on_end:
            callback = get_loop_callback(callback)
            _make_call(loop_state, callback)

    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            print("\nStopping loop...")
        else:
            raise

    return loop_state.state, loop_state.history, loop_state.elapsed


def _make_call(loop_state: LoopState[S], callback: LoopCallback[S]):
    try:
        loop_state.elapsed = loop_state.elapsed.update_time()
        logs, state = callback.__loop_callback__(loop_state)
        loop_state.logs.merge(logs)
        loop_state.accumulated_logs.merge(logs)
        loop_state.state = state
    except BaseException as e:
        raise type(e)(f"Error in callback {callback}: {e}") from e
