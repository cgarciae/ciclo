from dataclasses import dataclass
from typing import Any, List, Optional, Union
from ciclo.api import (
    S,
    Batch,
    Callback,
    Elapsed,
    History,
    InputCallable,
    InputTasks,
    Logs,
    LoopOutput,
    LoopState,
    Period,
    get_callback,
    inject,
)
from ciclo.utils import get_batch_size


def loop(
    state: S,
    dataset,
    tasks: InputTasks,
    *,
    stop: Union[Period, int, None] = None,
    on_start: Union[InputCallable, List[InputCallable], None] = None,
    on_end: Union[InputCallable, List[InputCallable], None] = None,
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

    if elapsed is None:
        elapsed = Elapsed.create()

    loop_state = LoopState(
        state=state,
        history=history,
        elapsed=elapsed,
        step_logs=Logs(),
        accumulated_logs=Logs(),
        metadata=metadata,
    )

    tasks_ = [
        (
            schedule,
            [
                get_callback(f)
                for f in (callbacks if isinstance(callbacks, list) else [callbacks])
            ],
        )
        for schedule, callbacks in tasks.items()
    ]
    batch = None

    try:
        for i, batch in enumerate(dataset):
            loop_state.step_logs = Logs()
            elapsed_initial = loop_state.elapsed.update(get_batch_size(batch))
            loop_state.elapsed = elapsed_initial

            # call on_start on first batch
            if i == 0 and on_start is not None:
                if not isinstance(on_start, list):
                    on_start = [on_start]
                for callback in on_start:
                    callback = get_callback(callback)
                    _make_call(loop_state, callback, batch)

            loop_state.step_logs = Logs()
            for s, (schedule, callbacks) in enumerate(tasks_):
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback, batch)
                        if loop_state.stop_iteration:
                            break
                if loop_state.stop_iteration:
                    break

            if loop_state.step_logs:
                loop_state.step_logs["elapsed"] = elapsed_initial
                loop_state.history.append(loop_state.step_logs)

            if loop_state.stop_iteration or (
                stop_period is not None and loop_state.elapsed >= stop_period
            ):
                break

        # call on_end on last batch
        if on_end is not None:
            loop_state.step_logs = Logs()
            if not isinstance(on_end, list):
                on_end = [on_end]
            for callback in on_end:
                callback = get_callback(callback)
                _make_call(loop_state, callback, batch)

    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            print("\nStopping loop...")
        else:
            raise

    return loop_state.state, loop_state.history, loop_state.elapsed


def _make_call(loop_state: LoopState, callback: Callback, batch: Batch):
    try:
        loop_state.elapsed = loop_state.elapsed.update_time()
        callback_outputs = inject(
            callback, loop_state.state, batch, loop_state.elapsed, loop_state
        )
        if callback_outputs is not None:
            logs, state = callback_outputs
            if logs is not None:
                loop_state.step_logs.merge(logs)
                loop_state.accumulated_logs.merge(logs)
            if state is not None:
                loop_state.state = state
    except BaseException as e:
        raise type(e)(f"Error in callback {callback}: {e}") from e
