from dataclasses import dataclass
from typing import Generic, List, Optional, Union
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
    get_batch_size,
    get_callback,
)


def loop(
    state: S,
    dataset,
    tasks: InputTasks,
    *,
    stop: Union[Period, int, None] = None,
    on_start: Optional[List[InputCallable]] = None,
    on_end: Optional[List[InputCallable]] = None,
    history: Optional[History] = None,
    elapsed: Optional[Elapsed] = None,
    catch_keyboard_interrupt: bool = True,
) -> LoopOutput[S]:

    if isinstance(stop, int):
        stop = Period(steps=stop)

    if history is None:
        history = History()

    if elapsed is None:
        elapsed = Elapsed.create()

    loop_state = LoopState(
        state=state,
        history=history,
        elapsed=elapsed,
        step_logs={},
        accumulated_logs={},
    )

    tasks_ = {
        schedule: [get_callback(f) for f in callbacks]
        for schedule, callbacks in tasks.items()
    }

    batch = None

    try:
        for i, batch in enumerate(dataset):
            logs_elapsed = loop_state.elapsed.update(get_batch_size(batch))
            loop_state.elapsed = logs_elapsed

            # call on_start on first batch
            if i == 0 and on_start is not None:
                for callback in on_start:
                    callback = get_callback(callback)
                    _make_call(loop_state, callback, batch)

            loop_state.step_logs = {}
            for schedule, callbacks in tasks_.items():
                if schedule(loop_state.elapsed):
                    for callback in callbacks:
                        _make_call(loop_state, callback, batch)
                        if loop_state.stop_iteration:
                            break
                if loop_state.stop_iteration:
                    break

            if loop_state.step_logs:
                loop_state.step_logs["elapsed"] = logs_elapsed
                loop_state.history.append(loop_state.step_logs)

            if loop_state.stop_iteration or (
                stop is not None and stop >= loop_state.elapsed
            ):
                break

        # call on_end on last batch
        if on_end is not None:
            loop_state.step_logs = {}
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
    loop_state.elapsed = loop_state.elapsed.update_time()
    callback_outputs = callback(loop_state.state, batch, loop_state.elapsed, loop_state)
    if callback_outputs is not None:
        logs, state = callback_outputs
        if logs is not None:
            loop_state.step_logs.update(logs)
            loop_state.accumulated_logs.update(logs)
        if state is not None:
            loop_state.state = state
