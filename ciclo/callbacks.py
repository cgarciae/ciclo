# ---------------------------------------
# callbacks
# ---------------------------------------


import functools
import importlib.util
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple, Union, overload

from flax.training import checkpoints as flax_checkpoints
from pkbar import Kbar
from tqdm import tqdm

from ciclo.api import (
    Batch,
    CallbackBase,
    Elapsed,
    LoopOutput,
    LoopState,
    Period,
    S,
    register_adapter,
)
from ciclo.schedules import every
from ciclo.utils import get_batch_size, is_scalar


# import wandb Run
def _get_Run():
    if importlib.util.find_spec("wandb") is not None:
        from wandb.wandb_run import Run
    else:
        locals()["Run"] = Any
    return Run


Run = _get_Run()


class OptimizationMode(str, Enum):
    min = auto()
    max = auto()


class inner_loop(CallbackBase):
    @overload
    def __init__(
        self,
        name_or_loop_fn: str,
        maybe_loop_fn: Callable[[S], LoopOutput[S]],
        *,
        output_state: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        name_or_loop_fn: Callable[[S], LoopOutput[S]],
        *,
        output_state: bool = False,
    ):
        ...

    def __init__(
        self,
        name_or_loop_fn: Union[str, Callable[[S], LoopOutput[S]]],
        maybe_loop_fn: Optional[Callable[[S], LoopOutput[S]]] = None,
        *,
        output_state: bool = False,
    ):
        if isinstance(name_or_loop_fn, str):
            assert maybe_loop_fn is not None
            self.name = name_or_loop_fn
            self.loop_fn = maybe_loop_fn
        else:
            assert maybe_loop_fn is None
            self.name = None
            self.loop_fn = name_or_loop_fn
        self.output_state = output_state

    def __call__(self, state, batch, elapsed, outer_loop_state):
        state, log_history, _ = self.loop_fn(state)
        logs = log_history[-1] if len(log_history) > 0 else {}
        logs = {
            collection: {
                k + f"_{self.name}" if self.name else k: v
                for k, v in values.items()
                if not isinstance(v, Elapsed)
            }
            for collection, values in logs.items()
            if collection != "elapsed"
        }
        return logs, (state if self.output_state else None)


class checkpoint(CallbackBase):
    def __init__(
        self,
        ckpt_dir: Union[str, os.PathLike],
        prefix: str = "checkpoint_",
        keep: int = 1,
        overwrite: bool = False,
        keep_every_n_steps: Optional[int] = None,
        async_manager: Optional[flax_checkpoints.AsyncManager] = None,
        monitor: Optional[str] = None,
        mode: Union[str, OptimizationMode] = "min",
    ):
        if isinstance(mode, str):
            mode = OptimizationMode[mode]

        if mode not in OptimizationMode:
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {list(OptimizationMode)}"
            )
        else:
            self.mode = mode

        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.keep = keep
        self.overwrite = overwrite
        self.keep_every_n_steps = keep_every_n_steps
        self.async_manager = async_manager
        self.monitor = monitor
        self.minimize = self.mode == OptimizationMode.min
        self._best: Optional[float] = None

    def __call__(self, state, batch: Batch, elapsed: Elapsed, loop_state: LoopState):
        save_checkpoint = True
        step_or_metric = elapsed.steps
        overwrite = self.overwrite

        if self.monitor is not None:
            try:
                value = loop_state.accumulated_logs.subkey_value(self.monitor)
            except KeyError:
                raise ValueError(f"Monitored value '{self.monitor}' not found in logs")

            if (
                self._best is None
                or (self.minimize and value < self._best)
                or (not self.minimize and value > self._best)
            ):
                self._best = value
                step_or_metric = value if self.mode == OptimizationMode.max else -value
            else:
                save_checkpoint = False

        if save_checkpoint:
            flax_checkpoints.save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                target=state,
                step=step_or_metric,
                prefix=self.prefix,
                keep=self.keep,
                overwrite=overwrite,
                keep_every_n_steps=self.keep_every_n_steps,
                async_manager=self.async_manager,
            )


class early_stopping(CallbackBase[S]):
    def __init__(
        self,
        monitor: str,
        patience: Union[int, Period],
        min_delta: float = 0,
        mode: Union[str, OptimizationMode] = "min",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
    ):
        if isinstance(mode, str):
            mode = OptimizationMode[mode]

        if mode not in OptimizationMode:
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {list(OptimizationMode)}"
            )
        else:
            self.mode = mode

        self.monitor = monitor
        self.patience = (
            patience if isinstance(patience, Period) else Period.create(patience)
        )
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.minimize = self.mode == OptimizationMode.min
        self._best = baseline
        self._best_state = None
        self._elapsed_start: Optional[Elapsed] = None

    def __call__(
        self, state: S, batch: Batch, elapsed: Elapsed, loop_state: LoopState
    ) -> Tuple[None, S]:

        if self._elapsed_start is None:
            self._elapsed_start = Elapsed.create()

        try:
            value = loop_state.accumulated_logs.subkey_value(self.monitor)
        except KeyError:
            raise ValueError(f"Monitored value '{self.monitor}' not found in logs")

        if (
            self._best is None
            or (self.minimize and value < self._best)
            or (not self.minimize and value > self._best)
        ):
            self._best = value
            self._best_state = state
            self._elapsed_start = elapsed

        if elapsed - self._elapsed_start >= self.patience:
            if self.restore_best_weights and self._best_state is not None:
                state = self._best_state
            loop_state.stop_iteration = True

        return None, state


class tqdm_bar(CallbackBase[S]):
    def __init__(
        self,
        total: Union[Period, int, None] = None,
        desc=None,
        leave=True,
        file=None,
        ncols=None,
        mininterval=0.1,
        maxinterval=10.0,
        miniters=None,
        ascii=None,
        disable=False,
        unit_scale=False,
        dynamic_ncols=False,
        smoothing=0.3,
        bar_format=None,
        initial=0,
        position=None,
        postfix=None,
        unit_divisor=1000,
        write_bytes=None,
        lock_args=None,
        nrows=None,
        colour=None,
        delay=0,
        gui=False,
        **kwargs,
    ):

        if isinstance(total, int):
            total = Period.create(steps=total)

        if total is not None:
            if total.steps is not None:
                bar_total = total.steps
                unit = "steps"
            elif total.samples is not None:
                bar_total = total.samples
                unit = "samples"
            elif total.time is not None:
                bar_total = total.time
                unit = "s"
                unit_scale = True
            elif total.date is not None:
                total.time = total.date - datetime.now().timestamp()
                bar_total = total.time
                unit = "s"
                unit_scale = True
            else:
                raise ValueError("Invalid total")
        else:
            bar_total = None
            unit = "it"

        self.total = total
        self.prev_step: Optional[int] = None
        self.prev_samples: Optional[int] = None
        self.prev_time: Optional[float] = None
        self.bar_total = bar_total
        self.bar = tqdm(
            desc=desc,
            total=bar_total,
            leave=leave,
            file=file,
            ncols=ncols,
            mininterval=mininterval,
            maxinterval=maxinterval,
            miniters=miniters,
            ascii=ascii,
            disable=disable,
            unit=unit,
            unit_scale=unit_scale,
            dynamic_ncols=dynamic_ncols,
            smoothing=smoothing,
            bar_format=bar_format,
            initial=initial,
            position=position,
            postfix=postfix,
            unit_divisor=unit_divisor,
            write_bytes=write_bytes,
            lock_args=lock_args,
            nrows=nrows,
            colour=colour,
            delay=delay,
            gui=gui,
            **kwargs,
        )

    def __call__(self, state, batch, elapsed: Elapsed, loop) -> None:

        if self.total is None or self.total.steps is not None:
            if self.prev_step is None:
                self.prev_step = elapsed.steps - 1
            self.bar.update(elapsed.steps - self.prev_step)
            self.prev_step = elapsed.steps
        elif self.total.samples is not None:
            if self.prev_samples is None:
                self.prev_samples = elapsed.samples - get_batch_size(batch)
            self.bar.update(elapsed.samples - self.prev_samples)
            self.prev_samples = elapsed.samples
        elif self.total.time is not None:
            if self.prev_time is None:
                self.prev_time = elapsed._date_start
            self.bar.update(elapsed.date - self.prev_time)
            self.prev_time = elapsed.date
        else:
            raise ValueError("Invalid total")


class keras_bar(CallbackBase[S]):
    def __init__(
        self,
        total: Union[Period, int, None] = None,
        epoch=None,
        num_epochs=None,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        always_stateful=False,
        unit_name="step",
    ):
        if isinstance(total, int):
            total = Period.create(steps=total)

        if total is not None:
            if total.steps is not None:
                bar_total = total.steps
                unit_name = "step"
            elif total.samples is not None:
                bar_total = total.samples
                unit_name = "sample"
            elif total.time is not None:
                bar_total = total.time
                unit_name = "s"
                unit_scale = True
            elif total.date is not None:
                total.time = total.date - datetime.now().timestamp()
                bar_total = total.time
                unit_name = "s"
                unit_scale = True
            else:
                raise ValueError("Invalid total")
        else:
            bar_total = None
            unit_name = "it"

        self.total = total
        self.prev_step: Optional[int] = None
        self.prev_samples: Optional[int] = None
        self.prev_time: Optional[float] = None
        self.bar_total = bar_total
        self.bar = Kbar(
            bar_total,
            epoch=epoch,
            num_epochs=num_epochs,
            width=width,
            verbose=verbose,
            interval=interval,
            stateful_metrics=stateful_metrics,
            always_stateful=always_stateful,
            unit_name=unit_name,
        )

    def __call__(self, state, batch, elapsed: Elapsed, loop_state: LoopState):
        if self.total is None or self.total.steps is not None:
            current = elapsed.steps
        elif self.total.samples is not None:
            current = elapsed.samples
        elif self.total.time is not None:
            current = elapsed.time
        else:
            raise ValueError("Invalid total")

        metrics: Dict[str, Any] = {}
        if "stateful_metrics" in loop_state.step_logs:
            stateful_metrics = loop_state.step_logs["stateful_metrics"]
            self.bar.stateful_metrics.update(stateful_metrics.keys())
            metrics.update(stateful_metrics)

        if "metrics" in loop_state.step_logs:
            metrics.update(loop_state.step_logs["metrics"])

        if metrics:
            self.bar.update(
                current,
                values=[(k, v) for k, v in metrics.items() if is_scalar(v)],
            )


class wandb_logger(CallbackBase[S]):
    def __init__(self, run: Run):
        self.run = run

    def __call__(self, state, batch, elapsed: Elapsed, loop_state: LoopState):
        data = {k: v for k, v in loop_state.step_logs.items() if is_scalar(v)}
        if len(data) > 0:
            self.run.log(data, step=elapsed.steps)


class NoOp(CallbackBase[S]):
    def __call__(self, state, batch, elapsed: Elapsed, loop_state: LoopState):
        pass


noop = NoOp()

# -------------------------------------------
# Adapters
# -------------------------------------------

if importlib.util.find_spec("clu") is not None:
    from clu.periodic_actions import PeriodicAction

    @dataclass(frozen=True)
    class PeriodicActionCallback(CallbackBase[S]):
        action: PeriodicAction

        def __call__(self, state, batch, elapsed: Elapsed, loop_state: LoopState):
            self.action(elapsed.steps, t=elapsed.date)

    @functools.partial(register_adapter, cls=PeriodicAction)
    def periodic_action_adapter(f: PeriodicAction):
        return PeriodicActionCallback(f)
