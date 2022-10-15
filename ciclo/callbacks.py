# ---------------------------------------
# callbacks
# ---------------------------------------


from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import os
from typing import Callable, Optional, Union, overload
from tqdm import tqdm
from pkbar import Kbar
from flax.training import checkpoints as flax_checkpoints

from ciclo.api import (
    S,
    Batch,
    Elapsed,
    LoopOutput,
    LoopState,
    Period,
    get_batch_size,
    is_scalar,
)


class OptimizationMode(Enum):
    min = auto()
    max = auto()


class inner_loop:
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
        state, log_history, *_ = self.loop_fn(state)
        logs = log_history[-1] if len(log_history) > 0 else {}
        logs = {
            k + f"_{self.name}" if self.name else k: v
            for k, v in logs.items()
            if not isinstance(v, Elapsed)
        }

        return logs, state if self.output_state else None


class checkpoint:
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
        if mode not in OptimizationMode:
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {list(OptimizationMode)}"
            )
        else:
            self.mode = OptimizationMode(mode)

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
            overwrite = True
            if self.monitor not in loop_state.accumulated_logs:
                raise ValueError(f"Monitored value '{self.monitor}' not found in logs")

            value = loop_state.accumulated_logs[self.monitor]
            if (
                self._best is None
                or (self.minimize and value < self._best)
                or (not self.minimize and value > self._best)
            ):
                self._best = value
                step_or_metric = value
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


class early_stopping:
    def __init__(
        self,
        monitor: str,
        patience: Union[int, Period],
        min_delta: float = 0,
        mode: Union[str, OptimizationMode] = "min",
        baseline: Optional[float] = None,
        restore_best_weights: bool = False,
    ):
        if mode not in OptimizationMode:
            raise ValueError(
                f"Invalid mode: {mode}, expected one of {list(OptimizationMode)}"
            )
        else:
            self.mode = OptimizationMode(mode)

        self.monitor = monitor
        self.patience = patience if isinstance(patience, Period) else Period(patience)
        self.min_delta = min_delta
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.minimize = self.mode == OptimizationMode.min
        self._best = baseline
        self._best_state = None
        self._elapsed_start: Optional[Elapsed] = None

    def __call__(self, state, batch: Batch, elapsed: Elapsed, loop_state: LoopState):
        if self.monitor not in loop_state.accumulated_logs:
            raise ValueError(f"Monitored value '{self.monitor}' not found in logs")

        if self._elapsed_start is None:
            self._elapsed_start = Elapsed.create()

        value = loop_state.accumulated_logs[self.monitor]
        if (
            self._best is None
            or (self.minimize and value < self._best)
            or (not self.minimize and value > self._best)
        ):
            self._best = value
            self._best_state = state
            self._elapsed_start = elapsed

        if self.patience >= elapsed - self._elapsed_start:
            if self.restore_best_weights and self._best_state is not None:
                state = self._best_state
            loop_state.stop_iteration = True

        return None, state


class tqdm_bar:
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
            total = Period(steps=total)

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

    def __call__(self, state, batch, elapsed: Elapsed, loop):

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


class keras_bar:
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
            total = Period(steps=total)

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

        self.bar.update(
            current,
            values=[(k, v) for k, v in loop_state.step_logs.items() if is_scalar(v)],
        )
