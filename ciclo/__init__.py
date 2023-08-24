__version__ = "0.1.9"


from . import managed
from .callbacks import checkpoint as checkpoint
from .callbacks import early_stopping as early_stopping
from .callbacks import inner_loop as inner_loop
from .callbacks import keras_bar as keras_bar
from .callbacks import noop as noop
from .callbacks import tqdm_bar as tqdm_bar
from .callbacks import wandb_logger as wandb_logger
from .logging import History, Logs
from .loops.common import ON_EPOCH_END as on_epoch_end
from .loops.common import ON_RESET_STEP as on_reset_step
from .loops.common import ON_TEST_BATCH_BEGIN as on_test_batch_begin
from .loops.common import ON_TEST_BATCH_END as on_test_batch_end
from .loops.common import ON_TEST_BEGIN as on_test_begin
from .loops.common import ON_TEST_END as on_test_end
from .loops.common import ON_TEST_STEP as on_test_step
from .loops.common import ON_TRAIN_BATCH_BEGIN as on_train_batch_begin
from .loops.common import ON_TRAIN_BATCH_END as on_train_batch_end
from .loops.common import ON_TRAIN_BEGIN as on_train_begin
from .loops.common import ON_TRAIN_END as on_train_end
from .loops.common import ON_TRAIN_STEP as on_train_step
from .loops.common import predict_loop as predict_loop
from .loops.common import test_loop as test_loop
from .loops.common import train_loop as train_loop
from .loops.loop import LoopCallbackBase, LoopElement, LoopState, loop
from .schedules import after as after
from .schedules import always as always
from .schedules import every as every
from .schedules import never as never
from .schedules import piecewise as piecewise
from .states.flax_state import FlaxState as FlaxState
from .states.flax_state import create_flax_state as create_flax_state
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
from .timetracking import Elapsed, Period, elapse
from .utils import at, callback, history, inject, logs

__all__ = [
    "FlaxState",
    "History",
    "LoopCallbackBase",
    "LoopElement",
    "LoopState",
    "Logs",
    "Strategy",
    "always",
    "at",
    "callback",
    "checkpoint",
    "create_flax_state",
    "early_stopping",
    "elapse",
    "every",
    "after",
    "get_strategy",
    "history",
    "inner_loop",
    "inject",
    "keras_bar",
    "logs",
    "managed",
    "never",
    "noop",
    "on_epoch_end",
    "on_reset_step",
    "on_test_batch_begin",
    "on_test_batch_end",
    "on_test_begin",
    "on_test_end",
    "on_test_step",
    "on_train_batch_begin",
    "on_train_batch_end",
    "on_train_begin",
    "on_train_end",
    "on_train_step",
    "piecewise",
    "predict_loop",
    "register_strategy",
    "test_loop",
    "tqdm_bar",
    "train_loop",
    "wandb_logger",
    "loop",
    "JIT",
    "DataParallel",
    "Eager",
    "Elapsed",
    "Period",
]
