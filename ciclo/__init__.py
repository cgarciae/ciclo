__version__ = "0.1.0"


from ciclo.types import LogsLike

from . import managed
from .callbacks import checkpoint as checkpoint
from .callbacks import early_stopping as early_stopping
from .callbacks import inner_loop as inner_loop
from .callbacks import keras_bar as keras_bar
from .callbacks import noop as noop
from .callbacks import tqdm_bar as tqdm_bar
from .callbacks import wandb_logger as wandb_logger
from .logging import History, Logs
from .loops.fit_loop import ON_EPOCH_END as on_epoch_end
from .loops.fit_loop import ON_TEST_BATCH_BEGIN as on_test_batch_begin
from .loops.fit_loop import ON_TEST_BATCH_END as on_test_batch_end
from .loops.fit_loop import ON_TEST_BEGIN as on_test_begin
from .loops.fit_loop import ON_TEST_END as on_test_end
from .loops.fit_loop import ON_TRAIN_BATCH_BEGIN as on_train_batch_begin
from .loops.fit_loop import ON_TRAIN_BATCH_END as on_train_batch_end
from .loops.fit_loop import ON_TRAIN_BEGIN as on_train_begin
from .loops.fit_loop import ON_TRAIN_END as on_train_end
from .loops.fit_loop import RESET_STEP as reset_step
from .loops.fit_loop import TEST_STEP as test_step
from .loops.fit_loop import TRAIN_STEP as train_step
from .loops.fit_loop import fit_loop as fit_loop
from .loops.loop import LoopCallbackBase, LoopElement, LoopState, loop
from .schedules import always as always
from .schedules import every as every
from .schedules import never as never
from .schedules import piecewise as piecewise
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
