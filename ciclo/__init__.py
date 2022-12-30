__version__ = "0.1.0"


from ciclo.types import LogsLike

from . import managed
from .callbacks import (
    checkpoint,
    early_stopping,
    inner_loop,
    keras_bar,
    noop,
    tqdm_bar,
    wandb_logger,
)
from .logging import History, Logs
from .loops import LoopCallbackBase, LoopElement, LoopState, loop
from .schedules import every, piecewise
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
from .timetracking import Elapsed, Period
from .utils import at, callback, elapse, history, inject, logs
