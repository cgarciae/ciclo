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
from .loops.fit_loop import fit_loop as fit_loop
from .loops.loop import LoopCallbackBase, LoopElement, LoopState, loop
from .schedules import always, every, never, piecewise
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
