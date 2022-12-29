__version__ = "0.1.0"


from . import managed
from .logging import Logs, History
from .timetracking import Period, Elapsed
from ciclo.types import LogsLike
from .utils import at, callback, logs, history, elapse, inject
from .callbacks import (
    checkpoint,
    early_stopping,
    inner_loop,
    keras_bar,
    tqdm_bar,
    wandb_logger,
    noop,
)
from .loops import loop, LoopState, LoopElement, LoopCallbackBase
from .schedules import every, piecewise
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
