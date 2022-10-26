__version__ = "0.1.0"


from . import managed
from .api import Elapsed, Period, LogsLike
from .utils import at, callback, elapse, logs, history
from .callbacks import (
    checkpoint,
    early_stopping,
    inner_loop,
    keras_bar,
    tqdm_bar,
    wandb_logger,
    noop,
    LoopCallbackBase,
)
from .loops import loop, LoopState
from .schedules import every, piecewise
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
