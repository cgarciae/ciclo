__version__ = "0.1.0"


from . import managed
from .api import Elapsed, Period, at
from .callbacks import inner_loop, keras_bar, tqdm_bar, checkpoint, early_stopping
from .loops import loop
from .schedules import every
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
