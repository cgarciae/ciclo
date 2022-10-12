__version__ = "0.1.0"


from . import managed
from .loops import (
    Elapsed,
    Loop,
    Period,
    at,
    every,
    inner_loop,
    keras_bar,
    loop,
    tqdm_bar,
)
from .strategies import (
    JIT,
    DataParallel,
    Eager,
    Strategy,
    get_strategy,
    register_strategy,
)
