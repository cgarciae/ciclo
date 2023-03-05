import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import optax
from flax import struct, traverse_util
from flax.core import FrozenDict
from flax.core.scope import CollectionFilter, DenyList, FrozenVariableDict

import ciclo
from ciclo import managed
from ciclo.strategies import Strategy
from ciclo.timetracking import Elapsed
from ciclo.types import CluMetric, MetricLike

Pytree = Any
Loss = Callable[..., jax.Array]
Index = Union[int, str]
KeyPath = Tuple[Index, ...]
M = TypeVar("M", bound=nn.Module)


def static_field(**kwargs):
    return struct.field(pytree_node=False, **kwargs)


def node_field(**kwargs):
    return struct.field(pytree_node=True, **kwargs)


class Metrics(struct.PyTreeNode, MetricLike):
    metrics: Dict[str, Union[CluMetric, MetricLike]]

    @classmethod
    def create(cls, metrics: Dict[str, Any]):
        # validate metrics
        for name, metric in metrics.items():
            if not isinstance(metric, (CluMetric, MetricLike)):
                raise ValueError(f"Invalid metric type for {name}: {type(metric)}")

        return cls(
            metrics=metrics,
        )

    def reset(self):
        metrics = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, MetricLike):
                metrics[name] = metric.reset()
            else:
                metrics[name] = metric.empty()

        return self.replace(metrics=metrics)

    def update(self, **kwargs):
        metrics = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, MetricLike):
                metrics[name] = metric.update(**kwargs)
            else:
                metrics[name] = metric.merge(metric.from_model_output(**kwargs))

        return self.replace(metrics=metrics)

    def batch_updates(self, **kwargs) -> "Metrics":
        metrics = {}
        for name, metric in self.metrics.items():
            if isinstance(metric, MetricLike):
                metrics[name] = metric.batch_updates(**kwargs)
            else:
                metrics[name] = metric.from_model_output(**kwargs)

        return self.replace(metrics=metrics)

    def merge(self, other: "Metrics") -> "Metrics":
        metrics = {}
        for name, metric in self.metrics.items():
            other_metric = other.metrics[name]
            if type(metric) != type(other_metric):
                raise ValueError(
                    f"Metric {name} has different types: {type(metric)} and "
                    f"{type(other_metric)}"
                )

            if isinstance(metric, MetricLike):
                assert isinstance(other_metric, MetricLike)
                metrics[name] = metric.merge(other_metric)
            else:
                assert isinstance(other_metric, CluMetric)
                metrics[name] = metric.merge(other_metric)

        return self.replace(metrics=metrics)

    def compute(self):
        logs = {}
        for name, metric in self.metrics.items():
            logs[name] = metric.compute()

        return logs

    def reduce(self):
        metrics = {}
        for name, metric in self.metrics.items():
            metrics[name] = metric.reduce()

        return self.replace(metrics=metrics)


class FlaxState(Generic[M], managed.ManagedState):
    """State for Flax models."""

    key: jax.random.KeyArray
    batch_stats: Optional[FrozenVariableDict]
    variables: FrozenVariableDict
    stateful_metrics: Metrics
    stateless_metrics: Dict[str, Callable[..., jax.Array]] = static_field()
    losses: Dict[str, Loss] = static_field()
    module_fn: Callable[[], M] = static_field()

    mutable_train: CollectionFilter = static_field()
    mutable_eval: CollectionFilter = static_field()
    rngs_train: Sequence[str] = static_field()
    rngs_eval: Sequence[str] = static_field()
    method_train: Union[str, Callable[..., Any]] = static_field()
    method_eval: Union[str, Callable[..., Any]] = static_field()
    logs_full_path: bool = static_field()
    keep_collections: Tuple[str, ...] = static_field()

    @property
    def module(self) -> M:
        return self.module_fn()

    def apply(
        self,
        key: jax.random.KeyArray,
        inputs: Any,
        training: bool,
    ):
        method = self.method_train if training else self.method_eval
        method = _unbounded_method(self.module, method)
        mutable = self.mutable_train if training else self.mutable_eval
        rng_names = self.rngs_train if training else self.rngs_eval
        arg_names = _function_argument_names(method)
        args, kwargs = _split_args_kwargs(inputs)

        if (
            arg_names is not None
            and "training" in arg_names
            and "training" not in kwargs
        ):
            kwargs["training"] = training

        rngs = _split_into_collection(key, rng_names)

        variables = self.variables
        if self.params is not None:
            variables = variables.copy({"params": self.params})
        if self.batch_stats is not None:
            variables = variables.copy({"batch_stats": self.batch_stats})

        apply_output = self.module.apply(
            variables,
            *args,
            rngs=rngs,
            mutable=mutable,
            method=method,
            **kwargs,
        )

        if mutable is False:
            outputs = apply_output
        else:
            outputs, variable_updates = apply_output
            variables = variables.copy(variable_updates)

        return outputs, self.replace(
            variables=variables,
            params=variables.get("params", None),
            batch_stats=variables.get("batch_stats", None),
        )

    @managed.train_step
    def train_step(self, batch, elapsed: Elapsed):
        logs, state = self._step(batch, elapsed, training=True)
        return logs, state

    @managed.step
    def test_step(self, batch, elapsed: Elapsed):
        logs, state = self._step(batch, elapsed, training=False)
        return logs, state

    @managed.step
    def predict_step(self, batch, elapsed: Elapsed):
        inputs, labels, sample_weight = unpack_x_y_sample_weight(batch)
        key = jax.random.fold_in(self.key, elapsed.steps)
        preds, _ = self.apply(key, inputs, training=False)

        if isinstance(preds, jax.Array):
            preds = {"preds": preds}
        else:
            preds = traverse_util.flatten_dict(preds, sep="/")

        logs = ciclo.logs()
        for name, value in preds.items():
            logs.add_output(name, value)

        return logs, None

    @managed.step
    def reset_step(self):
        stateful_metrics = self.stateful_metrics.reset()
        return self.replace(stateful_metrics=stateful_metrics)

    def _step(self, batch, elapsed: Elapsed, training: bool):
        inputs, labels, sample_weight = unpack_x_y_sample_weight(batch)

        key = jax.random.fold_in(self.key, elapsed.steps)
        seq_key, pred_key = jax.random.split(key, 2)
        preds, self = self.apply(pred_key, inputs, training=training)
        variables = self.variables

        if not isinstance(labels, Mapping):
            label_args = dict(target=labels)
        else:
            label_args = labels

        if not isinstance(inputs, Mapping):
            input_args = dict(inputs=inputs)
        else:
            input_args = inputs

        arguments = dict(
            preds=preds,
            module=self,
            params=self.params,
            batch_stats=self.batch_stats,
            key_seq=KeySeq(seq_key),
            sample_weight=sample_weight,
            **label_args,
            **input_args,
        )

        logs = ciclo.logs()

        for name, loss in self.losses.items():
            logs.add_loss(name, loss(**arguments))

        if "losses" in variables:
            aux_losses = flatten_names_unique(
                variables["losses"], only_last=not self.logs_full_path
            )
            for name, loss in aux_losses.items():
                loss = jnp.asarray(loss)
                logs.add_loss(name, loss)

        if "losses" not in logs:
            raise ValueError("No losses were added to the logs.")

        # Add losses to metric arguments
        for name, loss in logs.losses.items():
            if name not in arguments:
                arguments[name] = loss

        for name, metric in self.stateless_metrics.items():
            if callable(metric):
                logs.add_metric(name, metric(**arguments))
            else:
                raise TypeError(
                    f"Unexpected metric type {type(metric)} for metric {name}."
                )

        logs.add_stateful_metric(
            "stateful_metrics", self.stateful_metrics.batch_updates(**arguments)
        )

        if "metrics" in variables:
            aux_metrics = flatten_names_unique(
                variables["metrics"], only_last=not self.logs_full_path
            )
            for name, metric in aux_metrics.items():
                metric = jnp.asarray(metric)
                logs.add_metric(name, metric)

        # Filter variables
        variables = FrozenDict(
            {
                collection: variables[collection]
                for collection in variables
                if collection in self.keep_collections
            }
        )

        return logs, self.replace(variables=variables)

    # ---------------------------------------------------------------------------------
    # HighLevel API helpers
    # ---------------------------------------------------------------------------------


def init_module(
    module: nn.Module,
    inputs: Any,
    key: jax.random.KeyArray,
    mutable_init: CollectionFilter,
    rngs_init: Sequence[str],
    method_init: Union[str, Callable[..., Any]],
    init_training_value: bool,
):
    method = _unbounded_method(module, method_init)
    arg_names = _function_argument_names(method)
    args, kwargs = _split_args_kwargs(inputs)

    if arg_names is not None and "training" in arg_names and "training" not in kwargs:
        kwargs["training"] = init_training_value

    rngs = _split_into_collection(key, rngs_init)

    variables = module.init(
        rngs,
        *args,
        mutable=mutable_init,
        method=method,
        **kwargs,
    )
    return variables


def create_flax_state(
    module: M,
    inputs: Any,
    tx: optax.GradientTransformation,
    key: Optional[jax.random.KeyArray] = None,
    losses: Optional[Dict[str, Loss]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    mutable_init: CollectionFilter = True,
    mutable_train: CollectionFilter = DenyList(["params"]),
    mutable_eval: CollectionFilter = DenyList(["params", "batch_stats"]),
    rngs_init: Sequence[str] = ("params", "dropout", "default"),
    rngs_train: Sequence[str] = ("dropout", "default"),
    rngs_eval: Sequence[str] = ("default",),
    method_init: Union[str, Callable[..., Any]] = "__call__",
    method_train: Union[str, Callable[..., Any]] = "__call__",
    method_eval: Union[str, Callable[..., Any]] = "__call__",
    init_training_value: bool = True,
    logs_full_path: bool = False,
    keep_collections: Sequence[str] = (),
    strategy: Union[Strategy, str] = "jit",
) -> FlaxState[M]:
    if key is None:
        key = jax.random.PRNGKey(0)
    if losses is None:
        losses = {}
    if metrics is None:
        metrics = {}

    keep_collections = tuple(keep_collections)

    variables = init_module(
        module=module,
        inputs=inputs,
        key=key,
        mutable_init=mutable_init,
        rngs_init=rngs_init,
        method_init=method_init,
        init_training_value=init_training_value,
    )
    if "params" in variables:
        variables, params = variables.pop("params")
    else:
        params = None

    if "batch_stats" in variables:
        variables, batch_stats = variables.pop("batch_stats")
    else:
        batch_stats = None

    # Create metrics
    stateful_metrics = {}
    stateless_metrics = {}

    # Split metrics into stateful and stateless metrics
    for name, metric in metrics.items():
        if isinstance(metric, (CluMetric, MetricLike)):
            stateful_metrics[name] = metric
        elif issubclass(metric, CluMetric):
            stateful_metrics[name] = metric.empty()
        elif callable(metric):
            stateless_metrics[name] = metric
        else:
            raise ValueError(f"Invalid metric type for {name}: {type(metric)}")

    # Add stateful metrics for auxiliary losses
    if "losses" in variables:
        aux_losses = flatten_names_unique(
            variables["losses"], only_last=not logs_full_path
        )
        for name in aux_losses.keys():
            avg_name = f"avg_{name}"
            if avg_name not in stateful_metrics:
                stateful_metrics[avg_name] = jm.metrics.Mean().from_argument(name)

    # Add stateful metrics for losses
    for name in losses.keys():
        avg_name = f"avg_{name}"
        if avg_name not in stateful_metrics:
            stateful_metrics[avg_name] = jm.metrics.Mean().from_argument(name)

    # Filter variables
    variables = FrozenDict(
        {
            collection: variables[collection]
            for collection in variables
            if collection in keep_collections
        }
    )

    state = FlaxState.create(
        # TrainState
        apply_fn=module.apply,
        params=params,
        tx=tx,
        # ManagedState
        strategy=strategy,
        # FlaxState
        key=key,
        batch_stats=batch_stats,
        variables=variables,
        losses=losses,
        stateful_metrics=Metrics.create(stateful_metrics),
        stateless_metrics=stateless_metrics,
        module_fn=lambda: module,
        mutable_train=mutable_train,
        mutable_eval=mutable_eval,
        rngs_train=rngs_train,
        rngs_eval=rngs_eval,
        method_train=method_train,
        method_eval=method_eval,
        logs_full_path=logs_full_path,
        keep_collections=keep_collections,
    )
    return state


def _function_argument_names(f: Callable) -> Optional[List[str]]:
    """
    Returns:
        A list of keyword argument names or None if variable keyword arguments (`**kwargs`) are present.
    """
    kwarg_names = []

    for k, v in inspect.signature(f).parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            return None

        kwarg_names.append(k)

    return kwarg_names


def _split_args_kwargs(
    value: Any,
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    if isinstance(value, tuple):
        return value, {}
    elif isinstance(value, list):
        return tuple(value), {}
    elif isinstance(value, Mapping):
        return (), dict(value)
    else:
        return (value,), {}


def _split_into_collection(
    key: jnp.ndarray,
    collection: Sequence[str],
) -> Dict[str, jnp.ndarray]:
    """
    Split the key into the specified rngs.
    """

    keys = jax.random.split(key, len(collection))

    keys_collection = {col: keys[i] for i, col in enumerate(collection)}

    return keys_collection


def _unbounded_method(
    module: nn.Module,
    method: Union[str, Callable[..., Any]],
) -> Callable[..., Any]:
    if isinstance(method, str):
        return getattr(type(module), method)
    return method


def unpack_x_y_sample_weight(data):
    """Unpacks user-provided data tuple."""
    if not isinstance(data, tuple):
        return (data, None, None)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])

    raise ValueError("Data not understood.")


def flatten_names_unique(inputs: Any, only_last: bool = False) -> Dict[str, Any]:
    names: Set[str] = set()

    if only_last:
        return {
            get_unique_name(names, str(path[-1])): value
            for path, value in _flatten_names((), inputs)
        }
    else:
        return {
            get_unique_name(names, "/".join(map(str, path))): value
            for path, value in _flatten_names((), inputs)
        }


def get_unique_name(
    names: Set[str],
    name: str,
):
    if name in names:
        i = 1
        while f"{name}_{i}" in names:
            i += 1

        name = f"{name}_{i}"

    names.add(name)
    return name


def _flatten_names(path: KeyPath, inputs: Any) -> Iterable[Tuple[KeyPath, Any]]:
    if isinstance(inputs, (Tuple, List)):
        for i, value in enumerate(inputs):
            yield from _flatten_names(path, value)
    elif isinstance(inputs, Mapping):
        for name, value in inputs.items():
            yield from _flatten_names(path + (name,), value)
    else:
        yield (path, inputs)


class KeySeq:
    """KeySeq is simple module that can produce a sequence of PRNGKeys.

    Example:
    ```python
    class Dropout(Module):
        rng: KeySeq()

        def __init__(self, rate: float):
            self.next_key = KeySeq()
            ...

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            key = self.next_key()
            mask = jax.random.bernoulli(key, 1.0 - self.rate)
            ...
    ```
    """

    key: jax.random.KeyArray
    index: int

    def __init__(
        self,
        key: Union[jax.random.KeyArray, int],
    ):
        """
        Arguments:
            key: An optional PRNGKey to initialize the KeySeq with.
        """

        self.key = jax.random.PRNGKey(key) if isinstance(key, int) else key
        self.index = 0

    def next(self) -> jax.random.KeyArray:
        """
        Return a new PRNGKey and updates the internal rng state.

        Returns:
            A PRNGKey.
        """

        key = jax.random.fold_in(self.key, self.index)
        self.index += 1
        return key

    __next__ = next
