from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    overload,
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node

from ciclo.timetracking import Elapsed
from ciclo.types import CluMetric, LogPath, LogsLike


class Logs(LogsLike):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # copy mutable values
        for k, v in self.items():
            if isinstance(v, MutableMapping):
                self[k] = dict(v)

    @property
    def updates(self) -> Optional[LogsLike]:
        # raise error if accessed
        raise AttributeError("updates is a write-only attribute")

    @updates.setter
    def updates(self, updates: Optional[LogsLike]) -> None:
        if updates is not None:
            self.merge(updates)

    # ----------------------------------
    # logger behavior
    # ----------------------------------
    def add_entry(self, collection: str, name: str, value: Any) -> "Logs":
        if collection not in self:
            self[collection] = {}

        mapping = self[collection]

        if not isinstance(mapping, MutableMapping):
            raise ValueError(
                f"Invalid collection '{collection}' of type '{type(mapping).__name__}', must be a MutableMapping"
            )

        mapping[name] = value

        return self

    def add_entries(self, collection: str, values: Dict[str, Any]) -> "Logs":
        for name, value in values.items():
            self.add_entry(collection, name, value)

        return self

    def add_metric(self, name: str, value: Any) -> "Logs":
        if isinstance(value, CluMetric):
            raise ValueError(
                f"Metric '{name}' is a clu Metric which is stateful, use 'add_stateful_metric' instead"
            )
        return self.add_entry("metrics", name, value)

    def add_metrics(self, metrics: Dict[str, Any]):
        for name, value in metrics.items():
            self.add_metric(name, value)

    def add_stateful_metric(self, name: str, value: Any) -> "Logs":
        return self.add_entry("stateful_metrics", name, value)

    def add_stateful_metrics(self, metrics: Dict[str, Any]) -> "Logs":
        for name, value in metrics.items():
            self.add_stateful_metric(name, value)
        return self

    def add_loss(self, name: str, value: Any, *, add_metric: bool = False) -> "Logs":
        self.add_entry("losses", name, value)
        if add_metric:
            self.add_metric(name, value)
        return self

    def add_losses(
        self, losses: Dict[str, Any], *, add_metrics: bool = False
    ) -> "Logs":
        for name, value in losses.items():
            self.add_loss(name, value, add_metric=add_metrics)
        return self

    def add_output(self, name: str, value: Any, *, per_sample: bool = True) -> "Logs":
        collection = "per_sample_outputs" if per_sample else "outputs"
        self.add_entry(collection, name, value)

    # ----------------------------------
    # history behavior
    # ----------------------------------

    def entry_value(self, name: str) -> Any:
        path = self.entry_path(name)
        if path is None:
            raise KeyError(f"Key {name} not found in logs.")
        collection, name = path
        return self[collection][name]

    def entry_path(self, name: str) -> Optional[LogPath]:
        path = name.split(".")

        if len(path) == 1:
            name = path[0]
            collection = self.entry_collection(name)
            if collection is None:
                return None
        elif len(path) == 2:
            collection, name = path
        else:
            raise ValueError(f"Got more than 2 levels of nesting in key '{name}'")

        return collection, name

    def entry_collection(self, name: str) -> Optional[str]:
        collections = [col for col in self if name in self[col]]

        if len(collections) == 0:
            return None
        elif len(collections) == 1:
            return collections[0]
        else:
            raise ValueError(
                f"Found multiple collections for name '{name}' : {collections}. "
                "Use `collection.name` syntax."
            )

    def merge(self, collection_updates: LogsLike):
        for collection, updates in collection_updates.items():
            if not isinstance(updates, Mapping):
                raise ValueError(
                    f"Invalide value '{updates}' for collection '{collection}', value must be a Mapping"
                )
            if collection in self:
                entries = self[collection]
                if isinstance(entries, MutableMapping):
                    entries.update(updates)
                elif isinstance(entries, Mapping):
                    if type(entries) != type(updates):
                        raise ValueError(
                            f"Cannot merge collections of different types: {type(entries)} and {type(updates)}"
                        )
                    self[collection] = updates
                else:
                    raise ValueError(
                        f"Invalid collection '{collection}' of type '{type(entries).__name__}', must be a Mapping "
                        "or MutableMapping"
                    )
            else:
                # NOTE: we copy mutable mappings to avoid side effects
                if isinstance(updates, Dict):
                    self[collection] = updates.copy()
                elif isinstance(updates, MutableMapping):
                    self[collection] = dict(updates)
                else:
                    self[collection] = updates


def _logs_tree_flatten(self):
    return (dict(self),), None


def _logs_tree_unflatten(aux_data, children):
    self = Logs(children[0])
    return self


register_pytree_node(Logs, _logs_tree_flatten, _logs_tree_unflatten)

# ----------------------------------
# history
# ----------------------------------


class History(List[Logs]):
    @overload
    def collect(self, key: str) -> List[Any]:
        ...

    @overload
    def collect(self, key: str, *keys: str) -> Tuple[List[Any], ...]:
        ...

    def collect(
        self, key: str, *keys: str
    ) -> Union[Logs, List[Any], Tuple[List[Any], ...]]:
        keys = (key,) + keys
        outputs = tuple([] for _ in keys)
        for logs in self:
            paths = [logs.entry_path(key) for key in keys]
            if all(path is not None for path in paths):
                for i, path in enumerate(paths):
                    assert path is not None
                    collection, key = path
                    outputs[i].append(logs[collection][key])

        return outputs if len(keys) > 1 else outputs[0]

    def commit(self, elapsed: Elapsed, logs: LogsLike):
        # convert JAX arrays to numpy arrays to free memory
        logs = jax.tree_map(
            lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, logs
        )
        if not isinstance(logs, Logs):
            logs = Logs(logs)
        logs["elapsed"] = elapsed
        self.append(logs)


__all__ = ["Logs", "History"]
