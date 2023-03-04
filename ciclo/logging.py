from typing import Any, Dict, List, Optional, Tuple, Type, Union, overload

import jax
import numpy as np
from jax.tree_util import register_pytree_node

from ciclo.timetracking import Elapsed
from ciclo.types import CluMetric, LogPath

LogsLike = Dict[str, Dict[str, Any]]
Collection = str
Entry = str


class Entries(Dict[Entry, Any]):
    def __getattr__(self, entry: Entry) -> Any:
        if entry in self:
            return self[entry]
        else:
            raise AttributeError(f"Entry '{entry}' not found in logs.")


class Logs(Dict[Collection, Entries]):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # copy values
        for k, v in self.items():
            self[k] = Entries(v)

    @property
    def updates(self) -> Optional[LogsLike]:
        # raise error if accessed
        raise AttributeError("updates is a write-only attribute")

    @updates.setter
    def updates(self, updates: Optional[LogsLike]) -> None:
        if updates is not None:
            self.merge(updates)

    def __getattr__(self, collection: Collection) -> Entries:
        if collection in self:
            return self[collection]
        else:
            raise AttributeError(f"Collection '{collection}' not found in logs.")

    # ----------------------------------
    # logger behavior
    # ----------------------------------
    def add_entry(self, collection: Collection, entry: Entry, value: Any) -> "Logs":
        if collection not in self:
            self[collection] = Entries()
        self[collection][entry] = value
        return self

    def add_entries(self, collection: Collection, **values: Any) -> "Logs":
        for name, value in values.items():
            self.add_entry(collection, name, value)

        return self

    def add_metric(self, entry: Entry, value: Any) -> "Logs":
        if isinstance(value, CluMetric):
            raise ValueError(
                f"Metric '{entry}' is a clu Metric which is "
                "stateful, use 'add_stateful_metric' instead"
            )
        return self.add_entry("metrics", entry, value)

    def add_metrics(self, **metrics: Any):
        return self.add_entries("metrics", **metrics)

    def add_stateful_metric(self, name: Entry, value: Any) -> "Logs":
        return self.add_entry("stateful_metrics", name, value)

    def add_stateful_metrics(self, **metrics: Any) -> "Logs":
        return self.add_entries("stateful_metrics", **metrics)

    def add_loss(self, name: Entry, value: Any, *, add_metric: bool = False) -> "Logs":
        self.add_entry("losses", name, value)
        if add_metric:
            self.add_metric(name, value)
        return self

    def add_losses(
        self,
        *,
        add_metrics: bool = False,
        **losses: Any,
    ) -> "Logs":
        self.add_entries("losses", **losses)
        if add_metrics:
            self.add_metrics(**losses)
        return self

    def add_output(self, name: Entry, value: Any) -> "Logs":
        return self.add_entry("outputs", name, value)

    def add_outputs(self, **outputs: Any) -> "Logs":
        return self.add_entries("outputs", **outputs)

    # ----------------------------------
    # history behavior
    # ----------------------------------

    def entry_value(self, name: Entry) -> Any:
        path = self.entry_path(name)
        if path is None:
            raise KeyError(f"Key {name} not found in logs.")
        collection, name = path
        return self[collection][name]

    def entry_path(self, name: Entry) -> Optional[LogPath]:
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

    def entry_collection(self, name: Entry) -> Optional[str]:
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
            if collection in self:
                self[collection].update(updates)
            else:
                self[collection] = Entries(updates)  # create copy


def _mapping_tree_flatten(tree: Dict[Any, Any]):
    return (dict(tree),), None


def _mapping_tree_unflatten(cls: Type[Dict[Any, Any]], children: Tuple[Dict[str, Any]]):
    tree = cls(children[0])
    return tree


register_pytree_node(
    Logs,
    _mapping_tree_flatten,
    lambda _, children: _mapping_tree_unflatten(Logs, children),
)
register_pytree_node(
    Entries,
    _mapping_tree_flatten,
    lambda _, children: _mapping_tree_unflatten(Entries, children),
)

# ----------------------------------
# history
# ----------------------------------


class History(List[Logs]):
    @overload
    def collect(self, key: str) -> List[Any]:
        ...

    @overload
    def collect(self, *keys: str) -> Tuple[List[Any], ...]:
        ...

    def collect(self, key: str, *keys: str) -> Union[List[Any], Tuple[List[Any], ...]]:
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
            lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, logs
        )
        logs["elapsed"] = elapsed.to_dict()

        self.append(Logs(logs))
