from typing import Any, Dict, Optional

import jax
from flax.core import tracers
from jax.tree_util import register_pytree_node
from ciclo.api import Metric


class CallbackLogs(Dict[str, Dict[str, Any]]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_level = tracers.trace_level(tracers.current_trace())

    def add(self, collection: str, key: str, value: Any) -> "CallbackLogs":

        log_trace_level = tracers.trace_level(tracers.current_trace())
        if log_trace_level != self._trace_level:
            raise ValueError("log must be called from the same trace level")

        if collection not in self:
            self[collection] = {}

        self[collection][key] = value

        return self

    def add_many(self, collection: str, values: Dict[str, Any]) -> "CallbackLogs":

        for key, value in values.items():
            self.add(collection, key, value)

        return self

    def add_metric(
        self, key: str, value: Any, *, stateful: bool = False
    ) -> "CallbackLogs":
        if isinstance(value, Metric):
            stateful = True
        collection = "metrics" if not stateful else "stateful_metrics"
        return self.add(collection, key, value)

    def add_metrics(self, metrics: Dict[str, Any], *, stateful: bool = False):
        for key, value in metrics.items():
            self.add_metric(key, value, stateful=stateful)

    def add_loss(self, key: str, value: Any, *, add_metric: bool = False):
        self.add("losses", key, value)
        if add_metric:
            self.add_metric(key, value)

    def add_losses(self, losses: Dict[str, Any], *, add_metrics: bool = False):
        for key, value in losses.items():
            self.add_loss(key, value, add_metric=add_metrics)


def logs(**kwargs):
    return CallbackLogs(kwargs)


# -----------------------
# pytree definition
# -----------------------
def _tree_flatten(self):
    return (dict(self),), self._trace_level


def _tree_unflatten(aux_data, children):
    self = CallbackLogs(children[0])
    self._trace_level = aux_data
    return self


register_pytree_node(CallbackLogs, _tree_flatten, _tree_unflatten)
