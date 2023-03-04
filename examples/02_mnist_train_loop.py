# %%
from pathlib import Path
from time import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from clu.metrics import Accuracy, Average, Collection
from flax import struct
from flax.training import train_state

import ciclo

batch_size = 32

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
ds_test: tf.data.Dataset = tfds.load("mnist", split="test")
ds_test = ds_test.batch(32, drop_remainder=True).prefetch(1)


# Define model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


@struct.dataclass
class Metrics(Collection):
    loss: Average.from_output("loss")
    accuracy: Accuracy

    def update(self, **kwargs) -> "Metrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


class TrainState(train_state.TrainState):
    metrics: Metrics


@jax.jit
def train_step(state: TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = state.metrics.update(loss=loss, logits=logits, labels=batch["label"])
    logs = ciclo.logs()
    logs.add_stateful_metrics(**metrics.compute())
    return logs, state.replace(metrics=metrics)


@jax.jit
def test_step(state: TrainState, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metrics = state.metrics.update(loss=loss, logits=logits, labels=batch["label"])
    logs = ciclo.logs()
    logs.add_stateful_metrics(**metrics.compute())
    return logs, state.replace(metrics=metrics)


def reset_step(state: TrainState):
    return state.replace(metrics=state.metrics.empty())


# Initialize state
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    metrics=Metrics.empty(),
)

# training loop
total_samples = 32 * 100
total_steps = total_samples // batch_size
eval_steps = total_steps // 10
log_steps = total_steps // 50


state, history, _ = ciclo.train_loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.on_train_step: [train_step],
        ciclo.on_test_step: [test_step],
        ciclo.on_reset_step: [reset_step],
    },
    callbacks=[
        ciclo.checkpoint(
            f"logdir/{Path(__file__).stem}/{int(time())}",
            monitor="accuracy_test",
            mode="max",
        ),
        ciclo.keras_bar(total=total_steps),
    ],
    test_dataset=lambda: ds_test.as_numpy_iterator(),
    epoch_duration=eval_steps,
    stop=total_steps,
)

# %%

steps, loss, accuracy = history.collect("steps", "loss", "accuracy")
steps_test, loss_test, accuracy_test = history.collect(
    "steps", "loss_test", "accuracy_test"
)

_, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss, label="train")
axs[0].plot(steps_test, loss_test, label="test")
axs[0].legend()
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy, label="train")
axs[1].plot(steps_test, accuracy_test, label="test")
axs[1].legend()
axs[1].set_title("Accuracy")
plt.show()
