# %%
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from simple_pytree import Pytree, static_field

import ciclo

batch_size = 32

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(batch_size).repeat().prefetch(1)


# Define model
def linear_classifier(x):
    x = x / 255.0
    x = x.reshape((x.shape[0], -1))  # flatten
    x = hk.Linear(10)(x)
    return x


@dataclass
class State(Pytree):
    params: Any
    opt_state: optax.OptState
    tx: optax.GradientTransformation = static_field()
    apply_fn: Callable = static_field()

    @classmethod
    def create(
        cls, *, apply_fn: Callable, params: Any, tx: optax.GradientTransformation
    ):
        return cls(apply_fn=apply_fn, params=params, opt_state=tx.init(params), tx=tx)

    def apply_gradients(self, *, grads: Any):
        updates, opt_state = self.tx.update(grads, self.opt_state, self.params)
        params = optax.apply_updates(self.params, updates)
        return self.replace(params=params, opt_state=opt_state)


@jax.jit
def train_step(state: State, batch):
    inputs, labels = batch["image"], batch["label"]

    # update the model's state
    def loss_fn(params):
        logits = state.apply_fn(params, None, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)

    # add logs
    logs = ciclo.logs()
    logs.add_metric("loss", loss)
    logs.add_metric("accuracy", jnp.mean(jnp.argmax(logits, -1) == labels))

    return logs, state


# Initialize state
model = hk.transform(linear_classifier)
params = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = State.create(
    apply_fn=model.apply,
    params=params,
    tx=optax.adamw(1e-3),
)

# training loop
total_samples = 32 * 100
total_steps = total_samples // batch_size

state, history, _ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): train_step,
        ciclo.every(total_steps // 10): ciclo.checkpoint(
            f"logdir/haiku_mnist_simple/{int(time())}",
        ),
        **ciclo.keras_bar(total=total_steps),
    },
    stop=total_steps,
)

# %%
# plot the training history
steps, loss, accuracy = history.collect("steps", "loss", "accuracy")


fig, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss)
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy)
axs[1].set_title("Accuracy")
plt.show()

# %%
