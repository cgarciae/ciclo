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
from flax.training.train_state import TrainState

import ciclo

batch_size = 32

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(batch_size).repeat().prefetch(1)


# Define model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def train_step(state: TrainState, batch):
    inputs, labels = batch["image"], batch["label"]

    # update the model's state
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
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
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
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
            f"logdir/{Path(__file__).stem}/{int(time())}",
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
