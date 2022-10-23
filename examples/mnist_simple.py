# %%
from time import time

import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training.train_state import TrainState
import numpy as np

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(32).repeat().prefetch(1)

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

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    logs = {"loss": loss, "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)}
    return {"metrics": logs}, state


# Initialize state
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
)

# training loop
total_steps = 10_000

state, history, *_ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=total_steps),
        ],
        ciclo.every(1000): ciclo.checkpoint(f"logdir/mnist_simple/{int(time())}"),
    },
    stop=total_steps,
)

# %%
# plot the training history
steps, loss, accuracy = history.collect("steps", "loss", "accuracy")
steps, loss, accuracy = np.array(steps), np.array(loss), np.array(accuracy)
# calculate the moving average
loss_ma = np.convolve(loss, np.ones(32) / 32, mode="valid")
accuracy_ma = np.convolve(accuracy, np.ones(32) / 32, mode="valid")
steps_ma = steps[15:-16]

# use subplots to plot loss and accuracy on the same figure
fig, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss, label="loss")
axs[0].plot(steps_ma, loss_ma, label="loss (avg)")
axs[0].legend()
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy, label="accuracy")
axs[1].plot(steps_ma, accuracy_ma, label="accuracy (avg)")
axs[1].legend()
axs[1].set_title("Accuracy")
plt.show()

# %%
