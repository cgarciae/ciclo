# %%
from time import time

import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from ciclo import managed
from flax.training.train_state import TrainState

strategy = ciclo.get_strategy("data_parallel")
batch_size = strategy.lift_batch_size(32)

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)

# Define model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


@managed.train_step
def train_step(state: managed.ManagedState, batch):
    inputs, labels = batch["image"], batch["label"]
    logits = state.apply_fn({"params": state.params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    logs = ciclo.logs()
    logs.add_loss("loss", loss, add_metric=True)
    logs.add_metric("accuracy", jnp.mean(jnp.argmax(logits, -1) == labels))
    logs.add_output("logits", logits)
    return logs, state


# Initialize state
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = managed.ManagedState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(3e-3),
    strategy=strategy,
)

# training loop
total_samples = 32 * 100
total_steps = total_samples // batch_size

checkpoint_schedule = ciclo.every(total_steps // 10)
checkpoint = ciclo.checkpoint(f"logdir/mnist_simple/{int(time())}")
keras_bar = ciclo.keras_bar(total=total_steps)
end_period = ciclo.at(total_steps)

history = ciclo.history()
for elapsed, batch in ciclo.elapse(ds_train.as_numpy_iterator()):
    logs, state = train_step(state, batch)
    if checkpoint_schedule(elapsed):
        checkpoint(elapsed, state)
    keras_bar(elapsed, logs)
    history.commit(elapsed, logs)

    if elapsed >= end_period:
        break

# plot the training history
steps, loss, accuracy = history.collect("steps", "metrics.loss", "accuracy")

# %%
# use subplots to plot loss and accuracy on the same figure
fig, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss)
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy)
axs[1].set_title("Accuracy")
plt.show()

# %%
