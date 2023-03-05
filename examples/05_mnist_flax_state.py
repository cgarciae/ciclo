# %%
from pathlib import Path
from time import time

import flax.linen as nn
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import ciclo

batch_size = 32
total_samples = 32 * 100
total_steps = total_samples // batch_size
steps_per_epoch = total_steps // 10
test_steps = 10

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.map(lambda x: (x["image"], x["label"]))
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
ds_test: tf.data.Dataset = tfds.load("mnist", split="test")
ds_test = ds_test.map(lambda x: (x["image"], x["label"]))  # .take(10)
ds_test = ds_test.batch(32, drop_remainder=True).prefetch(1)


# Define model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


# Initialize state
model = Linear()
state = ciclo.create_flax_state(
    model,
    inputs=jnp.empty((1, 28, 28, 1)),
    tx=optax.adamw(1e-3),
    losses={"loss": jm.losses.Crossentropy()},
    metrics={"accuracy": jm.metrics.Accuracy()},
    strategy="jit",
)
state, history, _ = ciclo.train_loop(
    state,
    ds_train.as_numpy_iterator(),
    callbacks=[
        ciclo.keras_bar(total=total_steps),
        ciclo.checkpoint(
            f"logdir/{Path(__file__).stem}/{int(time())}",
            monitor="accuracy_test",
            mode="max",
        ),
    ],
    test_dataset=lambda: ds_test.as_numpy_iterator(),
    epoch_duration=steps_per_epoch,
    test_duration=test_steps,
    stop=total_steps,
)

# %%

steps, avg_loss, accuracy, avg_loss_test, accuracy_test = history.collect(
    "steps", "avg_loss", "accuracy", "avg_loss_test", "accuracy_test"
)

_, axs = plt.subplots(1, 2)
axs[0].plot(steps, avg_loss, label="train")
axs[0].plot(steps, avg_loss_test, label="test")
axs[0].legend()
axs[0].set_title("Avg Loss")
axs[1].plot(steps, accuracy, label="train")
axs[1].plot(steps, accuracy_test, label="test")
axs[1].legend()
axs[1].set_title("Accuracy")
plt.show()
