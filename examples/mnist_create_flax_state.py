# %%
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
total_samples = 32 * 10000
total_steps = total_samples // batch_size
eval_steps = total_steps // 10
log_steps = total_steps // 50

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.map(lambda x: (x["image"], x["label"]))
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
ds_test: tf.data.Dataset = tfds.load("mnist", split="test")
ds_test = ds_test.map(lambda x: (x["image"], x["label"]))
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
    tx=optax.adam(1e-3),
    losses={
        "loss": lambda preds, target, **kw: optax.softmax_cross_entropy_with_integer_labels(
            logits=preds, labels=target
        ).mean()
    },
    metrics={
        "accuracy": lambda preds, target, **kw: Accuracy.from_model_output(
            logits=preds, labels=target
        ).compute()
    },
)
state, history, _ = ciclo.train_loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.on_epoch_end: [
            ciclo.checkpoint(
                f"logdir/mnist_fit/{int(time())}", monitor="accuracy_test", mode="max"
            )
        ],
        ciclo.every(1): ciclo.keras_bar(total=total_steps),
    },
    test_dataset=lambda: ds_test.as_numpy_iterator(),
    test_every=eval_steps,
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

# %%
