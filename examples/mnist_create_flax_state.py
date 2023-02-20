# %%
from pathlib import Path
from time import time

import clu.metrics
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
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


@struct.dataclass
class Accuracy(clu.metrics.Accuracy):
    @classmethod
    def from_model_output(cls, *, preds: jax.Array, target: jax.Array, **kwargs):
        return super().from_model_output(logits=preds, labels=target)


AverageLoss = clu.metrics.Average.from_output("loss")


def cross_entropy_loss(preds, target, **kwargs):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=preds, labels=target
    ).mean()


# Initialize state
model = Linear()
state = ciclo.create_flax_state(
    model,
    inputs=jnp.empty((1, 28, 28, 1)),
    tx=optax.adamw(1e-3),
    losses={"loss": cross_entropy_loss},
    metrics={"accuracy": Accuracy, "avg_loss": AverageLoss},
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
    test_every=eval_steps,
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
