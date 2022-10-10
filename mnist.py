import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import tensorflow as tf
from flax.training import train_state
import ciclo
from clu.metrics import Collection, Accuracy, Average
from flax import struct

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(32).repeat().prefetch(1)
ds_valid: tf.data.Dataset = tfds.load("mnist", split="test")
ds_valid = ds_valid.batch(32, drop_remainder=True).prefetch(1)

# Define model
class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
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
def train_step(state: TrainState, batch, _):
    print("jitting train_step")

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = state.metrics.update(loss=loss, logits=logits, labels=batch["label"])
    return None, state.replace(metrics=metrics)


@jax.jit
def compute_metrics(state: TrainState, batch, _):
    print("jitting compute_metrics")
    return state.metrics.compute(), None


@jax.jit
def eval_step(state: TrainState, batch, _):
    print("jitting eval_step")
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metrics = state.metrics.update(loss=loss, logits=logits, labels=batch["label"])
    logs = metrics.compute()
    return logs, state.replace(metrics=metrics)


def reset_metrics(state: TrainState, batch, _):
    return None, state.replace(metrics=state.metrics.empty())


# Initialize state
model = CNN()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    metrics=Metrics.empty(),
)

# training loop
total_steps = 10_000
eval_steps = 1_000
log_steps = 200
eval_loop = ciclo.inner_loop(
    "valid",
    lambda state: ciclo.loop(
        state,
        ds_valid.as_numpy_iterator(),
        {ciclo.every(1): [eval_step]},
        on_start=[reset_metrics],
    ),
)
state, loop = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): [train_step],
        ciclo.every(log_steps): [compute_metrics, reset_metrics],
        ciclo.every(eval_steps): [eval_loop],
        ciclo.every(1): [ciclo.keras_bar(total=total_steps, always_stateful=True)],
    },
    stop=total_steps,
)
