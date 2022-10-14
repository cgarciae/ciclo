from functools import partial
import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from ciclo import managed

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(32 * 8).repeat().prefetch(1)

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


@managed.train_step
def train_step(state: managed.ManagedState, batch, _):
    inputs, labels = batch["image"], batch["label"]

    logits = state.apply_fn({"params": state.params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    managed.log("accuracy", jnp.mean(jnp.argmax(logits, -1) == labels))

    return loss, state


# Initialize state
model = CNN()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = managed.ManagedState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    strategy="jit",
)

# training loop
total_samples = 32 * 10 * 10_000

state, history, *_ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=ciclo.at(samples=total_samples)),
        ],
    },
    stop=ciclo.at(samples=total_samples),
)
