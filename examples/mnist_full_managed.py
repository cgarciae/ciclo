import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from ciclo import managed
from clu.metrics import Accuracy, Average


# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.shuffle(1024).batch(32 * 8).repeat().prefetch(1)
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


AverageLoss = Average.from_output("loss")


class ManagedState(managed.ManagedState):
    accuracy: Accuracy
    loss: AverageLoss


def loss_fn(state: ManagedState, batch, _):
    inputs, labels = batch["image"], batch["label"]

    logits = state.apply_fn({"params": state.params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()

    managed.log("accuracy", Accuracy.from_model_output(logits=logits, labels=labels))
    managed.log("loss", AverageLoss.from_model_output(loss=loss))

    return loss, state


train_step = managed.train_step(loss_fn)
eval_step = managed.eval_step(loss_fn)


@managed.step
def reset_metrics(state: ManagedState, batch, _):
    return None, state.replace(
        accuracy=Accuracy.empty(),
        loss=AverageLoss.empty(),
    )


# Initialize state
model = CNN()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = ManagedState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    accuracy=Accuracy.empty(),
    loss=AverageLoss.empty(),
    strategy="jit",
)

# training loop
total_samples = 32 * 10 * 10_000
eval_samples = 32 * 1 * 10_000

state, history, *_ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(steps=1): [train_step],
        ciclo.every(samples=eval_samples): [
            reset_metrics,
            ciclo.inner_loop(
                "valid",
                lambda state: ciclo.loop(
                    state,
                    ds_valid.as_numpy_iterator(),
                    {ciclo.every(steps=1): [eval_step]},
                ),
            ),
            ciclo.checkpoint(
                "logdir/mnist_full_managed",
                only_best_for="accuracy_valid",
                minimize=False,
            ),
            reset_metrics,
        ],
        ciclo.every(steps=1): [
            ciclo.keras_bar(total=ciclo.at(samples=total_samples), always_stateful=True)
        ],
    },
    stop=ciclo.at(samples=total_samples),
)
