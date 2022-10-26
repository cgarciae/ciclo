from curses import keyname
from time import time
import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from ciclo import managed
from clu.metrics import Accuracy, Average

print(jax.local_devices())

strategy = ciclo.get_strategy("jit")
batch_size = strategy.lift_batch_size(32)

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
ds_valid: tf.data.Dataset = tfds.load("mnist", split="test")
ds_valid = ds_valid.batch(batch_size, drop_remainder=True).prefetch(1)

# Define model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


AverageLoss = Average.from_output("loss")


class ManagedState(managed.ManagedState):
    accuracy: Accuracy
    loss: AverageLoss
    key: jax.random.KeyArray


def loss_fn(state: ManagedState, batch):
    inputs, labels = batch["image"], batch["label"]
    logits = state.apply_fn({"params": state.params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    logs = ciclo.logs()
    logs.add_loss("loss", loss)
    logs.add_metric(
        "accuracy", Accuracy.from_model_output(logits=logits, labels=labels)
    )
    logs.add_metric("loss", AverageLoss.from_model_output(loss=loss))
    return logs, state


train_step = managed.train_step(loss_fn)
eval_step = managed.step(loss_fn)


@managed.step
def reset_metrics(state: ManagedState):
    return None, state.replace(
        accuracy=Accuracy.empty(),
        loss=AverageLoss.empty(),
    )


# Initialize state
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = ManagedState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    accuracy=Accuracy.empty(),
    loss=AverageLoss.empty(),
    key=jax.random.PRNGKey(0),
    strategy=strategy,
)

# training loop
total_samples = 32 * 10_000
total_steps = total_samples // batch_size
eval_steps = total_steps // 10

state, history, *_ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        **train_step,
        ciclo.every(eval_steps, steps_offset=1): [
            reset_metrics,
            ciclo.inner_loop(
                "valid",
                lambda state: ciclo.loop(
                    state,
                    ds_valid.as_numpy_iterator(),
                    {ciclo.every(steps=1): eval_step},
                ),
            ),
            ciclo.checkpoint(
                f"logdir/mnist_full_managed/{int(time())}",
                monitor="accuracy_valid",
                mode="max",
                keep=3,
            ),
            ciclo.early_stopping(
                monitor="accuracy_valid",
                mode="max",
                patience=eval_steps * 2,
            ),
            reset_metrics,
        ],
        **ciclo.keras_bar(total=total_steps),
    },
    stop=ciclo.at(samples=total_samples),
)
