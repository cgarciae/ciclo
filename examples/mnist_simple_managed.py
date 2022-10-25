import ciclo
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from ciclo import managed

strategy = ciclo.get_strategy("data_parallel")
batch_size = strategy.lift_batch_size(32)

# load the MNIST dataset
ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)

# Model
class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x


# State
model = Linear()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = managed.ManagedState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
    strategy=strategy,
)

# Train step
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


# Training loop
total_samples = 32 * 1_000
total_steps = total_samples // batch_size

state, history, _ = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=ciclo.at(total_steps)),
        ],
    },
    stop=ciclo.at(samples=total_samples),
)

print(history[0].subkey_value("logits").shape)
