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
from clu.metrics import Accuracy, Average, Collection
from flax import struct
from flax.training import train_state
from ciclo import managed

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


@struct.dataclass
class Metrics(Collection):
    loss: Average.from_output("loss")
    accuracy: Accuracy

    def update(self, **kwargs) -> "Metrics":
        updates = self.single_from_model_output(**kwargs)
        return self.merge(updates)


AverageLoss = Average.from_output("loss")


class ManagedState(managed.ManagedState):
    accuracy: Accuracy
    loss: AverageLoss


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
    strategy=strategy,
)

# training loop
total_samples = 32 * 10_000
total_steps = total_samples // batch_size
eval_steps = total_steps // 10

checkpoint = ciclo.checkpoint(
    f"logdir/mnist_full/{int(time())}", monitor="accuracy_valid", mode="max"
)
early_stopping = ciclo.early_stopping(
    monitor="accuracy_valid", mode="max", patience=eval_steps * 2
)
is_time_to_eval = ciclo.every(eval_steps)
keras_bar = ciclo.keras_bar(total=total_steps)
end_period = ciclo.at(total_steps)

history = ciclo.history()
stop_iteration = False
for elapsed, batch in ciclo.elapse(ds_train.as_numpy_iterator()):
    logs = ciclo.logs()
    logs.updates, state = train_step(state, batch)

    if is_time_to_eval(elapsed):
        # --------------------
        # eval loop
        # --------------------
        eval_logs: ciclo.LogsLike = {}
        _, eval_state = reset_metrics(state)
        for eval_batch in ds_valid.as_numpy_iterator():
            eval_logs, eval_state = eval_step(eval_state, eval_batch)
        # --------------------
        logs.updates = {
            c: {f"{k}_valid": v for k, v in c_logs.items()}
            for c, c_logs in eval_logs.items()
        }
        checkpoint(elapsed, state, logs)
        stop_iteration, state = early_stopping(elapsed, state, logs)

    keras_bar(elapsed, logs)
    history.commit_logs(elapsed, logs)

    if stop_iteration or elapsed >= end_period:
        break

steps, loss, accuracy = history.collect("steps", "stateful_metrics.loss", "accuracy")
steps_valid, loss_valid, accuracy_valid = history.collect(
    "steps", "stateful_metrics.loss_valid", "accuracy_valid"
)

_, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss, label="train")
axs[0].plot(steps_valid, loss_valid, label="valid")
axs[0].legend()
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy, label="train")
axs[1].plot(steps_valid, accuracy_valid, label="valid")
axs[1].legend()
axs[1].set_title("Accuracy")
plt.show()

# %%
