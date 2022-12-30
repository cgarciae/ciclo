<style>
.md-grid {
    max-width: 100%;
    /* add 5% padding left and right*/
    padding: 0 5%;
}
</style>

# Getting Started


```bash
pip install ciclo
```

```python
import ciclo
```

## Example Setup

#### Create a dataset

Import MNIST using TensorFlow Datasets

```python
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 32

ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
```

#### Create a model
Define model architecture and a `create_state` function to initialize the model's weights and optimizer.

```python
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import flax.linen as nn

class Linear(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x

def create_state():
    model = Linear()
    variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adamw(1e-3),
    )
```

#### Define training step
Create a `train_step` function that takes a batch of data and updates the model's weights. Use Ciclo's `log` helper to log metrics.

```python
@jax.jit
def train_step(state: TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    # log metrics
    logs = ciclo.logs()
    logs.add_metric("loss", loss)
    logs.add_metric("accuracy", jnp.mean(jnp.argmax(logits, -1) == batch["label"]))
    return logs, state
```

## Training loop

=== "manual"

    ```python
    from time import time

    total_steps = 5_000
    call_checkpoint = ciclo.every(steps=1000) # Schedule
    checkpoint = ciclo.checkpoint(f"logdir/getting_started/{int(time())}") # Callback
    keras_bar = ciclo.keras_bar(total=total_steps, interval=0.4) # Callback
    end_period = ciclo.at(total_steps) # Period

    state = create_state() # initial state
    history = ciclo.history() # History
    # (Elapsed, Batch)
    for elapsed, batch in ciclo.elapse(ds_train.as_numpy_iterator()):
        logs = ciclo.logs() # Logs
        # update logs and state
        logs.updates, state = train_step(state, batch)
        # periodically checkpoint state
        if call_checkpoint(elapsed):
            checkpoint(elapsed, state) # save state

        keras_bar(elapsed, logs) # update progress bar
        history.commit(elapsed, logs) # commit logs to history
        # stop training when total_steps is reached
        if elapsed >= end_period:
            break
    ```

=== "loop"

    ```python
    total_steps = 5_000
    state = create_state()

    state, history, elapsed = ciclo.loop(
        state,
        ds_train.as_numpy_iterator(),
        {
            # Schedule: [Callback]
            ciclo.every(1): [train_step],
            ciclo.every(steps=1000): [
                ciclo.checkpoint(f"logdir/getting_started/{int(time())}")
            ],
            ciclo.every(1): [ciclo.keras_bar(total=total_steps, interval=0.4)],
        },
        stop=total_steps,
    )
    ```
Where:

* `every` is a simple periodic `Schedule`
* `checkpoint` is a `Callback` that serializes the `state`
* `keras_bar` is a `Callback` that displays a progress bar using log information
* `at` is a `Period`, it can determine when a certain amount of steps/samples/time has passed
* `logs` is a custom dictionary of type `Logs: Dict[str, Mapping[str, Any]]`, it contains helper methods to log values and merge logs from multiple callbacks.
* `history` is a custom list of type `History: List[Logs]`, it stores the logs from each iteration of the loop and contains helper methods to commit logs and collect values.
* `elapse` is a generator that yields an `(elapsed, batch)` tuple for each batch in the dataset. `Elapsed` is a pytree structure that contains information about the current iteration of the loop, number of samples seen, and amount of time passed since the start of the loop.

#### Loop function callbacks

```python
```

#### Collecting values from history
```python
import matplotlib.pyplot as plt

# collect metric values
steps, loss, accuracy = history.collect("steps", "loss", "accuracy")

fig, axs = plt.subplots(1, 2)
axs[0].plot(steps, loss)
axs[0].set_title("Loss")
axs[1].plot(steps, accuracy)
axs[1].set_title("Accuracy")
plt.show()
```

#### Syntactic sugar

If you have a single callback for a given schedule you can pass it directly. Furthermore, `CallbackBase` instances (all callbacks in Ciclo implement this) can be passed directly using the `**` operator if you want to run them on every iteration.

| Syntax | Expansion |
| --- | --- |
| `schedule: callback` | `schedule: [callback]` |
| `**callback` | `every(1): [callback]` |

The previous example can be rewritten as:

```python
total_steps = 5_000
state = create_state()

state, history, elapsed = ciclo.loop(
    state,
    ds_train.as_numpy_iterator(),
    {
        ciclo.every(1): train_step,
        ciclo.every(steps=1000): ciclo.checkpoint(
            f"logdir/getting_started/{int(time())}"
        ),
        **ciclo.keras_bar(total=total_steps, interval=0.4),
    },
    stop=total_steps,
)
```
