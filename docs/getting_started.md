<style>
.md-grid {
    max-width: 100%;
    /* add 5% padding left and right*/
    padding: 0 5%;
}
</style>

# Getting Started

`ciclo` is a functional library for training loops in JAX. It provides a set of utilities and abstractions to build complex training loops with any JAX framework. `ciclo` defines a set of building blocks that naturally compose together so they can scale up to build higher-level abstractions.

In this guide we will learn how to use `ciclo` to train a simple model on the MNIST dataset.

### Installation and usage

You can install `ciclo` from pypi:
```bash
pip install ciclo
```

To use `ciclo` you tipically the main module as it exposes most of the API:

```python
import ciclo
```

### Create a dataset

To begin with our example, first we will load the MNIST using TensorFlow Datasets and create a `tf.data.Dataset` that we will use to train our model.

```python
import tensorflow as tf
import tensorflow_datasets as tfds

batch_size = 32

ds_train: tf.data.Dataset = tfds.load("mnist", split="train", shuffle_files=True)
ds_train = ds_train.repeat().shuffle(1024).batch(batch_size).prefetch(1)
```

### Create a model
Next we will create a linear classifier. Here we will be using Flax but you can use any JAX framework.

```python
import flax.linen as nn

class Classifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        return x
```

### Define the state
Ciclo promotes grouping all the state such as parameters, optimizer state, random state, etc, into a single pytree structure we refer to as `state`. While state could be something as simple as a `dict`, since Flax also uses this pattern and provides a `TrainState` abstraction we will be using in this here.

Next we will instantiated the model, initialized its parameters, and constructed our state object using `TrainState`. `TrainState` will internally initialize the optimizer state using `optax`.

```python
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

model = Classifier()
variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, 28, 28, 1)))
state = TrainState.create(
    apply_fn=model.apply,
    params=variables["params"],
    tx=optax.adamw(1e-3),
)
```
If you are using other frameworks like Haiku or Equinox, feel free to leverage `TrainState` nevertheless as its API shoud be compatible with any JAX framework.

### Define a training procedure
For the training procedure we will create a `train_step` function that takes the `state` and a `batch` of data with the goal of updating the `state` and capturing some `logs`.

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

A couple of things are happening here:

* `.apply_gradients` is a method from `TrainState` that is updating the parameters and optimizer state using the gradients.
* We are using `ciclo.logs` to create a dictionary structure that will hold the metrics.

As we will see in the following section, the specific signature of `train_step` chosen here guarantees that its compatible with `ciclo.loop` and other utilities.

### The training loop

To define training loops Ciclo provides utilities like schedules, callbacks, logging, and time tracking. These can be used either with `ciclo.loop` or manually.

=== "loop"

    `ciclo.loop` offers a compact way of defining training loops by mapping schedules to callbacks. It takes an initial state and a dataset, and iterates over the dataset while updating the state with the callbacks until a stop condition is met.

    ```python
    total_steps = 5_000

    state, history, elapsed = ciclo.loop(
        state, # State
        ds_train.as_numpy_iterator(), # Iterable[Batch]
        {
            # Schedule: List[Callback]
            ciclo.every(1): [train_step],
            ciclo.every(steps=1000): [
                ciclo.checkpoint(f"logdir/model")
            ],
            ciclo.every(1): [ciclo.keras_bar(total=total_steps)],
        },
        stop=ciclo.at(total_steps), # Period
    )
    ```

    On every step, `loop` will go over all the schedules in order and check whether the current step matches the schedule, if it does, it will call the callbacks associated with the schedule. Each callback can update the state, add logs, or merely produce side effects (e.g. logging or checkpointing). `loop` will continue iterating until the stop condition is met or the dataset terminates.

=== "manual"

    Manual iteration is a little bit more verbose but you get full control.

    ```python
    total_steps = 5_000

    call_checkpoint = ciclo.every(steps=1000) # Schedule
    checkpoint = ciclo.checkpoint(f"logdir/model") # Callback
    keras_bar = ciclo.keras_bar(total=total_steps) # Callback
    end_period = ciclo.at(total_steps) # Period
    history = ciclo.history() # History
    # (Elapsed, Batch)
    for elapsed, batch in ciclo.elapse(ds_train.as_numpy_iterator(), stop=end_period):
        logs = ciclo.logs() # Logs
        # update logs and state
        logs.updates, state = train_step(state, batch)
        # periodically checkpoint state
        if call_checkpoint(elapsed):
            checkpoint(elapsed, state) # serialize state

        keras_bar(elapsed, logs) # update progress bar
        history.commit(elapsed, logs) # commit logs to history
    ```

    Here we are showing a somewhat equivalent version to what `loop` produces, however it could be simplified a bit.

#### Schedules
Schedules are used to determine when certain events should occure. In this example, we are using `ciclo.every` which can periodically trigger based on a given number of steps, samples seen, or time passed. In this case, we are calling `train_step` on every step and `ciclo.checkpoint` every 1000 steps.

#### Callbacks
Callbacks are used to perform useful actions such as logging, checkpointing, early stopping, etc. In this example, we are using `ciclo.keras_bar` to display a progress bar and `ciclo.checkpoint` to serialize the state. `loop` accepts as a calback any object that implements the `LoopCallback` protocol or functions of the form:

```python
# variadic: accepts between 0 and 4 arguments
def callback(
    [state, batch, elapsed, loop_state]
) -> (logs | None, state | None) | logs | state | None
```

In our example, `train_step` is a valid callback because it accepts `state` and `batch` as inputs, and returns `logs` and `state` as outputs, thus matching one of the possible signatures. If your function doesn't match any of the signatures, `loop` wont raise immediatly but you will get a runtime error later when the callback is called.

#### Logging
Ciclo provides two basic logging utilities, `ciclo.logs` and `ciclo.history`. `logs` is a custom nested dictionary structure that can be used to store metrics and other information, it has some convenience properties and methods to facilitate organizing information (e.g. `add_metric`), merging logs between different callbacks (`merge`, `updates`) and lookup nested values (e.g. `entry_value`). `history` is a list of `logs` that can be used to collect logs. `loop` will automatically `merge` the logs from all callbacks and `commit` them to the `history` at the end each step.

#### Time tracking
Ciclo provides a `ciclo.elapse` generator that yields an `(elapsed, batch)` tuple for each batch in the dataset. `elapsed` is a pytree structure that contains information about the current iteration of the loop, number of samples seen, and amount of time passed since the start of the loop. Ciclo also provides the `ciclo.at` utility to create a `Period` object that can be used to determine when a certain amount of steps/samples/time has passed.

### Collecting values from history

After training, we can collect the values from the `history` and plot them. For this we can use the `collect` method that takes some key names as inputs and returns the list of values for each key for timesteps in which they **all** appear in the logs.

```python
import matplotlib.pyplot as plt

# collect metric values
steps, losses, accuracies = history.collect("steps", "loss", "accuracy")

fig, axs = plt.subplots(1, 2)
axs[0].plot(steps, losses)
axs[0].set_title("Loss")
axs[1].plot(steps, accuracies)
axs[1].set_title("Accuracy")
plt.show()
```

Even though `logs` are nested dictionary structures, `collect` lets us specify just the names of the inner most keys and it will search for them, in case there is a clash and they appear more than once you will get an error, to fix it you will have to specify the full path instead e.g. `"metrics.accuracy"`. Note that the keys from `elapsed` are available in the logs as well, this is how we are able to collect the `steps`.
