[![codecov](https://codecov.io/gh/cgarciae/ciclo/branch/main/graph/badge.svg?token=3IKEUAU3C8)](https://codecov.io/gh/cgarciae/ciclo)

# 🌀 Ciclo
_A functional training loops library for JAX_

`ciclo` provides a set of utilities and abstractions to build complex training loops with any JAX framework. `ciclo` defines a set of building blocks that naturally compose together and scale up to build higher-level abstractions.

**Features**

✔️ Training utilities <br>
🌀 Loop language <br>
✔️ Predefined Loops <br>
🧪 Managed API (simplified training + parallelism support) [experimental] <br>
🧪 Framework support (predifined states) [experimental] <br>

## Status
Ciclo is still in early development, the API is subject to change, expect things to break. If you are interested in contributing, please reach out.

## Getting Started
To get started with Ciclo, you can install it via pip:

```python
pip install ciclo
```
Once you've installed Ciclo, you can import it into your Python script:

```python
import ciclo
```
  
### 🌀 Loop Language

The `loop` function in `ciclo` serves as a mini-language for defining training loops by composing functions. With the `tasks` dictionary, you can express the desired behavior of the loop as a composition of schedules and their corresponding callbacks.

To use the `loop` function, you first define your training steps as JAX functions, and then create a dictionary where the keys are schedules, and the values are lists of callbacks to execute at each scheduled interval.

```python
@jax.jit
def train_step(state, batch):
    ... # do JAX stuff to update state
    logs = ciclo.logs()
    logs.add_metric("accuracy", accuracy)
    logs.add_metric("loss", loss)
    return logs, state

total_steps = 100
state = create_state() # initial state

state, history, elapsed = ciclo.loop(
    state, # Pytree
    dataset, # Iterable[Batch]
    { # Schedule: List[Callback]
        ciclo.every(1): [train_step],
        ciclo.every(steps=10): [ciclo.checkpoint(f"logdir/model")],
        ciclo.every(1): [ciclo.keras_bar(total=total_steps)],
    },
    stop=total_steps,
)
```
```
80/100 [=====================>.....] - ETA: 42s - accuracy: 0.6148 - loss: 1.537120
```

At each iteration, callbacks can update the state and append new logs, the `loop` function returns the final state, the history of logs, and the elapsed time. Depending on the nature of each callback, the order in which they are executed may be very important e.g. `keras_bar` should always be last so that it can display the metrics produced by previous callbacks.

### 🧪 Predefined Loops

`ciclo` provides a set of predefined loops that you can use out of the box for common scenarios:

* `train_loop`: a training loop with an inner evaluation loop
* `test_loop`: an evaluation loop
* `predict_loop`: an inference loop

All of these loops are built on top of the `loop` function and extend the loop language with additional **named schedules** which run at specific points in the loop. They also provide a `callbacks` argument allows you to add callbacks without specifying a schedule, instead, if they contain an attribute that matches the name of a named schedule it will be automatically registered to that schedule. This also applies for methods of the `state` object.

Here's an example of how to use the `train_loop`:

```python
@jax.jit
def test_step(state, batch):
    ... # do JAX stuff
    logs = ciclo.logs()
    logs.add_metric("accuracy", accuracy)
    logs.add_metric("loss", loss)
    return logs, state

state, history, elapsed = ciclo.train_loop(
    state, # Pytree
    train_dataset, # Iterable[Batch]
    { # Schedule: List[Callback]
        ciclo.on_train_step: [train_step], # named schedules
        ciclo.on_test_step: [test_step], # named schedules
        ciclo.every(20): [some_callback], # regular schedules also supported
    },
    test_dataset=lambda: get_test_dataset(), # lazy test dataset definition
    epoch_duration=steps_per_epoch,
    callbacks=[
        # callback self-registration
        ciclo.keras_bar(total=total_steps), # runs on ciclo.on_train_batch_end
        ciclo.checkpoint(f"logdir/model"),  # runs on ciclo.on_epoch_end
    ],
    stop=total_steps,
)
```
```
80/100 [=====================>.....] - ETA: 42s - accuracy: 0.6148 - loss: 1.537120
```

### Managed API 🧪
The `managed` API aims to simplify the process of creating JAX programs for common patterns, such as `jit`, data parallelism with `pmap`, etc. To use this API, you need to define a compatible state type, which can be easily achieved by inheriting from `managed.ManagedState`.

```python
from ciclo import managed

state = managed.ManagedState.create(
    params=params, # can be any pytree
    tx=optax.adamw(1e-3), # optax optimizer
    strategy="jit", # "data-parallel" or "eager"
)
```

With the managed API, you can use the `train_step` decorator to define a training step easily. The `managed.train_step` function expects you to return logs with at least one loss, which will be used to automatically compute the gradients and update the parameters.

```python
@managed.train_step
def train_step(state, batch):
    loss = ... # compute loss
    logs = ciclo.logs()
    logs.add_loss("loss", loss) # <<< register loss
    return logs, state

for batch in train_dataset:
    logs, state = train_step(state, batch)
    print(f"loss: {logs.losses.loss}")
```

If you need to perform some computation under a strategy without automatically computing gradients and updating the parameters, you can use the `managed.step` decorator.

```python
@managed.step
def test_step(state, batch):
    loss = ... # compute loss
    logs = ciclo.logs()
    logs.add_metric("loss", loss)
    return logs, state
```

### Framework Support 🧪

To make JAX accessible for begginers, `ciclo` plans to provide a set of predefined state types for common frameworks, such as `flax`, `haiku`, and `equinox` (for now only `flax` is supported). These types are based on `ManagedState` and make it easy to perform tasks like training, evaluation, and inference without having to have a deep understanding of either the framework or JAX. Similar to Keras you just provide a model, an optimizer, some losses and metrics:

```python
import flax.linen as nn

model = nn.Sequential([
    lambda x: x.reshape((x.shape[0], -1)) / 255.0,
    nn.Dense(128),
    nn.relu,
    nn.Dense(10),
])
state = ciclo.create_flax_state(
    model,
    inputs=jnp.empty((1, 28, 28, 1)),
    tx=optax.adamw(1e-3),
    losses={"loss": cross_entropy_loss},
    metrics={"accuracy": Accuracy, "avg_loss": AverageLoss},
    strategy="jit", # "data-parallel" or "eager"
)
```

These custom `state` objects provide `train_step`, `test_step`, and `predict_step` methods, this means that when used with the `train_loop`, `test_loop`, and `predict_loop` APIs, they provide a highly simplified experience:

```python
state, history, elapsed = ciclo.train_loop(
    state, # methods are automatically registered
    train_dataset,
    callbacks=[
        ciclo.keras_bar(total=total_steps),
    ],
    test_dataset=lambda: get_test_dataset(),
    epoch_duration=steps_per_epoch,
    stop=total_steps,
)
```

You can also use the `train_step`, `test_step`, and `predict_step` methods directly to create custom training procedures:

```python
for batch in train_dataset:
    logs, state = state.train_step(batch)
    print(f"loss: {logs.losses.loss}")
```


### Manual Iteration
Ciclo provides a set of loosely coupled APIs that can be used independently from the `loop` API to create custom training procedures when more control is needed. In the example below, we demonstrate how to manually iterate through a training dataset using `ciclo`.

First, we define a few callbacks and utilities such as `call_checkpoint`, `checkpoint`, `keras_bar`, and `history`. Then, we use the `ciclo.elapse` function to iterate through the training dataset for a specified number of steps. During each iteration, we update the `logs` and `state` using the `train_step` function. We periodically checkpoint the `state`, update a progress bar, and commit the logs to the `history`.

```python
call_checkpoint = ciclo.every(steps=1000) # Schedule
checkpoint = ciclo.checkpoint(f"logdir/model") # Callback
keras_bar = ciclo.keras_bar(total=total_steps) # Callback
history = ciclo.history() # History
# (Elapsed, Batch)
for elapsed, batch in ciclo.elapse(train_dataset, stop=total_steps):
    logs = ciclo.logs() # Logs
    # update logs and state
    logs.updates, state = train_step(state, batch)
    # periodically checkpoint state
    if call_checkpoint(elapsed):
        checkpoint(elapsed, state) # serialize state

    keras_bar(elapsed, logs) # update progress bar
    history.commit(elapsed, logs) # commit logs to history
```
This approach allows for fine-grained control over the training process and enables customization of various aspects of the training loop.

## Examples

For a more in-depth look at how to use `ciclo`, check out our [examples](./examples) folder which contains a set of python scripts that demonstrate how to use `ciclo` to train models using different APIs.

* [00 Linear Regression](examples/00_linear_regression_pure_jax.py) (pure jax)
* [01 Simple MNIST](examples/01_mnist_simple.py)
* [02 Using train_loop](examples/02_mnist_train_loop.py)
* [03 Manual Iteration](examples/03_mnist_manual_iteration.py)
* [04 Managed API](examples/04_mnist_managed_api.py)
* [05 Using create_flax_state](examples/05_mnist_flax_state.py)