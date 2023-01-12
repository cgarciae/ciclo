[![codecov](https://codecov.io/gh/cgarciae/ciclo/branch/main/graph/badge.svg?token=3IKEUAU3C8)](https://codecov.io/gh/cgarciae/ciclo)

# Ciclo
_Training loop utilities and abstractions for JAX_

`ciclo` is a functional library for training loops in JAX. It provides a set of utilities and abstractions to build complex training loops with any JAX framework. `ciclo` defines a set of building blocks that naturally compose together so they can scale up to build higher-level abstractions.

**Features**

✔️ Training utilities <br>
✔️ Loop language <br>
🧪 [experimental] Managed API (simplified training + parallelism) <br>
💡 [idea] Framework support (predifined states + steps) <br>
💡 [idea] Predefined Loops (e.g. `fit`, `evaluate`, `predict`) <br>

<details><summary><b>Why Ciclo?</b></summary>


- In JAX functions are first-class citizens, instead of monolithic classes like `Model` or `Trainer` in other frameworks, there is a lot of benefit in a functional API for the training interface as well.<br>
- The JAX community is very focused on research, and as such there is a lot of interest in flexibility and control over the training loop. For this reason, `ciclo` provides some basic utilities and lets the user choose their desired level of abstraction.<br>
- Choosing the wrong abstractions can often break a framework, when this happens users often abandone the framework altogether. `ciclo` tries to avoid this by providing a set of utilities than can stand on their own so they can be useful even if the user decides to build their own training loop, but allows them to compose together and be used with ever increasing levels of abstraction. Ideally in the future a user should be able to pick anything from a Keras-like simplified experience to defining their own loops, or just coding the training loop manually and still have a good experience.<br><br>


<b>Comparison with other libraries</b><br><br>

- What about Elegy? Ciclo can be seen as the next version of Elegy that is built with better foundations. While Elegy started with a very rigid high-level API and gradually added more flexibility, Ciclo starts with low-level utilities and gradually adds more abstraction.<br>
- What about `clu`? Ciclo took from inspiration from `clu` and rather than compete with it, Ciclo aims to complement it. At the lowest level they both compose by virtue of just providing utilities that work with JAX, however, whenever possible Ciclo's abstractions provide support for `clu`'s utilities e.g. `loop` supports `clu`'s `PeriodicAction`s.<br>

</details>

## Installation

```bash
pip install ciclo
```

## Status
Ciclo is still in early development, the API is subject to change, expect things to break. If you are interested in contributing, please reach out.
  
## Utilities

* Time tracking
* Logging
* Schedules
* Callbacks
* Loops
* Parallelism [experimental]

## Training loop
The `loop` function serves as a mini-language for defining training loops as a composition of functions. The tasks dictionary lets you express the desired behavior of the loop as a composition of schedules and callbacks.

```python
@jax.jit
def train_step(state, batch):
    # update model state and create logs
    ...
    return logs, state

total_steps = 10_000
state = create_state() # initial state

state, history, elapsed = ciclo.loop(
    state, # state: Pytree
    dataset, # dataset: Iterable[Batch]
    {
        # Dict[Schedule, List[Callback]]
        ciclo.every(1): [train_step],
        ciclo.every(steps=1000): [ciclo.checkpoint(f"logdir/model")],
        ciclo.every(1): [ciclo.keras_bar(total=total_steps)],
    },
    stop=total_steps, # stop: Optional[int | Period]
)
```
### Loop function callbacks

Callbacks are either objects that implement the `LoopCallback` protocol or functions of the form:

```python
# variadic: accepts between 0 and 4 arguments
def f(
    [state, batch, elapsed, loop_state]
) -> (logs | None, state | None) | logs | state | None
```
You can safely `jit` all functions of this form except for the ones that accept the `loop_state` argument since it is a mutable object.

### Custom training loops
You can use Ciclo's utilities to build your own training loop when you need more control. 

<details><summary><b>Show</b></summary>

For example, the previous loop can be written as:

```python
total_steps = 5_000
state = create_state() # initial state

call_checkpoint = ciclo.every(steps=1000) # Schedule
checkpoint = ciclo.checkpoint(f"logdir/model") # Callback
keras_bar = ciclo.keras_bar(total=total_steps) # Callback
end_period = ciclo.at(total_steps) # Period
history = ciclo.history() # History
# (Elapsed, Batch)
for elapsed, batch in ciclo.elapse(ds_train.as_numpy_iterator()):
    logs = ciclo.logs() # Logs
    # update logs and state
    logs.updates, state = train_step(state, batch)
    # periodically checkpoint state
    if call_checkpoint(elapsed):
        checkpoint(elapsed, state) # serialize state

    keras_bar(elapsed, logs) # update progress bar
    history.commit(elapsed, logs) # commit logs to history
    # stop training when total_steps is reached
    if elapsed >= end_period:
        break
```

</details><br>

