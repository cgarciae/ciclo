# Ciclo
_Training loop utilities and abstractions for JAX_

`ciclo` is a library for training loops in JAX. It provides a set of utilities and abstractions to build complex training loops and higher-level frameworks.

**Features**

✔️ Training utilities <br>
✔️ Loop language <br>
🧪 Managed API (Distributed Strategies) <br>
💡 Predefined Loops (e.g. `fit`) <br>
💡 Framework support <br>

<details><summary><b>Why Ciclo?</b></summary>


- In JAX functions are first-class citizens, instead of monolithic classes like `Model` or `Trainer` in other frameworks, there is a lot of benefit in a functional API for the training interface as well.<br>
- The JAX community is very focused on research, and as such there is a lot of interest in flexibility and control over the training loop. For this reason, `ciclo` provides some basic utilities and lets the user choose their desired level of abstraction.<br><br>

<b>Comparison with other libraries</b><br><br>

- What about Elegy? Ciclo can be seen as the next version of Elegy that is built with better foundations. While Elegy started with a very rigid high-level API and gradually added more flexibility, Ciclo starts with low-level utilities and gradually adds more abstraction.<br>
- What about `clu`? Ciclo took from inspiration from `clu` and rather than compete with it, Ciclo aims to complement it. At the lowest level they both compose by virtue of just providing utilities that work with JAX, however, whenever possible Ciclo's abstractions provide support for `clu`'s utilities e.g. `loop` supports `clu`'s `PeriodicAction`s.<br>

</details>

### Installation

```bash
pip install ciclo
```

---
  
## Training utilities

* Time tracking: `at`, `elapse`, ...
* Schedules: `every`, ...
* Callbacks: `checkpoint`, `early_stopping`, `keras_bar`, ...
* Logging: `logs`, `history`


### Time Tracking

  
```python
end_period = ciclo.at(steps=10_000) # {steps, samples, time}

for elapsed, batch in ciclo.elapse(dataset):
    # Elapsed(steps=..., samples=..., time=...)
    if elapsed >= end_period:
        break
```

### Schedules

  
```python
eval_period = ciclo.every(steps=1000) # {steps, samples, time}

for elapsed, batch in ciclo.elapse(dataset):

    if eval_period(elapsed):
        # eval code
```


### Callbacks

  
```python
call_checkpoint = ciclo.every(steps=1000)
checkpoint = ciclo.checkpoint(f"logdir/my-model")
keras_bar = ciclo.keras_bar(total=total_steps)

for elapsed, batch in ciclo.elapse(dataset):
    logs, state = train_step(state, batch) # jax function

    if call_checkpoint(elapsed):
        checkpoint(elapsed, state) # state checkpointing

    keras_bar(elapsed, logs) # keras progress bar
```


### Logging

```python
@jax.jit
def train_step(state, batch):
    ...
    logs = ciclo.logs()
    logs.add_loss("loss": loss)
    logs.add_metric("accuracy": jnp.mean(jnp.argmax(logits, -1) == labels))
    return logs, state
```
  
```python
history = ciclo.history()

for elapsed, batch in ciclo.elapse(dataset):
    logs = ciclo.logs()
    # merge log updates
    logs.updates, state = train_step(state, batch)
    # commit logs
    history.commit(elapsed, logs)
# collect log values (e.g. for plotting)
steps, loss, accuracy = history.collect("steps", "loss", "accuracy")
```


<details><summary>Complete Example</summary>

```python
@jax.jit
def train_step(state: TrainState, batch):
    ...
    logs = ciclo.logs()
    logs.add_metric("loss": loss)
    logs.add_metric("accuracy": jnp.mean(jnp.argmax(logits, -1) == labels))
    return logs, state

total_steps = 10_000
call_checkpoint = ciclo.every(steps=1000)
checkpoint = ciclo.checkpoint(f"logdir/my-model")
keras_bar = ciclo.keras_bar(total=total_steps)
end_period = ciclo.at(steps=total_steps)
history = ciclo.history()

for elapsed, batch in ciclo.elapse(dataset):
    logs = ciclo.logs()
    logs.updates, state = train_step(state, batch)
    
    if call_checkpoint(elapsed):
        checkpoint(elapsed, state)
    
    keras_bar(elapsed, logs)
    history.commit(elapsed, logs)
    if elapsed >= end_period:
        break

steps, loss, accuracy = history.collect("steps", "loss", "accuracy")
```

</details><br>

---

## Loop language
  
```python
def loop(
    state: State,
    dataset: Iterable[Batch],
    tasks: {Schedule: [Callback]},
) -> (State, History, Elapsed)
```
  
```python

total_steps = 10_000

state, history, elapsed = ciclo.loop(
    state,
    dataset,
    {
        ciclo.every(1): [train_step],
        ciclo.every(steps=1000): [ciclo.checkpoint(f"logdir/my-model")],
        ciclo.every(1): [ciclo.keras_bar(total=total_steps)],
    },
    stop=ciclo.at(steps=total_steps)
)
```

<details><summary>Syntactic Sugar</summary>

If you have a single callback for a given schedule you can pass it directly. Furthermore, for `CallbackBase` instances (all callbacks in Ciclo implement this) that need to be run at every iteration, you can avoid having to specify the schedule by using the Mapping expansion `'**'` operator.

| Syntax | Expansion |
| --- | --- |
| `schedule: callback` | `schedule: [callback]` |
| `**callback` | `every(1): [callback]` |

```python

  total_steps = 10_000
  
  state, history, elapsed = ciclo.loop(
    state,
    dataset,
    {
      ciclo.every(1): train_step,
      ciclo.every(steps=1000): ciclo.checkpoint(f"logdir/my-model"),
      **ciclo.keras_bar(total=total_steps),
    },
    stop=ciclo.at(steps=total_steps)
  )
```

</details><br>

### Loop function callbacks
  
```python
def f(
    [state, batch, elapsed, loop_state] # accept between 0 and 4 args
) -> (logs | None, state | None) | logs | state | None
```
Where:

* state: `Any`, cannot be `tuple` or `dict` for single return value
* batch: `Any`, current batch
* elapsed: `Elapsed`, current elapsed steps/samples/time, jit-able
* loop_state: `LoopState`, contains information about the current loop state, not jit-able
* logs: `LogsLike = Dict[str, Dict[str, Any]]`