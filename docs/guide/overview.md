# Ciclo
_Training loop utilities and abstractions for the JAX ecosystem_

**Main Features**

✔️ Training utilities <br>
✔️ Loop language <br>
🧪 Managed API (Distributed Strategies) <br>
💡 Predefined Loops (e.g. `fit`) <br>
💡 Framework support <br>

---
  
## Training utilities

* Time tracking: `at`, `elapse`, ...
* Schedules: `every`, ...
* Callbacks: `checkpoint`, `early_stopping`, `keras_bar`, ...
* Logging: `logs`, `history`


### Time Tracking

  
```python
end_period = ciclo.at(steps=10_000) # {steps, samples, time}

for elapsed, batch in ciclo.elapse(dataset): # track steps/samples/time
# elapsed = Elapsed(steps=..., samples=..., time=...)
if elapsed >= end_period:
    break
```

### Schedules

  
```python
eval_period = ciclo.every(steps=1000) # {steps, samples, time}

for elapsed, batch in ciclo.elapse(dataset):

    if eval_period(elapsed):
        # evaluate model here
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
def train_step(state: TrainState, batch):
    ...
    logs = ciclo.logs()
    logs.add_metric("loss": loss)
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
# collect log values e.g. for plotting
steps, loss, accuracy = history.collect("steps", "loss", "accuracy)
```


### Complete Example

<details><summary>Example</summary>

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

</details>

---

## Loop language
  
### Loop
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
**Sugar**
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

### Loop function callbacks
  
```python
def f(
    [state, batch, elapsed, loop_state] # accept between 0 and 4 args
) -> (logs | None, state | None) | logs | state | None
```
Where:

* logs: `LogsLike = Dict[str, Dict[str, Any]]`
* state: `Any`, cannot be `tuple` or `dict` for single return value


### Loop callbacks
  
```python

class LoopCallback(Protocol):
    __loop_callback__: LoopState -> Tuple[LogsLike, State]
```


### LoopState
  
```python
  
class LoopState:
    state: State
    batch: Batch
    history: History
    elapsed: Elapsed
    logs: Logs
    accumulated_logs: Logs
    metadata: Any
    stop_iteration: bool
  
```




