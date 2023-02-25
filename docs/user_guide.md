
WIP

```python
import ciclo
```
```python
def step(state, batch):
    return state + batch

state, history, elapsed = ciclo.loop(
    state=0,
    dataset=range(10),
    tasks={
        ciclo.every(1): [step],
    },
)

assert state == 45
```
```python hl_lines="5"
def step(state, batch):
    return state + batch

state, history, elapsed = ciclo.loop(
    state=0, # any pytree
    dataset=range(10),
    tasks={
        ciclo.every(1): [step],
    },
)
```
```python hl_lines="6"
def step(state, batch):
    return state + batch

state, history, elapsed = ciclo.loop(
    state=0,
    dataset=range(10), # any iterable of pytrees
    tasks={
        ciclo.every(1): [step],
    },
)
```
```python hl_lines="8"
def step(state, batch):
    return state + batch

state, history, elapsed = ciclo.loop(
    state=0,
    dataset=range(10), 
    tasks={
        ciclo.every(1): [step], # Schedule: List[Callback]
    },
)
```
### Schedules
```python
def schedule(elapsed: Elapsed) -> bool
```
```python
ciclo.every(steps=10) # by steps
ciclo.every(samples=10) # by samples
ciclo.every(time=10) # by time (seconds)
ciclo.every(steps=20, steps_offset=5) # by steps with offset
```

### Callbacks
```python
def callback([state, batch, elasped, loop_state]) 
    -> None | state | logs | (logs | None, state | None)
```
```python
def step(state, batch) -> state
```
```python
ciclo.keras_bar
ciclo.checkpoint
ciclo.wandb
ciclo.early_stopping
```
### Logs
```python
def step(state, batch):
    state = state + batch

    logs = ciclo.logs()
    logs.add_entry("states", "state", state) # (collection, name, value)
    logs.add_entry("inputs", "batch", batch)

    return logs, state
```
```python hl_lines="9"
state, history, elapsed = ciclo.loop(
    state=0,
    dataset=range(10), 
    tasks={
        ciclo.every(1): [step], # Schedule: List[Callback]
    },
)

states, batches = history.collect("states.state", "inputs.batch")

assert batches = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```
```python
states, batches = history.collect("state", "batch") # shorthand
```
```python
Logs.add_metric # adds an entry to the "metrics" collection
Logs.add_stateful_metric # adds an entry to the "stateful_metrics" collection
Logs.add_loss # adds an entry to the "losses" collection
Logs.add_output # adds an entry to the "outputs" collection
```

### Example: Linear Regression

```python
import numpy as np
import jax
import jax.numpy as jnp
import ciclo
```
```python
X = np.linspace(0, 1, 100)
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)
```
```python
def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]
```
```python
state = {"w": 0.0, "b": 0.0}
```
```python
@jax.jit
def train_step(state, batch):
    x, y = batch

    def loss_fn(params):
        w, b = params
        y_pred = w * X + b
        return jnp.mean((y - y_pred) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn((state["w"], state["b"]))

    # sdg
    state = jax.tree_map(lambda x, g: x - 0.1 * g, state, grad)

    logs = ciclo.logs()
    logs.add_metric("loss", loss)

    return logs, state
```
```python
@jax.jit
def test_step(state):
    y_pred = state["w"] * X + state["b"]
    loss = jnp.mean((Y - y_pred) ** 2)

    logs = ciclo.logs()
    logs.add_metric("mse", loss)

    return logs
```
```python
total_steps = 10_000

state, history, elapsed = ciclo.loop(
    state=state,
    dataset=dataset(batch_size=32),
    tasks={
        ciclo.every(100): [test_step],
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=total_steps)
        ],

    },
    stop=total_steps,
)
```
