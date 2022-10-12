# ciclo

## API
```python
from ciclo import Elapsed
State = Any
Batch = Any
Logs = Dict[str, Any]

def callback(
    state: State, batch: Batch, elapsed: Elapsed
) -> Tuple[Logs, State]:
    ...

def schedule(elapsed: Elapsed) -> bool:
    ...

state, loop = ciclo.loop(
    state,   # Pytree
    dataset, # Iterable[Batch]
    {schedule: [callback]},
)
```

## Sample Usage
```python
@jax.jit
def train_step(state: TrainState, batch, _):
    inputs, labels = batch["image"], batch["label"]

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    logs = {"loss": loss, "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)}
    return logs, state


state, loop = ciclo.loop(
    state,
    train_ds.as_numpy_iterator(),
    {
        ciclo.every(1): [
            train_step, 
            ciclo.keras_bar(total=total_steps), # progress bar
        ],
        ciclo.every(100): [
            Profile(logdir=logdir), # clu.periodic_actions
        ],
    },
    stop=total_steps,
)
```
```
4650/10000 [============>.................] - ETA: 8s - accuracy: 0.9779 - loss: 0.0718  
```
