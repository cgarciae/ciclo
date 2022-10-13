
## Motivation

We want to define composable APIs that enable the creation of complex training loops and higher-level JAX frameworks.

```python
total_steps = 10_000

_, history, state = loop(
    state,
    dataset,
    {
        every(steps=1): [
            train_step, 
            keras_bar(total=total_steps),
        ],
    },
    stop=total_steps,
)
```

## Callback Spec
```python
def callback(
    state, batch, [broadcasts, statics]
) -> (outputs, logs, state) | (logs, state) | None:
```
<details><summary>Description</summary>

Where:
* `state` is a pytree containing the training state
* `batch` is a pytree containing the batch data
* `broadcasts` (optional) is a pytree containing additional inputs that can be broadcasted to all devices
* `statics` (optional) is a hashable object containing additional static inputs

`broadcasts` and `statics` are passed as positional arguments.

Callbacks can return:
* `outputs` (optional) is a dict containing the outputs of the computation
* `logs` (optional) is a dict containing the log information
* `state` (optional) is a pytree containing the updated training state


Returns can be either a 3-tuple of `(outputs, logs, state)`, a 2-tuple of `(logs, state)`, or `None`. Additionally, any of the outputs (`state`, `logs`, or `state`) can be `None` if they are not needed.

</details><br>

## Loops

### Loop callbacks
```python

def callback(
    state, batch, [elapsed, loop]
) -> (outputs, logs, state) | (logs, state) | None:
```
<details><summary>Description</summary>

Loop callbacks implement the callback spec with the following instantiation: 

* `elapsed` (optional) is a pytree containing information about steps, samples, time, and date since the start of the loop.
* `loop` (optional) is a regular python object that runs the loop.

</details><br>

### Schedules
```python
def schedule(elapsed: Elapsed) -> bool:
```

<details><summary>Description</summary>

`schedule`s take in an `elapsed` instance and returns a boolean indicating whether that time is within the schedule.

</details><br>

### Loops
```python
def loop(
    state, dataset: [batch], tasks: {schedule: [callback]}
) -> (loop_state, state):
```

<details><summary>Description</summary>

Loops are comprised of a `state` that is threaded through the loop, a `dataset` that is iterated over, and a dictionary of `schedules` to `callbacks` that execute various tasks such as training, logging, evaluation, etc.

Loops return the final `state` and a `loop_state` that contains information such the `log` history, `output` history, and the `elapsed` time.

</details><br>

## Examples

<details><summary>Sample Usage with Flax</summary>

```python
@jax.jit
def train_step(state: TrainState, batch):
    inputs, labels = batch["image"], batch["label"]

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels
        ).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = (state.params)
    state = state.apply_gradients(grads=grads)
    logs = {
        "loss": loss, 
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)
    }
    return logs, state


_, history, state = loop(
    state, # TrainState
    train_ds.as_numpy_iterator(),
    {
        every(1): [
            train_step, 
            keras_bar(total=total_steps), # progress bar
        ],
        every(100): [
            Profile(logdir=logdir), # clu.periodic_actions
        ],
    },
    stop=total_steps,
)
```
```
4650/10000 [============>.................] - ETA: 8s - accuracy: 0.9779 - loss: 0.0718  
```

</details><br>