import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import ciclo

input_shape = 3


def get_dataset(batch_size: int):
    while True:
        yield {
            "image": np.empty((batch_size, input_shape)),
            "label": np.random.randint(0, 10, size=(batch_size,)),
        }


class TestEquinox:
    def test_01_minist_equinox(self):
        batch_size = 3
        total_steps = 3

        # Define model
        class LinearClassifier(eqx.Module):
            linear: eqx.nn.Linear

            def __init__(self, *, key: jax.random.KeyArray):
                self.linear = eqx.nn.Linear(input_shape, 10, key=key)

            def __call__(self, x):
                x = x.reshape((x.shape[0], -1)) / 255.0  # flatten
                x = jax.vmap(self.linear)(x)
                return x

        # Define State
        class State(eqx.Module):
            params: LinearClassifier
            opt_state: optax.OptState
            tx: optax.GradientTransformation = eqx.static_field()

            @classmethod
            def create(cls, params: LinearClassifier, tx: optax.GradientTransformation):
                return cls(params=params, opt_state=tx.init(params), tx=tx)

            def apply_gradients(self, grads):
                updates, opt_state = self.tx.update(
                    grads, self.opt_state, params=self.params
                )
                params = optax.apply_updates(self.params, updates)
                return dataclasses.replace(self, params=params, opt_state=opt_state)

        @jax.jit
        def train_step(state: State, batch):
            inputs, labels = batch["image"], batch["label"]

            # update the model's state
            def loss_fn(params: LinearClassifier):
                logits = params(inputs)
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=labels
                ).mean()
                return loss, logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)

            # add logs
            logs = ciclo.logs()
            logs.add_metric("loss", loss)
            logs.add_metric("accuracy", jnp.mean(jnp.argmax(logits, -1) == labels))

            return logs, state

        # Initialize state
        model = LinearClassifier(key=jax.random.PRNGKey(0))
        state = State.create(
            params=model,
            tx=optax.adamw(1e-3),
        )

        # training loop
        state, history, _ = ciclo.loop(
            state,
            get_dataset(batch_size),
            {
                ciclo.every(1): train_step,
                **ciclo.keras_bar(total=total_steps),
            },
            stop=total_steps,
        )

        # plot the training history
        steps, loss, accuracy = history.collect("steps", "loss", "accuracy")

        assert len(history) == total_steps
        assert len(loss) == len(accuracy) == total_steps
        assert steps == list(range(total_steps))
