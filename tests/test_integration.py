# %%
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_metrics as jm
import numpy as np
import optax
from clu.metrics import Accuracy, Average, Collection
from flax import struct
from flax.training import train_state

import ciclo

input_shape = 3


def get_dataset(batch_size: int):
    while True:
        yield {
            "image": np.empty((batch_size, input_shape)),
            "label": np.random.randint(0, 10, size=(batch_size,)),
        }


def get_tuple_dataset(batch_size: int):
    while True:
        yield (
            np.empty((batch_size, input_shape)),
            np.random.randint(0, 10, size=(batch_size,)),
        )


class TestIntegration:
    def test_simple_loop(self):
        TrainState = train_state.TrainState
        batch_size = 3
        total_steps = 3
        ds_train = get_dataset(batch_size)

        @jax.jit
        def train_step(state: TrainState, batch):
            inputs, labels = batch["image"], batch["label"]

            def loss_fn(params):
                logits = state.apply_fn({"params": params}, inputs)
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=labels
                ).mean()
                return loss, logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)
            logs = ciclo.logs()
            logs.add_loss("loss", loss)
            logs.add_metric("accuracy", jnp.mean(jnp.argmax(logits, -1) == labels))
            return logs, state

        # Initialize state
        def create_state():
            model = nn.Dense(features=10)
            variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, input_shape)))
            return TrainState.create(
                apply_fn=model.apply,
                params=variables["params"],
                tx=optax.adamw(1e-3),
            )

        state = create_state()
        with TemporaryDirectory() as logdir:
            state, history, _ = ciclo.loop(
                state,
                ds_train,
                {
                    ciclo.every(1): train_step,
                    ciclo.every(2): ciclo.checkpoint(f"{logdir}/model"),
                    **ciclo.keras_bar(total=total_steps),
                },
                stop=total_steps,
            )

            steps, loss, accuracy = history.collect("steps", "loss", "accuracy")

            assert Path(f"{logdir}/model").exists()
            assert len(history) == total_steps
            assert len(loss) == len(accuracy) == total_steps
            assert steps == [0, 1, 2]

    def test_loop_with_validation(self):
        batch_size = 3
        total_steps = 3 * 4
        ds_train = get_dataset(batch_size)
        ds_valid = get_dataset(batch_size)

        @struct.dataclass
        class Metrics(Collection):
            loss: Average.from_output("loss")
            accuracy: Accuracy

            def update(self, **kwargs) -> "Metrics":
                updates = self.single_from_model_output(**kwargs)
                return self.merge(updates)

        class TrainState(train_state.TrainState):
            metrics: Metrics

        @jax.jit
        def train_step(state: TrainState, batch):
            def loss_fn(params):
                logits = state.apply_fn({"params": params}, batch["image"])
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=logits, labels=batch["label"]
                ).mean()
                return loss, logits

            (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)
            metrics = state.metrics.update(
                loss=loss, logits=logits, labels=batch["label"]
            )
            logs = ciclo.logs()
            logs.add_stateful_metrics(**metrics.compute())
            return logs, state.replace(metrics=metrics)

        @jax.jit
        def eval_step(state: TrainState, batch):
            logits = state.apply_fn({"params": state.params}, batch["image"])
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=batch["label"]
            ).mean()
            metrics = state.metrics.update(
                loss=loss, logits=logits, labels=batch["label"]
            )
            logs = ciclo.logs()
            logs.add_stateful_metrics(**metrics.compute())
            return logs, state.replace(metrics=metrics)

        def reset_metrics(state: TrainState):
            return state.replace(metrics=state.metrics.empty())

        # Initialize state
        def create_state():
            model = nn.Dense(features=10)
            variables = model.init(jax.random.PRNGKey(0), jnp.empty((1, input_shape)))
            return TrainState.create(
                apply_fn=model.apply,
                params=variables["params"],
                tx=optax.adamw(1e-3),
                metrics=Metrics.empty(),
            )

        state = create_state()
        with TemporaryDirectory() as logdir:
            state, history, _ = ciclo.loop(
                state,
                ds_train,
                {
                    ciclo.every(1): train_step,
                    ciclo.every(2): [
                        ciclo.inner_loop(
                            "valid",
                            lambda state: ciclo.loop(
                                state,
                                ds_valid,
                                {ciclo.every(1): eval_step},
                                on_start=[reset_metrics],
                                stop=1,
                            ),
                        ),
                        ciclo.checkpoint(
                            f"{logdir}/model",
                            monitor="accuracy_valid",
                            optimization_mode="max",
                        ),
                        ciclo.early_stopping(
                            monitor="accuracy_valid",
                            optimization_mode="max",
                            patience=100,
                        ),
                    ],
                    **ciclo.keras_bar(total=total_steps),
                },
                stop=total_steps,
            )

            steps, loss, accuracy = history.collect("steps", "loss", "accuracy")

            assert Path(f"{logdir}/model").exists()
            assert len(history) == total_steps
            assert len(loss) == len(accuracy) == total_steps
            assert steps == list(range(total_steps))

    def test_create_flax_state(self):
        batch_size = 3
        total_steps = 3 * 4
        steps_per_epoch = 3
        test_steps = 2

        # Define model
        class Linear(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=10)(x)
                self.put_variable("losses", "a", jnp.array(1.0))
                self.put_variable("metrics", "b", jnp.array(2.0))
                return x

        # Initialize state
        model = Linear()
        state = ciclo.create_flax_state(
            model,
            inputs=jnp.empty((1, input_shape)),
            tx=optax.adamw(1e-3),
            losses={"loss": jm.losses.Crossentropy()},
            metrics={"accuracy": jm.metrics.Accuracy()},
            strategy="jit",
        )
        state, history, _ = ciclo.train_loop(
            state,
            get_tuple_dataset(batch_size),
            callbacks=[
                ciclo.keras_bar(total=total_steps),
                ciclo.checkpoint(
                    f"logdir/{Path(__file__).stem}/{int(time())}",
                    monitor="accuracy_test",
                    optimization_mode="max",
                ),
            ],
            test_dataset=lambda: get_tuple_dataset(batch_size),
            epoch_duration=steps_per_epoch,
            test_duration=test_steps,
            stop=total_steps,
        )

        (
            steps,
            avg_loss,
            accuracy,
            avg_loss_test,
            accuracy_test,
            avg_a,
            b,
            avg_a_test,
            b_test,
        ) = history.collect(
            "steps",
            "avg_loss",
            "accuracy",
            "avg_loss_test",
            "accuracy_test",
            "avg_a",
            "b",
            "avg_a_test",
            "b_test",
        )

        total_logs = total_steps // steps_per_epoch

        assert len(steps) == total_logs
        assert len(avg_loss) == total_logs
        assert len(accuracy) == total_logs
        assert len(avg_loss_test) == total_logs
        assert len(accuracy_test) == total_logs
        assert len(avg_a) == total_logs
        assert len(b) == total_logs
        assert len(avg_a_test) == total_logs
        assert len(b_test) == total_logs
