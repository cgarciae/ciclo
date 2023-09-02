import jax.numpy as jnp

import ciclo


def dummy_inner_loop_fn(_):
    log_history = [
        {
            "stateful_metrics": {
                "A": jnp.array(1.0, dtype=jnp.float32),
                "B": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C": jnp.array(0.0, dtype=jnp.float32),
                "D": jnp.array(0.0, dtype=jnp.float32),
            },
            "elapsed": {
                "steps": 1,
                "samples": 1,
            },
        },
        {
            "stateful_metrics": {
                "A": jnp.array(0.0, dtype=jnp.float32),
            },
            "metrics": {
                "C": jnp.array(1.0, dtype=jnp.float32),
            },
            "elapsed": {
                "steps": 2,
                "samples": 2,
            },
        },
    ]
    return None, log_history, None


class TestInnerLoop:
    def test_inner_loop_default_aggregation(self):
        inner_loop = ciclo.callbacks.inner_loop(
            "test",
            dummy_inner_loop_fn,
        )

        log_history, _ = inner_loop(None)

        assert log_history == {
            "stateful_metrics": {
                "A_test": jnp.array(0.0, dtype=jnp.float32),
                "B_test": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C_test": jnp.array(1.0, dtype=jnp.float32),
                "D_test": jnp.array(0.0, dtype=jnp.float32),
            },
        }

    def test_inner_loop_callable_aggregation(self):
        inner_loop = ciclo.callbacks.inner_loop(
            "test",
            dummy_inner_loop_fn,
            aggregation=sum,
        )

        log_history, _ = inner_loop(None)

        assert log_history == {
            "stateful_metrics": {
                "A_test": jnp.array(1.0, dtype=jnp.float32),
                "B_test": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C_test": jnp.array(1.0, dtype=jnp.float32),
                "D_test": jnp.array(0.0, dtype=jnp.float32),
            },
        }

    def test_inner_loop_mean_aggregation(self):
        inner_loop = ciclo.callbacks.inner_loop(
            "test",
            dummy_inner_loop_fn,
            aggregation="mean",
        )

        log_history, _ = inner_loop(None)

        assert log_history == {
            "stateful_metrics": {
                "A_test": jnp.array(0.5, dtype=jnp.float32),
                "B_test": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C_test": jnp.array(0.5, dtype=jnp.float32),
                "D_test": jnp.array(0.0, dtype=jnp.float32),
            },
        }

    def test_inner_loop_aggregation_dict(self):
        inner_loop = ciclo.callbacks.inner_loop(
            "test",
            dummy_inner_loop_fn,
            aggregation={"stateful_metrics": "sum", "metrics": "min"},
        )

        log_history, _ = inner_loop(None)

        assert log_history == {
            "stateful_metrics": {
                "A_test": jnp.array(1.0, dtype=jnp.float32),
                "B_test": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C_test": jnp.array(0.0, dtype=jnp.float32),
                "D_test": jnp.array(0.0, dtype=jnp.float32),
            },
        }

        inner_loop = ciclo.callbacks.inner_loop(
            "test",
            dummy_inner_loop_fn,
            aggregation={"stateful_metrics": "first"},
        )

        log_history, _ = inner_loop(None)

        assert log_history == {
            "stateful_metrics": {
                "A_test": jnp.array(1.0, dtype=jnp.float32),
                "B_test": jnp.array(1.0, dtype=jnp.float32),
            },
            "metrics": {
                "C_test": jnp.array(1.0, dtype=jnp.float32),
                "D_test": jnp.array(0.0, dtype=jnp.float32),
            },
        }


class TestEarlyStopping:
    def test_patience(self):
        dataset = jnp.minimum(jnp.arange(10), 5)

        def train_step(state, batch):
            logs = ciclo.logs()
            logs.add_metric("x", batch)
            return logs, state

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping("x", optimization_mode="max", patience=1),
                ],
            },
        )

        assert len(history) == 7

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping("x", optimization_mode="max", patience=3),
                ],
            },
        )

        assert len(history) == 9

    def test_initial_patience(self):
        dataset = jnp.maximum(jnp.minimum(jnp.arange(10), 5), 2)

        def train_step(state, batch):
            logs = ciclo.logs()
            logs.add_metric("x", batch)
            return logs, state

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x", optimization_mode="max", patience=1, initial_patience=1
                    ),
                ],
            },
        )

        assert len(history) == 2

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x", optimization_mode="max", patience=1, initial_patience=3
                    ),
                ],
            },
        )

        assert len(history) == 7

    def test_min_optimization_mode(self):
        dataset = jnp.maximum(jnp.minimum(jnp.arange(9, 0, -1), 6), 3)

        def train_step(state, batch):
            logs = ciclo.logs()
            logs.add_metric("x", batch)
            return logs, state

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x", optimization_mode="min", patience=1, initial_patience=4
                    ),
                ],
            },
        )

        assert len(history) == 8

    def test_min_delta(self):
        dataset = jnp.arange(0, 1, 0.1)

        def train_step(state, batch):
            logs = ciclo.logs()
            logs.add_metric("x", batch)
            return logs, state

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x",
                        optimization_mode="max",
                        patience=1,
                        min_delta=0.01,
                    ),
                ],
            },
        )

        assert len(history) == 10

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x", optimization_mode="max", patience=1, min_delta=0.1
                    ),
                ],
            },
        )

        assert len(history) == 2

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x",
                        optimization_mode="max",
                        patience=3,
                        min_delta=0.05,
                    ),
                ],
            },
        )

        assert len(history) == 10

    def test_min_relative_delta(self):
        dataset = jnp.arange(0, 1, 0.1)

        def train_step(state, batch):
            logs = ciclo.logs()
            logs.add_metric("x", batch)
            return logs, state

        _, history, _ = ciclo.loop(
            None,
            dataset,
            {
                ciclo.every(1): [
                    train_step,
                    ciclo.early_stopping(
                        "x",
                        optimization_mode="max",
                        patience=1,
                        min_delta=0.5,
                        delta_mode="relative",
                    ),
                ],
            },
        )

        assert len(history) == 4
