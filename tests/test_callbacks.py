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


class TestCallbacks:
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
