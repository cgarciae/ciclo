# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import ciclo

X = np.linspace(0, 1, 100)
Y = 0.8 * X + 0.1 + np.random.normal(0, 0.1, size=X.shape)


def dataset(batch_size):
    while True:
        idx = np.random.choice(len(X), size=batch_size)
        yield X[idx], Y[idx]


state = {"w": 0.0, "b": 0.0}


@jax.jit
def forward(state, x):
    w, b = state["w"], state["b"]
    return w * x + b


def mse(y, y_pred):
    return jnp.mean((y - y_pred) ** 2)


@jax.jit
def train_step(state, batch):
    x, y = batch

    def loss_fn(state):
        y_pred = forward(state, x)
        return mse(y, y_pred)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(state)

    # sdg
    state = jax.tree_map(lambda w, g: w - 0.1 * g, state, grad)

    logs = ciclo.logs()
    logs.add_metric("loss", loss)

    return logs, state


@jax.jit
def test_step(state):
    y_pred = forward(state, X)
    loss = mse(Y, y_pred)

    logs = ciclo.logs()
    logs.add_metric("mse", loss)

    return logs


total_steps = 10_000

state, history, elapsed = ciclo.loop(
    state=state,
    dataset=dataset(batch_size=32),
    tasks={
        ciclo.every(100): [test_step],
        ciclo.every(1): [
            train_step,
            ciclo.keras_bar(total=total_steps),
        ],
    },
    stop=total_steps,
)


y_pred = forward(state, X)

plt.scatter(X, Y, color="blue")
plt.plot(X, y_pred, color="black")
plt.show()
