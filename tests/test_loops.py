import ciclo
import numpy as np


class TestLoops:
    def test_bug(self):

        state = None
        N = 0

        def step():
            nonlocal N
            N += 1
            logs = {"stateful_metrics": {"a": N}, "metrics": {"b": -N}}
            return logs, None

        def data():
            x = np.empty((2, 3))
            while True:
                yield x

        state, history, _ = ciclo.loop(
            state,
            data(),
            {
                ciclo.every(1): step,
            },
            stop=ciclo.at(3),
        )

        a_list, b_list = history.collect("a", "b")

        assert a_list == list(range(1, 4))
        assert b_list == list(range(-1, -4, -1))
