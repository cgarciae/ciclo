import numpy as np
import pytest

import ciclo


class TestLoops:
    def test_basic_loop(self):
        def increment(state, key):
            state[key] += 1
            return None, state

        state = {"a": 0, "b": 0}

        state, history, elapsed = ciclo.loop(
            state,
            ciclo.elapse(range(10)),
            {
                ciclo.every(1): lambda state: increment(state, "a"),
                ciclo.every(2): lambda state: increment(state, "b"),
            },
        )

        assert state["a"] == 10
        assert state["b"] == 5

    @pytest.mark.skip(reason="Integer and boolean schedules removed for now")
    def test_integer_schedules(self):
        def increment(state, key):
            state[key] += 1
            return None, state

        state = {"a": 0, "b": 0}

        state, history, elapsed = ciclo.loop(
            state,
            ciclo.elapse(range(10)),
            {
                1: lambda state: increment(state, "a"),
                2: lambda state: increment(state, "b"),
            },
        )

        assert state["a"] == 10
        assert state["b"] == 5

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
