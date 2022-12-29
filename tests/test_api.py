import ciclo
import pytest


class TestAPI:
    def test_elapsed_gt_period(self):
        period = ciclo.Period.create(steps=10)
        elapsed = ciclo.Elapsed.create()

        assert elapsed < period

        for i in range(10):
            elapsed = elapsed.update(1)
            assert elapsed <= period

        assert elapsed == period

        elapsed = elapsed.update(1)

        assert elapsed > period

    def test_inject(self):
        def f(x, y):
            return x + y

        def g(x, y, z):
            return x + y + z

        args = (1, 2, 3)
        a = ciclo.inject(f)(*args)
        b = ciclo.inject(g)(*args)

        assert a == 3
        assert b == 6

    def test_inject_with_methods(self):
        class Obj:
            def f(self, x, y, z=1, *, w=2):
                return x + y

            def g(self, x, y, z):
                return x + y + z

        obj = Obj()
        args = (1, 2, 3)
        a = ciclo.inject(obj.f)(*args)
        b = ciclo.inject(obj.g)(*args)

        assert a == 3
        assert b == 6

    def test_logs(self):
        logs = ciclo.logs()
        logs.add_loss("some_loss", 1)
        logs.add_metric("some_metric", 2)
        logs.add_metric("other_metric", 3, stateful=True)

        assert logs["losses"] == {"some_loss": 1}
        assert logs["metrics"] == {"some_metric": 2}
        assert logs["stateful_metrics"] == {"other_metric": 3}

    def test_logs_merge(self):
        logs = ciclo.logs()
        updates = {
            "losses": {"some_loss": 1},
            "metrics": {"some_metric": 2},
        }
        logs.merge(updates)

        assert logs["losses"] == {"some_loss": 1}
        assert logs["metrics"] == {"some_metric": 2}

    def test_logs_updates(self):
        logs = ciclo.logs()
        logs.updates = {
            "losses": {"some_loss": 1},
            "metrics": {"some_metric": 2},
        }

        assert logs["losses"] == {"some_loss": 1}
        assert logs["metrics"] == {"some_metric": 2}

    def test_logs_updates_no_get(self):
        logs = ciclo.logs()

        with pytest.raises(AttributeError):
            logs.updates
