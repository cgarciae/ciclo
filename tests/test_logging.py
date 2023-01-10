import pytest

import ciclo


class TestLogging:
    def test_logs(self):
        logs = ciclo.logs()
        logs.add_loss("some_loss", 1)
        logs.add_metric("some_metric", 2)
        logs.add_stateful_metric("other_metric", 3)

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
