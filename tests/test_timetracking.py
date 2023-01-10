import pytest

import ciclo


class TestTimetracking:
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
