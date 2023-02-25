import numpy as np

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

    def test_elapse_batch_size(self):
        def dataset():
            for i in range(3):
                yield {"x": np.empty((3, 2)), "y": np.empty((4, 1))}

        for i, (elapsed, batch) in enumerate(ciclo.elapse(dataset())):
            assert elapsed.samples == 4 * i

    def test_elapse_batch_size_fn(self):
        def dataset():
            for i in range(3):
                yield {"x": np.empty((3, 2)), "y": np.empty((4, 1))}

        def min_first_axis(shapes):
            return min(shape[0] for shape in shapes)

        for i, (elapsed, batch) in enumerate(
            ciclo.elapse(dataset(), batch_size_fn=min_first_axis)
        ):
            assert elapsed.samples == 3 * i

    def test_elapse_batch_size_empty_pytree(self):
        def dataset():
            for i in range(3):
                yield {}

        for i, (elapsed, batch) in enumerate(ciclo.elapse(dataset())):
            assert elapsed.samples == i

    def test_elapse_batch_size_scalars(self):
        def dataset():
            return range(3)

        for i, (elapsed, batch) in enumerate(ciclo.elapse(dataset())):
            assert elapsed.samples == i
            assert batch == i
