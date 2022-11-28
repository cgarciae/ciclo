import ciclo
from ciclo.api import inject


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
        a = inject(f, *args)
        b = inject(g, *args)

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
        a = inject(obj.f, *args)
        b = inject(obj.g, *args)

        assert a == 3
        assert b == 6
