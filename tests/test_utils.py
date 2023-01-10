import ciclo


class TestUtils:
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
