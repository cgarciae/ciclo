import dataclasses

import numpy as np

import ciclo


@dataclasses.dataclass(frozen=True)
class Shaped:
    shape: tuple


def get_fake_dataset(steps, shape):

    sample = Shaped(shape)

    for _ in range(steps):
        yield sample


class TestSchedules:
    def test_every(self):

        schedule = ciclo.every(steps=1)

        for elapsed, batch in ciclo.elapse(get_fake_dataset(10, (2, 3))):
            assert schedule(elapsed)

    def test_every_2(self):

        schedule = ciclo.every(steps=2)

        iterator = iter(ciclo.elapse(get_fake_dataset(4, (2, 3))))

        elapsed, batch = next(iterator)
        assert schedule(elapsed)

        elapsed, batch = next(iterator)
        assert not schedule(elapsed)

        elapsed, batch = next(iterator)
        assert schedule(elapsed)

        elapsed, batch = next(iterator)
        assert not schedule(elapsed)

    def test_every_offset(self):

        schedule = ciclo.every(steps=5, steps_offset=3)

        iterator = iter(ciclo.elapse(get_fake_dataset(10, (2, 3))))

        elapsed, batch = next(iterator)
        assert not schedule(elapsed)

        elapsed, batch = next(iterator)
        assert not schedule(elapsed)

        elapsed, batch = next(iterator)
        assert not schedule(elapsed)

        elapsed, batch = next(iterator)
        assert schedule(elapsed)

    def test_bug(self):
        def train_step():
            return ciclo.logs().add_metric("acc", 1)

        def test_step():
            return ciclo.logs().add_metric("acc", 0.8)

        state, history, elapsed = ciclo.loop(
            None,
            get_fake_dataset(100, (2, 3)),
            {
                ciclo.every(1): train_step,
                ciclo.every(10): [
                    ciclo.inner_loop(
                        "valid",
                        lambda state: ciclo.loop(
                            state,
                            get_fake_dataset(3, (2, 3)),
                            {
                                ciclo.every(1): test_step,
                            },
                        ),
                    ),
                ],
                # **ciclo.keras_bar(total=total_step),
            },
            # stop=total_step,
        )

        steps, acc, acc_valid = history.collect("steps", "acc", "acc_valid")
        assert steps[0] == 0
