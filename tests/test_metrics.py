import pytest

from idd.metrics import auroc, f1_max


@pytest.mark.parametrize("scores, expected", [
    ([0.1, 0.2, 0.8, 0.9], 1.0),  # every defect above every good image
    ([0.8, 0.9, 0.1, 0.2], 0.0),  # every defect below
    ([0.5, 0.5, 0.5, 0.5], 0.5),  # all tied: coin flip
    ([0.1, 0.8, 0.2, 0.9], 0.75),  # 3 of the 4 (defect, good) pairs ordered right
])
def test_auroc(scores, expected):
    assert auroc([0, 0, 1, 1], scores) == pytest.approx(expected)


def test_f1_max_picks_best_threshold():
    # Threshold 0.6 catches both defects (0.6, 0.9) with one false alarm (0.7): F1 = 2*2 / (2+1+2) = 0.8.
    f1, threshold = f1_max([0, 1, 0, 1], [0.1, 0.6, 0.7, 0.9])
    assert f1 == pytest.approx(0.8)
    assert threshold == 0.6
