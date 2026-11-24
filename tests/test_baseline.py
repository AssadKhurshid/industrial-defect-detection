import numpy as np

from idd.baseline import PixelStatsBaseline


def test_defect_scores_higher_than_good():
    rng = np.random.default_rng(0)

    def part():  # a "photographed part": same gray level everywhere plus camera noise
        return (128 + rng.normal(0, 3, (64, 64))).astype(np.float32)

    model = PixelStatsBaseline().fit([part() for _ in range(20)])
    defect = part()
    defect[20:30, 20:30] = 255  # a bright 10x10 spot, like a scratch or stain

    assert model.score(defect) > 5 * model.score(part())
    assert model.anomaly_map(defect).argmax() // 64 in range(20, 30)  # hottest row is inside the spot
