import torch
import torch.nn.functional as F

from idd.patchcore import PatchCore, greedy_coreset


def tiny_backbone(x):
    """Stand-in for the CNN: two average-pooled maps at stride 8 and 16, like layer2 and layer3.
    No weights to download, so the test runs anywhere."""
    return F.avg_pool2d(x, 8), F.avg_pool2d(x, 16)


def test_greedy_coreset_covers_both_clusters():
    a = torch.zeros(50, 2)
    b = torch.full((50, 2), 10.0)
    chosen = greedy_coreset(torch.cat([a, b]), n=2)
    assert sorted(int(i) // 50 for i in chosen) == [0, 1]  # one point from each cluster


def test_patchcore_finds_the_defect_patch():
    torch.manual_seed(0)
    good = torch.rand(8, 3, 64, 64) * 0.1  # dim, slightly noisy parts
    model = PatchCore(backbone=tiny_backbone, coreset_ratio=0.5).fit([good[:4], good[4:]])

    defect = torch.rand(1, 3, 64, 64) * 0.1
    defect[:, :, 16:32, 40:56] = 1.0  # bright spot in rows 16-31, columns 40-55

    amap = model.anomaly_map(defect)[0]  # 8x8 grid, one cell per 8x8 pixels
    row, col = divmod(int(amap.argmax()), amap.shape[1])
    assert 2 <= row <= 3 and 5 <= col <= 6  # hottest cell lies on the spot
    assert model.score(defect) > 5 * model.score(torch.rand(1, 3, 64, 64) * 0.1)
