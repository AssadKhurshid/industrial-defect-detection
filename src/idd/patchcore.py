"""PatchCore-lite: anomaly detection with pretrained CNN features and nearest neighbours.

Idea (Roth et al., 2022, "Towards Total Recall in Industrial Anomaly Detection"):
1. A CNN pretrained on ImageNet turns every small region (patch) of an image into a feature vector.
2. Fit: collect the patch features of all good training images (the "memory bank"), then keep a
   spread-out subset of them (greedy coreset) so the bank stays small.
3. Score: for each patch of a new image, the distance to its nearest neighbour in the bank.
   A patch far from every good patch is anomalous; the image score is the largest distance.

"Lite": no per-image reweighting of scores and a plain greedy coreset; enough to show the method.
"""

import torch
import torch.nn.functional as F

from idd.preprocess import to_input


def load_image(path, size: int = 224) -> torch.Tensor:
    """Image file -> (3, size, size) tensor, prepared like the backbone's ImageNet training images."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image {path}")
    return torch.from_numpy(to_input(image, size))


def wide_resnet50_backbone():
    """Pretrained Wide ResNet-50 returning its layer2 and layer3 feature maps (mid-level features:
    edges and textures, not ImageNet class meanings)."""
    from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

    net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()

    def forward(x):
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        layer2 = net.layer2(net.layer1(x))
        return layer2, net.layer3(layer2)

    forward.net = net  # so .to(device) can move the weights
    return forward


def greedy_coreset(features: torch.Tensor, n: int, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Indices of n rows chosen so every row is close to some chosen row (k-center greedy):
    start anywhere, then repeatedly add the row farthest from everything chosen so far.

    Distances are computed on a random projection to proj_dim dimensions: it keeps distances
    roughly the same (Johnson-Lindenstrauss) and makes the search ~10x faster.
    """
    generator = torch.Generator().manual_seed(seed)
    if features.shape[1] > proj_dim:
        proj = torch.randn(features.shape[1], proj_dim, generator=generator).to(features)
        features = features @ proj
    first = int(torch.randint(len(features), (1,), generator=generator))
    chosen = [first]
    min_dist = torch.cdist(features, features[first : first + 1]).squeeze(1)
    for _ in range(n - 1):
        nxt = int(min_dist.argmax())
        chosen.append(nxt)
        min_dist = torch.minimum(min_dist, torch.cdist(features, features[nxt : nxt + 1]).squeeze(1))
    return torch.tensor(chosen)


class PatchCore:
    def __init__(self, backbone=None, coreset_ratio: float = 0.1, device: str = "cpu"):
        self.backbone = backbone or wide_resnet50_backbone()
        if hasattr(self.backbone, "net"):
            self.backbone.net.to(device)
        self.coreset_ratio = coreset_ratio
        self.device = device

    @torch.no_grad()
    def patch_features(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """(B, 3, H, W) -> (B, h*w, C) patch features and the patch grid size (h, w)."""
        layer2, layer3 = self.backbone(images.to(self.device))
        # Average each feature with its 3x3 neighbours, so a patch also "sees" a bit of context.
        layer2 = F.avg_pool2d(layer2, 3, stride=1, padding=1)
        layer3 = F.avg_pool2d(layer3, 3, stride=1, padding=1)
        # layer3 is coarser: upsample it to layer2's grid and stack both feature sets per patch.
        layer3 = F.interpolate(layer3, size=layer2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([layer2, layer3], dim=1)
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)

    def fit(self, batches) -> "PatchCore":
        """batches: iterable of (B, 3, H, W) tensors of good images."""
        bank = torch.cat([self.patch_features(batch)[0].flatten(0, 1) for batch in batches])
        n = max(1, int(len(bank) * self.coreset_ratio))
        self.memory = bank[greedy_coreset(bank, n)]
        return self

    @torch.no_grad()
    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, h, w): each patch's distance to its nearest good patch."""
        feats, (h, w) = self.patch_features(images)
        dist = torch.stack([torch.cdist(f, self.memory).min(dim=1).values for f in feats])
        return dist.reshape(-1, h, w)

    def score(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B,) image scores: the most anomalous patch."""
        return self.anomaly_map(images).flatten(1).max(dim=1).values
