"""Read the MVTec AD folder layout of one category into a list of samples.

Layout:
    <category>/train/good/*.png                     only defect-free images
    <category>/test/good/*.png                      defect-free test images
    <category>/test/<defect>/*.png                  one folder per defect type
    <category>/ground_truth/<defect>/<name>_mask.png  pixel mask for each defect image
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    image: Path
    label: int  # 0 = good, 1 = defect
    defect: str  # "good" or the defect type, e.g. "scratch"
    mask: Path | None  # pixel mask for defect images, None for good ones


def load_split(category_dir: str | Path, split: str) -> list[Sample]:
    """All samples of one category's "train" or "test" split, in a stable (sorted) order."""
    if split not in ("train", "test"):
        raise ValueError(f'split must be "train" or "test", got {split!r}')
    category_dir = Path(category_dir)
    split_dir = category_dir / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"no {split}/ folder in {category_dir}")

    samples = []
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for image in sorted(defect_dir.glob("*.png")):
            if defect == "good":
                samples.append(Sample(image, 0, defect, None))
                continue
            mask = category_dir / "ground_truth" / defect / f"{image.stem}_mask.png"
            if not mask.is_file():  # every MVTec defect image has a mask; a missing one means broken data
                raise FileNotFoundError(f"missing mask for {image}: expected {mask}")
            samples.append(Sample(image, 1, defect, mask))
    return samples
