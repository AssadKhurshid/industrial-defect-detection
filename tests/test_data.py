import pytest

from idd.data import load_split


def make_category(root, with_mask=True):
    """Fake MVTec category: empty files are enough, the loader only reads the folder layout."""
    files = [
        "train/good/000.png",
        "train/good/001.png",
        "test/good/000.png",
        "test/scratch/000.png",
        "test/crack/000.png",
    ]
    if with_mask:
        files += ["ground_truth/scratch/000_mask.png", "ground_truth/crack/000_mask.png"]
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).touch()
    return root


def test_train_split_is_only_good(tmp_path):
    samples = load_split(make_category(tmp_path), "train")
    assert [(s.image.name, s.label, s.defect, s.mask) for s in samples] == [
        ("000.png", 0, "good", None),
        ("001.png", 0, "good", None),
    ]


def test_test_split_has_labels_defects_and_masks(tmp_path):
    root = make_category(tmp_path)
    samples = load_split(root, "test")
    # sorted by folder: crack, good, scratch
    assert [(s.defect, s.label) for s in samples] == [("crack", 1), ("good", 0), ("scratch", 1)]
    assert samples[0].mask == root / "ground_truth" / "crack" / "000_mask.png"
    assert samples[1].mask is None


def test_missing_mask_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing mask"):
        load_split(make_category(tmp_path, with_mask=False), "test")
