"""Dataset loaders for the AT&T (ORL) faces and MNIST, and a per-class train/test split."""
from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ATT_URL = "https://www.kaggle.com/api/v1/datasets/download/kasikrit/att-database-of-faces"
ATT_PAGE = "https://www.kaggle.com/datasets/kasikrit/att-database-of-faces"
ATT_SHAPE = (112, 92)
MNIST_SHAPE = (28, 28)
MNIST_N_TRAIN = 60_000  # OpenML keeps the original order: 60 000 training images, then 10 000 test images


def download_att(target: Path) -> None:
    """Download the AT&T faces zip from Kaggle and extract the folders s1 ... s40 into ``target``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "att.zip"
        try:
            urllib.request.urlretrieve(ATT_URL, zip_path)
        except OSError as err:
            raise RuntimeError(
                f"Could not download the AT&T faces ({err}). Download the zip from {ATT_PAGE} "
                f"and unzip it into {target} (folders s1 ... s40)."
            ) from err
        with zipfile.ZipFile(zip_path) as zf:
            root = target.resolve()
            for name in zf.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise RuntimeError(f"Unexpected path in the AT&T zip: {name}")
            zf.extractall(root)


def read_pgm(path: Path | str) -> np.ndarray:
    """Grey-scale image as a float64 array of pixel values 0-255."""
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.float64)


def load_att(data_dir: Path | str | None = None, download: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """AT&T faces: 40 people x 10 images of 112 x 92 pixels.

    ``data_dir`` is the folder holding s1 ... s40 (default ``data/att_faces``; downloaded if missing).
    Returns X of shape (400, 10304) with pixel values 0-255 and y, the person index 0-39.
    Rows are ordered by person, then by image number.
    """
    root = Path(data_dir) if data_dir else DATA_DIR / "att_faces"
    if not (root / "s1").is_dir():
        if not download:
            raise FileNotFoundError(f"AT&T faces not found in {root} (expected folders s1 ... s40)")
        download_att(root)
    images, labels = [], []
    for person in range(1, 41):
        files = sorted((root / f"s{person}").glob("*.pgm"), key=lambda p: int(p.stem))
        if len(files) != 10:
            raise FileNotFoundError(f"expected 10 images in {root / f's{person}'}, found {len(files)}")
        for f in files:
            img = read_pgm(f)
            if img.shape != ATT_SHAPE:
                raise ValueError(f"{f}: shape {img.shape}, expected {ATT_SHAPE}")
            images.append(img.ravel())
            labels.append(person - 1)
    return np.stack(images), np.asarray(labels)


def load_mnist(data_dir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MNIST (OpenML ``mnist_784``): 70 000 images of 28 x 28 pixels, cached in ``data_dir``.

    Returns X of shape (70000, 784) with pixel values 0-255 and y, the digit. The first
    ``MNIST_N_TRAIN`` rows are the standard training set, the rest the standard test set.
    """
    from sklearn.datasets import fetch_openml

    home = Path(data_dir) if data_dir else DATA_DIR
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True, data_home=str(home))
    return X.astype(np.float64), y.astype(np.int64)


def load(name: str, data_dir: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """``(X, y, image_shape)`` for ``name`` in {"att", "mnist"}."""
    if name == "att":
        att_dir = Path(data_dir) / "att_faces" if data_dir else None
        return (*load_att(att_dir), ATT_SHAPE)
    if name == "mnist":
        return (*load_mnist(data_dir), MNIST_SHAPE)
    raise ValueError(f"unknown dataset {name!r}")


def split_per_class(y: np.ndarray, n_train: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random split with exactly ``n_train`` training samples of every class; the rest is the test set."""
    train = np.concatenate([rng.choice(np.flatnonzero(y == c), n_train, replace=False) for c in np.unique(y)])
    train = np.sort(train)
    return train, np.setdiff1d(np.arange(len(y)), train)
