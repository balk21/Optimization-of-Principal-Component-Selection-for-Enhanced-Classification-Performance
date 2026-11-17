import numpy as np
import pytest
from PIL import Image

from pcaselect import data


def _write_fake_att(root):
    for person in range(1, 41):
        folder = root / f"s{person}"
        folder.mkdir(parents=True)
        for i in range(1, 11):
            pixels = np.full(data.ATT_SHAPE, person, dtype=np.uint8)
            pixels[0, 0] = i
            Image.fromarray(pixels).save(folder / f"{i}.pgm")


def test_load_att_reads_folders_in_numeric_order(tmp_path):
    _write_fake_att(tmp_path)
    X, y = data.load_att(tmp_path, download=False)
    assert X.shape == (400, 112 * 92)
    assert X.dtype == np.float64
    assert np.array_equal(y, np.repeat(np.arange(40), 10))
    assert np.array_equal(X[:, 1], y + 1)                    # person s<k> is label k - 1
    assert np.array_equal(X[:10, 0], np.arange(1, 11))      # 1.pgm ... 10.pgm, not 1, 10, 2, ...


def test_load_att_without_download_fails_cleanly(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_att(tmp_path, download=False)


def test_split_per_class():
    y = np.repeat(np.arange(40), 10)
    train, test = data.split_per_class(y, 5, np.random.default_rng(0))
    assert len(train) == len(test) == 200
    assert not set(train) & set(test)
    assert np.all(np.bincount(y[train]) == 5)
    again, _ = data.split_per_class(y, 5, np.random.default_rng(0))
    assert np.array_equal(train, again)
