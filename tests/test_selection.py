import numpy as np

from pcaselect import selection


def test_slope_is_numpy_gradient():
    ks = np.arange(1, 50, 5)
    e = 1000.0 / ks
    assert np.allclose(selection.slope(ks, e), np.gradient(e, ks))


def test_report_rule_takes_first_flat_point():
    ks = np.arange(1, 101, 10)
    e = np.r_[np.linspace(500, 100, 6), [100, 100, 100, 100]]  # flat from k = 61 on
    # np.gradient uses central differences, so k = 51 still has slope -4
    assert selection.report_rule(ks, e) == 61


def test_report_rule_falls_back_to_largest_slope():
    ks = np.arange(1, 101, 10).astype(float)
    e = 0.02 * (ks - 50) ** 2 - 5 * ks + 1000  # slope -5 + 0.04 (k - 50): never ~0, largest at k = 91
    d = np.gradient(e, ks)
    assert np.all(np.abs(d) > 1e-2)
    assert selection.report_rule(ks, e) == int(ks[np.argmax(d)])


def test_report_rule_tolerance_is_absolute():
    ks = np.arange(1, 101, 10)
    e = 1000.0 * np.exp(-ks / 10.0)
    # dividing the error by 255 (pixels in 0-1 instead of 0-255) moves the selected k
    assert selection.report_rule(ks, e / 255) < selection.report_rule(ks, e)


def test_knee_of_a_broken_line():
    ks = np.arange(1, 101)
    e = np.where(ks <= 20, 1000 - 45 * (ks - 1), 145 - (ks - 20))  # steep until k = 20, then nearly flat
    assert selection.knee_rule(ks, e) == 20


def test_knee_is_where_slope_equals_the_chord():
    ks = np.arange(1, 401).astype(float)
    e = 3000 * np.exp(-ks / 40) + 500 * (1 - ks / 400)
    k = selection.knee_rule(ks, e)
    chord = (e[-1] - e[0]) / (ks[-1] - ks[0])
    d = np.gradient(e, ks)
    assert abs(d[k - 1] - chord) < abs(d[k - 11] - chord)
    assert abs(d[k - 1] - chord) < abs(d[k + 9] - chord)


def test_variance_rule():
    ratio = np.array([0.5, 0.25, 0.125, 0.0625, 0.0625])  # cumulative 0.5, 0.75, 0.875, 0.9375, 1
    assert selection.variance_rule(ratio, 0.5) == 1
    assert selection.variance_rule(ratio, 0.75) == 2
    assert selection.variance_rule(ratio, 0.8) == 3
    assert selection.variance_rule(ratio, 0.95) == 5
    assert selection.variance_rule(ratio, 1.0) == 5
