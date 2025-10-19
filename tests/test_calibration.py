import numpy as np

from services.calibration import (
    ScoreCalibrator,
    reliability_curve,
    brier_score,
    log_loss,
    CalibrationStore,
    CalibratorArtifact,
)


def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    # raw scores in [0, 1]
    s = rng.uniform(0, 1, size=n)
    # true probability is a sigmoid-transformed version of a rescaled score
    a = 4.0
    p_true = 1 / (1 + np.exp(-a * (s - 0.5)))
    y = rng.binomial(1, p_true, size=n)
    return s, y, p_true


def test_isotonic_calibration_improves_metrics():
    s, y, p_true = make_synthetic(n=3000)

    # baseline uses raw score as probability (miscalibrated)
    base_brier = brier_score(y, s)
    base_logloss = log_loss(y, s)

    # fit isotonic calibrator
    cal = ScoreCalibrator(method="isotonic")
    art = cal.fit(scores=s, labels=y, meta={"desc": "synthetic"}, valid_days=7)
    p_cal = cal.predict(s)

    # metrics should improve after calibration
    cal_brier = brier_score(y, p_cal)
    cal_logloss = log_loss(y, p_cal)

    assert cal_brier < base_brier
    assert cal_logloss < base_logloss

    # reliability curve should get closer to diagonal compared to baseline
    base_centers, base_fracs, _ = reliability_curve(y, s, n_bins=10)
    cal_centers, cal_fracs, _ = reliability_curve(y, p_cal, n_bins=10)

    # compute mean absolute gap to diagonal y=x by bin centers
    base_gap = float(np.mean(np.abs(base_centers - base_fracs)))
    cal_gap = float(np.mean(np.abs(cal_centers - cal_fracs)))

    assert cal_gap < base_gap

    # distribution comparison: calibrated probabilities should be within [0,1]
    assert (p_cal >= 0).all() and (p_cal <= 1).all()
    # and the mean predicted probability should be close to the empirical rate
    assert abs(p_cal.mean() - y.mean()) < abs(s.mean() - y.mean())


def test_quantile_calibration_and_persistence(tmp_path):
    s, y, _ = make_synthetic(n=1500, seed=7)
    cal = ScoreCalibrator(method="quantile", n_bins=8)
    art = cal.fit(scores=s, labels=y, meta={"method": "quantile"}, valid_days=3)

    # prediction works
    p = cal.predict(s)
    assert len(p) == len(s)
    assert (p >= 0).all() and (p <= 1).all()

    # persistence roundtrip
    store = CalibrationStore(base_dir=str(tmp_path))
    key = "unit_test_key"
    path = store.save(key, art)
    loaded = store.load(key)
    assert loaded is not None
    assert loaded.method == "quantile"
    assert loaded.bin_edges is not None and len(loaded.bin_edges) == len(art.bin_edges)
