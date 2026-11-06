"""Synthetic numerical tests for Stage-5 estimation and H_ID covariance."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
for ancestor in HERE.parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break
spec = importlib.util.spec_from_file_location("test_stage5", HERE / "03_estimate_core_v1.py")
stage5 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = stage5
spec.loader.exec_module(stage5)


def synthetic():
    rng = np.random.default_rng(721)
    x = rng.normal(size=(80, 3))
    p = stage5.expit(x @ np.array([.3, -.4, .2]))
    y = rng.binomial(1, p).astype(float)
    w = rng.uniform(.4, 2.0, size=len(y))
    return x, y, w


class Stage5NumericalTests(unittest.TestCase):
    def test_gradient_and_hessian_finite_differences(self):
        x, y, w = synthetic()
        beta = np.array([.2, -.3, .1])
        ll, gradient, information, *_ = stage5.binary_quantities(beta, x, y, w)
        step = 1e-5
        numeric_gradient = []
        numeric_hessian = []
        for j in range(len(beta)):
            direction = np.eye(len(beta))[j] * step
            plus = stage5.binary_quantities(beta+direction, x, y, w)
            minus = stage5.binary_quantities(beta-direction, x, y, w)
            numeric_gradient.append((plus[0]-minus[0])/(2*step))
            numeric_hessian.append(-(plus[1]-minus[1])/(2*step))
        np.testing.assert_allclose(gradient, numeric_gradient, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(information, np.array(numeric_hessian).T, atol=1e-8, rtol=1e-8)

    def test_single_observation_households_equal_individual_hc0(self):
        x, y, w = synthetic()
        hid = stage5.pd.Series([f"h{i}" for i in range(len(y))], dtype="string")
        beta = stage5.scipy_fit(x, y, w)["beta"]
        qa = stage5.covariance_qa(x, y, w, hid, beta)
        np.testing.assert_allclose(qa["covariances"]["HID_cluster_HC0"], qa["covariances"]["HC0"], atol=1e-12)

    def test_repeated_household_score_aggregation(self):
        x, y, w = synthetic()
        hid = stage5.pd.Series([f"h{i//2}" for i in range(len(y))], dtype="string")
        beta = stage5.scipy_fit(x, y, w)["beta"]
        qa = stage5.covariance_qa(x, y, w, hid, beta)
        expected_cluster_scores = np.array([
            qa["scores"][hid.eq(key).to_numpy()].sum(axis=0)
            for key in sorted(hid.unique())
        ])
        np.testing.assert_allclose(qa["cluster_scores"], expected_cluster_scores, atol=1e-12)
        bread = qa["covariances"]["model"]
        expected = bread @ (expected_cluster_scores.T @ expected_cluster_scores) @ bread
        np.testing.assert_allclose(qa["covariances"]["HID_cluster_HC0"], expected, atol=1e-12)

    def test_swapped_labels_same_estimate_and_covariance(self):
        x, y, w = synthetic()
        hid = stage5.pd.Series([f"h{i//2}" for i in range(len(y))], dtype="string")
        original = stage5.scipy_fit(x, y, w)
        swapped = stage5.scipy_fit(-x, 1-y, w)
        np.testing.assert_allclose(original["beta"], swapped["beta"], atol=1e-6)
        cov1 = stage5.covariance_qa(x, y, w, hid, original["beta"])["covariances"]["HID_cluster_HC0"]
        cov2 = stage5.covariance_qa(-x, 1-y, w, hid, swapped["beta"])["covariances"]["HID_cluster_HC0"]
        np.testing.assert_allclose(cov1, cov2, atol=1e-8)

    def test_common_weight_scaling_invariant_sandwich(self):
        x, y, w = synthetic()
        hid = stage5.pd.Series([f"h{i//2}" for i in range(len(y))], dtype="string")
        first = stage5.scipy_fit(x, y, w)
        second = stage5.scipy_fit(x, y, 7*w)
        np.testing.assert_allclose(first["beta"], second["beta"], atol=1e-6)
        cov1 = stage5.covariance_qa(x, y, w, hid, first["beta"])["covariances"]["HID_cluster_HC0"]
        cov2 = stage5.covariance_qa(x, y, 7*w, hid, second["beta"])["covariances"]["HID_cluster_HC0"]
        np.testing.assert_allclose(cov1, cov2, atol=1e-8)

    def test_parameter_reordering(self):
        x, y, w = synthetic()
        hid = stage5.pd.Series([f"h{i//2}" for i in range(len(y))], dtype="string")
        fit = stage5.scipy_fit(x, y, w)
        order = [2, 0, 1]
        reordered = stage5.scipy_fit(x[:, order], y, w)
        np.testing.assert_allclose(reordered["beta"], fit["beta"][order], atol=1e-6)
        original_cov = stage5.covariance_qa(x, y, w, hid, fit["beta"])["covariances"]["HID_cluster_HC0"]
        reordered_cov = stage5.covariance_qa(x[:, order], y, w, hid, reordered["beta"])["covariances"]["HID_cluster_HC0"]
        np.testing.assert_allclose(reordered_cov, original_cov[np.ix_(order, order)], atol=1e-8)

    def test_frozen_input_validation(self):
        before = stage5.verify_protected()
        sample, effective, metadata = stage5.load_accepted()
        self.assertEqual(len(sample), metadata["sample"]["stage1_n"])
        self.assertEqual(sample.H_ID.nunique(), metadata["sample"]["households"])
        self.assertEqual(len(stage5.PREDICTORS), 26)
        self.assertEqual(list(effective.columns[-len(stage5.PREDICTORS):]), list(stage5.PREDICTORS))
        self.assertEqual(before, stage5.verify_protected())


if __name__ == "__main__":
    unittest.main()
