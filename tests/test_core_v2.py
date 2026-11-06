"""Deterministic checks for the isolated fixed Core-V2 specification."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    path = ROOT / "scripts/05_mnl/stage1" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prep = load("test_core_v2_prepare", "01_prepare_core_v2.py")
diag = load("test_core_v2_diagnostics", "02_diagnose_core_v2.py")
estimate = load("test_core_v2_estimator", "03_estimate_core_v2.py")


def source_row(**changes) -> pd.DataFrame:
    row = {"CHOICE_ID": "x", "H_ID": "h", "CHOICE": 1, "AV_1": 1, "AV_2": 1,
           "W_GEW": 1.5, "PT1_OTHER": 0, "PT1_MISSING": 0,
           "PT2_OTHER": 0, "PT2_MISSING": 0, "STATUS1_MISSING": 0,
           "STATUS2_MISSING": 0, "TOUR_DISTANCE_KM": 20,
           "TOUR_DISTANCE_COMPLETE_FLAG": 1,
           "TOUR_N_DISTINCT_PURPOSES": 2,
           "TOUR_PURPOSE_COUNT_0": 0, "TOUR_PURPOSE_COUNT_1": 0,
           "TOUR_PURPOSE_COUNT_2": 1, "TOUR_PURPOSE_COUNT_3PLUS": 0,
           "TOUR_PURPOSE_UNKNOWN_FLAG": 0,
           "TOUR_PURPOSE_FEATURES_AVAILABLE_FLAG": 1,
           "TOUR_N_NONHOME_STOPS": 2, "TOUR_MAIN_PURPOSE": 1,
           "TOUR_CLOSED": 1, "HP_SEX": 1, "erwerb": 1,
           "alter_gr5": 4, "oek_status": 3, "RegioStaR4": 11}
    row.update(changes)
    return pd.DataFrame([row])


class CoreV2SpecificationTests(unittest.TestCase):
    def test_fixed_structure_and_stage0_names(self):
        self.assertEqual((len(prep.MAIN), len(prep.INTERACTIONS), len(prep.PREDICTORS)), (5, 16, 21))
        self.assertEqual(len(set(prep.PREDICTORS)), 21)
        prep.verify_stage0_v2_specification()

    def test_vehicle_difference_and_interaction_formulas(self):
        row = {column: 0 for _, left, right in prep.MAIN for column in (left, right)}
        row.update(PT1_HEV=1, PT2_HEV=0, PT1_PHEV=0, PT2_PHEV=1,
                   PT1_BEV=0, PT2_BEV=1, STATUS1_LOW=1, STATUS2_LOW=0,
                   STATUS1_HIGH=0, STATUS2_HIGH=1)
        row.update(TOUR_DISTANCE_C_10KM=2, MULTI_PURPOSE=1, MAIN_PURPOSE_WORK=1,
                   MAIN_PURPOSE_ESCORT=0, MAIN_PURPOSE_SHOPPING=0, MALE=1,
                   EMPLOYED=1, AGE_UNDER30=1, VERY_HIGH_ECON=0, URBAN_REGION=1)
        frame = pd.DataFrame([row])
        design = prep.effective_design(frame)
        self.assertEqual(design.loc[0, list(prep.PREDICTORS[:5])].tolist(), [1, -1, -1, 1, -1])
        for name, context, target in prep.INTERACTIONS:
            self.assertEqual(design.at[0, name], frame.at[0, context]*design.at[0, target])
        np.testing.assert_allclose(design.to_numpy(), -prep.effective_design(frame, swap=True).to_numpy())

    def test_documented_context_rules_and_exclusions(self):
        out, reasons = prep.context_and_reasons(source_row())
        self.assertFalse(out[list(reasons)].any(axis=1).iloc[0])
        self.assertEqual((out.MULTI_PURPOSE.iloc[0], out.MALE.iloc[0],
                          out.EMPLOYED.iloc[0], out.AGE_UNDER30.iloc[0],
                          out.VERY_HIGH_ECON.iloc[0], out.URBAN_REGION.iloc[0]),
                         (1, 1, 1, 1, 0, 1))
        for changes, reason in [
            ({"PT1_OTHER": 1}, "invalid_v2_vehicle"),
            ({"STATUS2_MISSING": 1}, "invalid_v2_vehicle"),
            ({"CHOICE": 3}, "invalid_choice_or_availability"),
            ({"W_GEW": 0}, "invalid_weight_or_household"),
            ({"TOUR_DISTANCE_KM": -1}, "invalid_tour_distance"),
            ({"TOUR_DISTANCE_COMPLETE_FLAG": 0}, "incomplete_tour_distance"),
            ({"TOUR_N_DISTINCT_PURPOSES": 0, "TOUR_PURPOSE_COUNT_0": 1,
              "TOUR_PURPOSE_COUNT_2": 0, "TOUR_PURPOSE_UNKNOWN_FLAG": 1}, "uncertain_or_invalid_purpose_count"),
            ({"TOUR_MAIN_PURPOSE": "", "TOUR_CLOSED": 0}, "missing_or_invalid_main_purpose"),
            ({"HP_SEX": 3}, "invalid_sex"),
            ({"erwerb": 9}, "invalid_employment"),
            ({"alter_gr5": 3}, "invalid_age"),
            ({"oek_status": 9}, "invalid_economic_status"),
            ({"RegioStaR4": 99}, "invalid_urban_region"),
        ]:
            frame, _ = prep.context_and_reasons(source_row(**changes))
            self.assertTrue(frame[reason].iloc[0], (changes, reason))

    def test_distinct_purpose_binary_and_unknown_rule(self):
        for count, category, unknown, expected, ambiguous in [
            (0, "0", 1, None, True), (1, "1", 0, 0, False),
            (1, "1", 1, None, True), (2, "2", 0, 1, False),
            (2, "2", 1, 1, False), (3, "3PLUS", 0, 1, False)]:
            indicators = {f"TOUR_PURPOSE_COUNT_{name}": int(name == category)
                          for name in prep.PURPOSE_COUNTS}
            frame, _ = prep.context_and_reasons(source_row(
                TOUR_N_DISTINCT_PURPOSES=count, TOUR_PURPOSE_UNKNOWN_FLAG=unknown,
                TOUR_N_NONHOME_STOPS=1, **indicators))
            self.assertEqual(frame.uncertain_or_invalid_purpose_count.iloc[0], ambiguous)
            if expected is None:
                self.assertTrue(pd.isna(frame.MULTI_PURPOSE.iloc[0]))
            else:
                self.assertEqual(frame.MULTI_PURPOSE.iloc[0], expected)
        invalid, _ = prep.context_and_reasons(source_row(TOUR_PURPOSE_COUNT_1=1))
        self.assertTrue(invalid.uncertain_or_invalid_purpose_count.iloc[0])
        single_with_many_stops, _ = prep.context_and_reasons(source_row(
            TOUR_N_DISTINCT_PURPOSES=1, TOUR_N_NONHOME_STOPS=4,
            TOUR_PURPOSE_COUNT_1=1, TOUR_PURPOSE_COUNT_2=0))
        self.assertEqual(single_with_many_stops.MULTI_PURPOSE.iloc[0], 0)

    def test_repeated_purposes_count_once(self):
        valid = {1, 2, 3, 4, 5, 6, 7}
        for observations, expected in [([1, 1], 1), ([1, 4], 2), ([1, 4, 1], 2)]:
            distinct = len(set(observations) & valid)
            self.assertEqual(distinct, expected)
            category = str(distinct) if distinct < 3 else "3PLUS"
            indicators = {f"TOUR_PURPOSE_COUNT_{name}": int(name == category)
                          for name in prep.PURPOSE_COUNTS}
            frame, _ = prep.context_and_reasons(source_row(
                TOUR_N_DISTINCT_PURPOSES=distinct, TOUR_N_NONHOME_STOPS=len(observations),
                **indicators))
            self.assertEqual(frame.MULTI_PURPOSE.iloc[0], int(distinct >= 2))

    def test_partial_distance_is_excluded_even_with_numeric_total(self):
        frame, _ = prep.context_and_reasons(source_row(
            TOUR_DISTANCE_KM=17.5, TOUR_DISTANCE_COMPLETE_FLAG=0))
        self.assertTrue(frame.incomplete_tour_distance.iloc[0])
        self.assertFalse(frame.invalid_tour_distance.iloc[0])

    def test_sex_age_economic_urban_boundaries(self):
        female, _ = prep.context_and_reasons(source_row(HP_SEX=2, erwerb=0,
                                                        alter_gr5=6, oek_status=1, RegioStaR4=21))
        self.assertEqual((female.MALE.iloc[0], female.EMPLOYED.iloc[0],
                          female.AGE_UNDER30.iloc[0], female.VERY_HIGH_ECON.iloc[0],
                          female.URBAN_REGION.iloc[0]), (0, 0, 0, 0, 0))
        high, _ = prep.context_and_reasons(source_row(oek_status=5, alter_gr5=15, RegioStaR4=12))
        self.assertEqual((high.VERY_HIGH_ECON.iloc[0], high.AGE_UNDER30.iloc[0],
                          high.URBAN_REGION.iloc[0]), (1, 0, 1))
        for code in (1, 2, 3, 4):
            frame, _ = prep.context_and_reasons(source_row(oek_status=code))
            self.assertEqual(frame.VERY_HIGH_ECON.iloc[0], 0)
        invalid, _ = prep.context_and_reasons(source_row(oek_status=9))
        self.assertTrue(pd.isna(invalid.VERY_HIGH_ECON.iloc[0]))
        self.assertNotIn("ECON_STATUS_CENTERED", prep.PREDICTORS)

    def test_current_shared_source_hashes_and_rejection(self):
        expected = prep.current_source_hashes()
        self.assertEqual(set(expected), {
            "data_processed/model_input/single_driver_mnl_design_matrix.csv",
            "data_processed/model_input/allocation_candidate_model_input.csv",
            "data_processed/reconstruction/phase4/home_based_tour_features.csv",
            "scripts/05_mnl/01_estimate_v2_only_mnl.py",
            "scripts/05_mnl/stage1/01_prepare_core_v2.py",
        })
        prep.verify_current_sources(expected)
        old = prep.sha256
        protected = prep.SINGLE_DRIVER_MNL_DESIGN_MATRIX_PATH
        with patch.object(prep, "sha256", side_effect=lambda path: "0"*64 if path == protected else old(path)):
            with self.assertRaises(AssertionError):
                prep.verify_current_sources(expected)

    def test_output_isolation_and_collision(self):
        self.assertIn(prep.ROOT.relative_to(prep.OUTPUTS_DIR).as_posix(),
                      ("mnl/stage1/core_v2", "mnl/stage1/.core_v2_replacement_staging"))
        self.assertTrue((prep.ROOT / "data/core_v2_input_manifest.json").is_file())
        with self.assertRaises(FileExistsError):
            prep.prepare()
        self.assertEqual(estimate.OUT.relative_to(prep.ROOT).as_posix(), "estimation")
        if estimate.OUT.exists():
            with self.assertRaises(FileExistsError):
                estimate.verify_gate()
        else:
            self.assertEqual(estimate.verify_gate()[2]["status"], "PASS_NO_BLOCKER")

    def test_saved_sample_design_and_no_unintended_filter(self):
        sample, design, metadata, _ = diag.load_prepared()
        flow = pd.read_csv(prep.DATA / "core_v2_sample_flow.csv").set_index("metric")
        self.assertEqual(len(sample), int(flow.loc["final_core_v2", "count"]))
        self.assertEqual(int(flow.loc["initial_single_driver", "count"])-len(sample),
                         int(flow.loc["unique_exclusions", "count"]))
        self.assertEqual(len(design.columns), 27)
        self.assertTrue(sample.CHOICE_ID.equals(design.CHOICE_ID))
        self.assertEqual(metadata["sample"]["final_n"], 17093)
        self.assertEqual(set(design.columns[-21:]), set(prep.PREDICTORS))
        self.assertFalse(set(design.columns) & {"D_MULTI_ACTIVITY_BEV", "D_ECON_BEV", "D_ECON_STATUS_HIGH"})
        self.assertTrue(prep.numeric(sample.TOUR_DISTANCE_COMPLETE_FLAG).eq(1).all())
        self.assertTrue(prep.numeric(sample.MULTI_PURPOSE).isin([0, 1]).all())
        self.assertAlmostEqual(prep.numeric(sample.TOUR_DISTANCE_C_10KM).mean(), 0, places=11)
        self.assertAlmostEqual(prep.numeric(sample.TOUR_DISTANCE_KM).mean(), metadata["distance_mean_km"])
        self.assertEqual(int(flow.loc["incomplete_tour_distance", "count"]),
                         metadata["distance_incomplete_initial_n"])

    def test_full_rank_gate_and_zero_support_rejection(self):
        sample, design, _, _ = diag.load_prepared()
        metrics, _, _, _ = diag.matrix_diagnostics(design[list(prep.PREDICTORS)])
        self.assertEqual((metrics["rank"], metrics["scaled_rank"]), (21, 21))
        broken = design[list(prep.PREDICTORS)].copy()
        broken[prep.PREDICTORS[-1]] = 0
        with self.assertRaises(AssertionError):
            diag.matrix_diagnostics(broken)

    def test_independent_binary_likelihood_and_label_swap(self):
        x = np.array([[1., 0.], [-1., 1.], [0.5, -1.]])
        y = np.array([1., 0., 1.])
        w = np.array([1., 2., 1.5])
        beta = np.array([0.2, -0.1])
        old = estimate.binary_quantities(beta, x, y, w)
        swapped = estimate.binary_quantities(beta, -x, 1-y, w)
        self.assertAlmostEqual(old[0], swapped[0])
        np.testing.assert_allclose(old[3], 1-swapped[3])
        np.testing.assert_allclose(old[5], swapped[5])
        self.assertTrue(np.linalg.eigvalsh(old[2]).min() > 0)


if __name__ == "__main__":
    unittest.main()
