"""Small synthetic regression tests; never call Biogeme estimation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from src.thesis_pipeline.model_input import tour_features

HERE = Path(__file__).resolve().parent
for ancestor in HERE.parents:
    if (ancestor / ".git").exists():
        sys.path.insert(0, str(ancestor))
        break


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prep = import_file("test_prepare_core_v1", HERE / "01_prepare_core_v1.py")
diagnose = import_file("test_diagnose_core_v1", HERE / "02_diagnose_core_v1.py")
encoder = import_file("test_single_driver_age_registry", HERE.parents[1] / "03_model_input" / "05_build_single_driver_mnl_design_matrix.py")


def example_context() -> pd.DataFrame:
    return pd.DataFrame({
        "TOUR_TRAVEL_TIME_MIN": [30, 60, 90, 120],
        "TOUR_N_NONHOME_STOPS": [0, 1, 2, 3],
        "HOUSEHOLD_ACCOMPANIED": [0, 1, 0, 1],
        "HP_SEX": [1, 2, 1, 2],
        "alter_gr5": [4, 8, 11, 15],
        "household_type": ["young_household", "family_household", "adult_household", "senior_household"],
        "RegioStaR4": [11, 12, 21, 22],
    }).pipe(lambda frame: pd.concat([
        frame,
        encoder.apply_categorical(frame.assign(CHOICE_ID=["a", "b", "c", "d"]),
            next(spec for spec in encoder.TRANSFORMATION_REGISTRY if spec.key == "age_band"))[0]
    ], axis=1))


def example_vehicle() -> pd.DataFrame:
    frame, _ = prep.encode_context(example_context())
    frame["CHOICE"] = [1, 2, 1, 2]
    frame["AV_1"] = [1, 1, 1, 0]
    frame["AV_2"] = [1, 1, 1, 1]
    frame["H_ID"] = ["a", "b", "c", "d"]
    frame["W_GEW"] = [1.0, 2.0, 3.0, 4.0]
    for name, left, right in prep.MAIN:
        frame[left] = [1, 0, 1, 0] if "AGE" not in name else [10., 7., 4., 9.]
        frame[right] = [0, 1, 0, 1] if "AGE" not in name else [5., 9., 8., 3.]
    return frame


class CoreV1Tests(unittest.TestCase):
    def test_script05_age_registry_and_stage1_mapping(self):
        codes = [3, *range(4, 16), None]
        frame = pd.DataFrame({"CHOICE_ID": [str(i) for i in range(len(codes))],
                              "alter_gr5": codes})
        spec = next(item for item in encoder.TRANSFORMATION_REGISTRY if item.key == "age_band")
        self.assertEqual(spec.recommended_reference, "AGE_30_49")
        encoded, _ = encoder.apply_categorical(frame, spec)
        self.assertEqual(encoded[prep.AGE_COLUMNS].sum(axis=1).tolist(), [1] * len(codes))
        self.assertEqual(encoded.AGE_UNDER18.tolist(), [1] + [0] * 13)
        self.assertEqual(encoded.AGE_UNDER30.tolist(), [0, 1, 1] + [0] * 11)
        self.assertEqual(encoded.AGE_30_49.tolist(), [0] * 3 + [1] * 4 + [0] * 7)
        self.assertEqual(encoded.AGE_50PLUS.tolist(), [0] * 7 + [1] * 6 + [0])
        self.assertEqual(encoded.AGE_MISSING.tolist(), [0] * 13 + [1])
        checked = pd.concat([frame, encoded], axis=1)
        checked["TOUR_TRAVEL_TIME_MIN"] = 60
        checked["TOUR_N_NONHOME_STOPS"] = 1
        checked["HOUSEHOLD_ACCOMPANIED"] = 0
        checked["HP_SEX"] = 1
        checked["household_type"] = "family_household"
        checked["RegioStaR4"] = 11
        result, _ = prep.encode_context(checked)
        self.assertTrue(result.loc[0, "under18_driver"])
        self.assertFalse(result.loc[1, "under18_driver"])
        self.assertTrue(result.loc[len(codes)-1, "invalid_age"])
        checked.loc[1, "AGE_UNDER30"] = 0
        with self.assertRaisesRegex(AssertionError, "disagrees"):
            prep.encode_context(checked)

    def test_tour_driving_accompaniment_truth_table(self):
        cases = [
            ("yes_late", [1, 1, 1], [2, 716, 1], 1),
            ("all_no", [1, 1, 1], [2, 716, 2], 0),
            ("unknown_response", [1, 1], [2, 999], None),
            ("yes_over_unknown", [1, 1], [1, 999], 1),
            ("passenger_only", [2], [1], None),
            ("noncar_only", [706], [1], None),
            ("ambiguous_driver", [1, 3], [2, 2], None),
            ("missing_driver", [1, None], [2, 2], None),
        ]
        rows = []
        for name, statuses, companions, _ in cases:
            for index, (status, companion) in enumerate(zip(statuses, companions)):
                rows.append({"TOUR_ID": name, "PERSON_KEY": name,
                             "SOURCE_ROW_ID": f"{name}-{index}",
                             "pkw_fmf": status, "W_BEGL_HH": companion})
        result = tour_features.tour_household_accompaniment(pd.DataFrame(rows))
        for name, _, _, expected in cases:
            if expected is None:
                self.assertTrue(pd.isna(result[name]))
            else:
                self.assertEqual(result[name], expected)
        mixed = pd.DataFrame(rows)
        mixed.loc[mixed.TOUR_ID.eq("all_no"), "PERSON_KEY"] = "different_person"
        mixed.loc[mixed.SOURCE_ROW_ID.eq("all_no-0"), "PERSON_KEY"] = "all_no"
        with self.assertRaisesRegex(AssertionError, "different persons"):
            tour_features.tour_household_accompaniment(mixed)

    def test_context_references_and_stop_zero(self):
        encoded, mean = prep.encode_context(example_context())
        self.assertEqual(mean, 75)
        self.assertEqual(encoded.MULTISTOP.tolist(), [0, 0, 1, 1])
        self.assertEqual(encoded.FEMALE.tolist(), [0, 1, 0, 1])
        self.assertEqual(encoded.AGE_UNDER30.tolist(), [1, 0, 0, 0])
        self.assertEqual(encoded.AGE_30_49.tolist(), [0, 1, 0, 0])
        self.assertEqual(encoded.AGE_50PLUS.tolist(), [0, 0, 1, 1])
        self.assertEqual(encoded.FAMILY.tolist(), [0, 1, 0, 0])
        self.assertEqual(encoded.URBAN_REGION.tolist(), [1, 1, 0, 0])
        self.assertEqual(encoded.TOUR_TIME_C_H.tolist(), [-.75, -.25, .25, .75])
        self.assertAlmostEqual(encoded.TOUR_TIME_C_H.mean(), 0)

    def test_invalid_context_exclusion_flags(self):
        frame = example_context()
        frame.loc[0, "TOUR_N_NONHOME_STOPS"] = np.nan
        frame.loc[1, "HP_SEX"] = 3
        frame.loc[2, "household_type"] = "unknown"
        frame.loc[3, "RegioStaR4"] = 99
        frame.loc[0, "HOUSEHOLD_ACCOMPANIED"] = np.nan
        encoded, _ = prep.encode_context(frame, 75)
        self.assertTrue(encoded.loc[0, "invalid_stops"])
        self.assertTrue(encoded.loc[0, "invalid_accompaniment"])
        self.assertTrue(encoded.loc[1, "invalid_sex_including_diverse"])
        self.assertTrue(encoded.loc[2, "invalid_household_type"])
        self.assertTrue(encoded.loc[3, "invalid_regiostar4"])
        self.assertTrue(np.isnan(encoded.loc[0, "MULTISTOP"]))
        self.assertTrue(np.isnan(encoded.loc[1, "FEMALE"]))

    def test_design_formulas_labels_and_determinism(self):
        frame = example_vehicle()
        first = prep.effective_design(frame)
        second = prep.effective_design(frame)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(list(first), prep.PREDICTORS)
        self.assertEqual(first.shape[1], 26)
        self.assertEqual(len(prep.INTERACTIONS), 19)
        for name, left, right in prep.MAIN:
            np.testing.assert_array_equal(first[name], frame[left] - frame[right])
        for name, context, target in prep.INTERACTIONS:
            np.testing.assert_array_equal(first[name], frame[context] * first[target])
        np.testing.assert_allclose(first, -prep.effective_design(frame, swap=True))

    def test_rank_and_zero_variance(self):
        deficient = np.array([[1., 2., 0.], [2., 4., 0.], [3., 6., 0.], [4., 8., 0.]])
        rank, _, _ = diagnose.numerical_rank(deficient)
        self.assertEqual(rank, 1)
        self.assertEqual(np.linalg.norm(deficient, axis=0).tolist()[-1], 0)
        full = np.array([[1., 0.], [0., 1.], [1., 1.]])
        self.assertEqual(diagnose.numerical_rank(full)[0], 2)

    def test_availability_excludes_deterministic_support(self):
        frame = example_vehicle()
        design = prep.effective_design(frame)
        support, _ = diagnose.interaction_support(frame, design)
        row = support.set_index("predictor").loc["D_FEMALE_BEV"]
        self.assertEqual(row.context_active_all, 2)
        self.assertEqual(row.context_active_both_available, 1)
        self.assertEqual(row.nonzero_target_delta_active_both, 1)

    def test_stage0_specification_matches(self):
        accepted = prep.load_v1()
        self.assertEqual([r[0] for r in accepted.DELTA_X_SPECS], [r[0] for r in prep.MAIN])
        self.assertEqual([(r[2], r[3]) for r in accepted.DELTA_X_SPECS],
                         [(r[1], r[2]) for r in prep.MAIN])

    def test_target_always_chosen_despite_alternating_car_labels(self):
        frame = example_vehicle()
        frame[["AV_1", "AV_2"]] = 1
        frame["CHOICE"] = [1, 2, 1, 2]
        support, _ = diagnose.interaction_support(frame, prep.effective_design(frame))
        row = support.set_index("predictor").loc["D_TIME_BEV"]
        self.assertEqual((row.chosen_car1_n, row.chosen_car2_n), (2, 2))
        self.assertEqual((row.chosen_target_n, row.chosen_non_target_n), (4, 0))
        self.assertIn("target always chosen", row.warning)

    def test_target_never_chosen(self):
        frame = example_vehicle()
        frame[["AV_1", "AV_2"]] = 1
        frame["CHOICE"] = [2, 1, 2, 1]
        support, _ = diagnose.interaction_support(frame, prep.effective_design(frame))
        row = support.set_index("predictor").loc["D_TIME_BEV"]
        self.assertEqual((row.chosen_target_n, row.chosen_non_target_n), (0, 4))
        self.assertIn("target never chosen", row.warning)

    def test_balanced_target_choice_and_delta_subsets(self):
        frame = example_vehicle()
        frame[["AV_1", "AV_2"]] = 1
        frame["CHOICE"] = [1, 1, 1, 1]
        support, _ = diagnose.interaction_support(frame, prep.effective_design(frame))
        row = support.set_index("predictor").loc["D_TIME_BEV"]
        self.assertEqual((row.chosen_target_n, row.chosen_non_target_n), (2, 2))
        self.assertEqual((row.positive_delta_chosen_target_n, row.negative_delta_chosen_target_n), (2, 0))
        self.assertNotIn("target always chosen", row.warning)
        self.assertNotIn("target never chosen", row.warning)

    def test_chosen_vehicle_age_direction(self):
        frame = example_vehicle()
        frame[["AV_1", "AV_2"]] = 1
        frame["VEHICLE_AGE_2"] = [5., 9., 8., 9.]
        frame["CHOICE"] = [1, 2, 1, 2]
        result = diagnose.vehicle_age_choice_support(frame).set_index("subset").loc["all_stage1"]
        self.assertEqual((result.chosen_older_n, result.chosen_younger_n, result.same_age_n), (2, 1, 1))
        self.assertAlmostEqual(result.chosen_minus_nonchosen_mean, (5 + 2 - 4) / 3)

    def test_time_bin_boundaries_and_choice_reconciliation(self):
        frame = pd.concat([example_vehicle(), example_vehicle().iloc[:2]], ignore_index=True)
        frame["TOUR_TRAVEL_TIME_MIN"] = [0, 30, 60, 120, 240, 241]
        frame, _ = prep.encode_context(frame, mean_minutes=0)
        frame[["AV_1", "AV_2"]] = 1
        frame["CHOICE"] = [1, 2, 1, 2, 1, 2]
        support, bins = diagnose.interaction_support(frame, prep.effective_design(frame))
        bev = bins.loc[bins.predictor.eq("D_TIME_BEV")]
        self.assertEqual(bev.informative_contrasts_n.tolist(), [2, 1, 1, 1, 1])
        overall = support.set_index("predictor").loc["D_TIME_BEV"]
        self.assertEqual(bev.chosen_target_n.sum(), overall.chosen_target_n)
        self.assertEqual(bev.chosen_non_target_n.sum(), overall.chosen_non_target_n)
        self.assertEqual(bev.informative_contrasts_n.sum(), overall.informative_contrasts_n)

    def test_weight_concentration_equal_and_unequal(self):
        frame = example_vehicle()
        frame[["AV_1", "AV_2"]] = 1
        frame["W_GEW"] = [1., 1., 1., 1.]
        design = prep.effective_design(frame)
        equal = diagnose.weighted_interaction_support(frame, design).set_index("predictor").loc["D_TIME_BEV"]
        self.assertEqual(equal.kish_n_eff_descriptive, 4)
        self.assertEqual(equal.kish_n_eff_per_occasion, 1)
        frame["W_GEW"] = [1., 1., 1., 7.]
        frame["H_ID"] = ["a", "a", "b", "a"]
        unequal = diagnose.weighted_interaction_support(frame, design).set_index("predictor").loc["D_TIME_BEV"]
        self.assertAlmostEqual(unequal.kish_n_eff_descriptive, 100 / 52)
        self.assertAlmostEqual(unequal.max_individual_weight_share, .7)
        self.assertAlmostEqual(unequal.max_household_weight_share, .9)
        self.assertEqual(unequal.contributing_households_n, 2)

    def test_source_integrity_during_diagnostics(self):
        protected = [diagnose.DATA / "core_v1_context_sample.csv",
                     diagnose.DATA / "core_v1_effective_design.csv",
                     diagnose.DATA / "core_v1_transformations.json"]
        before = {str(p): prep.sha256(p) for p in protected}
        diagnose.diagnose()
        self.assertEqual(before, {str(p): prep.sha256(p) for p in protected})

    def test_reproducible_diagnostic_files(self):
        diagnose.diagnose()
        paths = sorted(p for p in diagnose.ROOT.rglob("*") if p.is_file())
        before = {str(p): prep.sha256(p) for p in paths}
        diagnose.diagnose()
        self.assertEqual(before, {str(p): prep.sha256(p) for p in paths})


if __name__ == "__main__":
    unittest.main()
