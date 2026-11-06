"""Focused synthetic regression checks for purpose construction and comparison."""
from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from src.thesis_pipeline.model_input import purpose_qa, tour_features, vehicle_choice


def trips(*rows: dict) -> pd.DataFrame:
    defaults = dict(TOUR_ID="t", PERSON_KEY="p", H_ID="h", CHAIN_ELIGIBLE="1",
                    W_RBW="0", pkw_fmf="1", W_WAUTO="1", FLEET_CAR_SET="1|2",
                    TRIP_TRAVEL_TIME_MIN=10, _MAIN_PURPOSE=1, _HOME_DESTINATION=0,
                    TOUR_CLOSING_TRIP_FLAG=0)
    return pd.DataFrame([{**defaults, **row, "SOURCE_ROW_ID": str(i)} for i, row in enumerate(rows)])


def result(*rows: dict) -> pd.Series:
    return tour_features.longest_household_driver_purpose(trips(*rows)).iloc[0]


class LongestPurposeTests(unittest.TestCase):
    def test_closed_single_return_trip(self):
        row = result(dict(_HOME_DESTINATION=1, TOUR_CLOSING_TRIP_FLAG=1))
        self.assertEqual(row.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2, 1)
        self.assertEqual(row.TOUR_LONGEST_RETURN_HOME_FLAG, 1)
        self.assertEqual(row.TOUR_LONGEST_IS_CLOSING_TRIP_FLAG, 1)

    def test_open_observed_trip_is_not_official_main(self):
        row = result(dict(_MAIN_PURPOSE=7, _HOME_DESTINATION=0,
                          TOUR_CLOSING_TRIP_FLAG=0))
        self.assertEqual(row.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2, 7)
        self.assertEqual(row.TOUR_LONGEST_IS_CLOSING_TRIP_FLAG, 0)

    def test_multistop_mixed_and_nonclosing(self):
        row = result(dict(TRIP_TRAVEL_TIME_MIN=20, _MAIN_PURPOSE=6),
                     dict(TRIP_TRAVEL_TIME_MIN=60, pkw_fmf="2", _MAIN_PURPOSE=7),
                     dict(TRIP_TRAVEL_TIME_MIN=15, _HOME_DESTINATION=1,
                          TOUR_CLOSING_TRIP_FLAG=1))
        self.assertEqual(row.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2, 6)
        self.assertEqual(row.TOUR_LONGEST_NOT_CLOSING_TRIP_FLAG, 1)

    def test_no_household_driver_and_invalid_duration(self):
        for data in [dict(pkw_fmf="2"), dict(W_WAUTO="3")]:
            self.assertEqual(result(data).TOUR_NO_ELIGIBLE_HH_DRIVER_TRIP_FLAG, 1)
        self.assertEqual(result(dict(TRIP_TRAVEL_TIME_MIN=None)).TOUR_NO_VALID_DRIVER_DURATION_FLAG, 1)

    def test_missing_longest_purpose_and_ties(self):
        row = result(dict(TRIP_TRAVEL_TIME_MIN=30, _MAIN_PURPOSE=99),
                     dict(TRIP_TRAVEL_TIME_MIN=10, _MAIN_PURPOSE=1))
        self.assertTrue(pd.isna(row.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2))
        self.assertEqual(row.TOUR_LONGEST_PURPOSE_MISSING_FLAG, 1)
        common = result(dict(_MAIN_PURPOSE=4), dict(_MAIN_PURPOSE=4))
        self.assertEqual(common.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2, 4)
        self.assertEqual(common.TOUR_LONGEST_TIE_COMMON_PURPOSE_FLAG, 1)
        for second in [5, 99]:
            unresolved = result(dict(_MAIN_PURPOSE=4), dict(_MAIN_PURPOSE=second))
            self.assertTrue(pd.isna(unresolved.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2))
            self.assertEqual(unresolved.TOUR_LONGEST_TIE_UNRESOLVED_FLAG, 1)

    def test_tied_maximum_includes_closing_trip(self):
        row = result(dict(_MAIN_PURPOSE=6),
                     dict(_MAIN_PURPOSE=6, _HOME_DESTINATION=1,
                          TOUR_CLOSING_TRIP_FLAG=1))
        self.assertEqual(row.TOUR_LONGEST_DRIVER_TRIP_HWZWECK2, 6)
        self.assertEqual(row.TOUR_LONGEST_DURATION_TIE_FLAG, 1)
        self.assertEqual(row.TOUR_LONGEST_IS_CLOSING_TRIP_FLAG, 1)


class VehicleContinuityTests(unittest.TestCase):
    def summary(self, *rows):
        frame = trips(*rows)
        frame["CHAIN_SEQUENCE_NUM"] = range(1, len(frame) + 1)
        return tour_features.vehicle_continuity(frame)

    def test_switch_and_uncertain_identity_coexist(self):
        summary, detail = self.summary({}, {"W_WAUTO": "2"}, {"W_WAUTO": "9"})
        row = summary.iloc[0]
        self.assertEqual(row.TOUR_QA_N_DISTINCT_HH_CARS, 2)
        self.assertEqual(row.TOUR_QA_OBSERVED_CAR_SWITCH_FLAG, 1)
        self.assertEqual(row.TOUR_QA_DRIVER_VEHICLE_UNCERTAIN_FLAG, 1)
        self.assertEqual(row.TOUR_QA_SWITCH_WITH_UNCERTAINTY_FLAG, 1)
        self.assertEqual(detail.QA_IDENTIFIED_HH_DRIVER_FLAG.tolist(), [1, 1, 0])

    def test_one_driver_trip_is_not_complete_round_trip(self):
        summary, _ = self.summary({}, {"pkw_fmf": "2", "TOUR_CLOSING_TRIP_FLAG": 1})
        self.assertEqual(summary.iloc[0].TOUR_QA_N_IDENTIFIED_HH_DRIVER_TRIPS, 1)
        self.assertEqual(summary.iloc[0].TOUR_QA_COMPLETE_MULTITRIP_SINGLE_CAR_FLAG, 0)
        self.assertEqual(summary.iloc[0].TOUR_QA_CLOSING_FIRST_CAR_FLAG, 0)

    def test_same_car_closing_return_and_mixed_mode(self):
        summary, _ = self.summary({}, {"pkw_fmf": "2"},
                                  {"W_WAUTO": "1", "TOUR_CLOSING_TRIP_FLAG": 1})
        self.assertEqual(summary.iloc[0].TOUR_QA_OBSERVED_CAR_SWITCH_FLAG, 0)
        self.assertEqual(summary.iloc[0].TOUR_QA_COMPLETE_MULTITRIP_SINGLE_CAR_FLAG, 1)
        self.assertEqual(summary.iloc[0].TOUR_QA_CLOSING_FIRST_CAR_FLAG, 1)


class PurposeCodeTests(unittest.TestCase):
    def test_exact_documented_integer(self):
        values = pd.Series(["1", "6", "7", "99", "", None])
        parsed = tour_features.documented_purpose(values, set(range(1, 8)) | {99}, "hwzweck2")
        self.assertEqual(parsed.iloc[:4].tolist(), [1, 6, 7, 99])
        self.assertTrue(parsed.iloc[4:].isna().all())
        with self.assertRaises(ValueError):
            tour_features.documented_purpose(pd.Series(["6.4"]), set(range(1, 8)) | {99}, "hwzweck2")
        with self.assertRaises(ValueError):
            tour_features.documented_purpose(pd.Series(["8"]), set(range(1, 8)) | {99}, "hwzweck2")

    def test_future_detailed_first_purpose_code_fails_current_registry(self):
        script = (vehicle_choice.ROOT / "scripts/03_model_input/"
                  "05_build_single_driver_mnl_design_matrix.py")
        spec = spec_from_file_location("purpose_matrix_test_registry", script)
        module = module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        first = next(s for s in module.TRANSFORMATION_REGISTRY if s.key == "first_purpose")
        with self.assertRaises(ValueError):
            module.apply_categorical(pd.DataFrame({"CHOICE_ID": ["x"],
                                                   "TOUR_FIRST_PURPOSE": ["11"]}), first)


class ComparisonTests(unittest.TestCase):
    def test_open_personal_tour_has_no_official_main(self):
        path = vehicle_choice.ROOT / "data_processed/reconstruction/phase4/home_based_tour_features.csv"
        tours = pd.read_csv(path, usecols=["TOUR_CLOSED", "TOUR_MAIN_PURPOSE"],
                            dtype=str, keep_default_na=False)
        self.assertTrue(tours.loc[tours.TOUR_CLOSED.eq("0"), "TOUR_MAIN_PURPOSE"].eq("").all())

    def test_two_personal_tours_keep_separate_purposes(self):
        tours = pd.DataFrame({"TOUR_ID": ["t1", "t2"], "PERSON_KEY": ["p", "p"],
                              "TOUR_CLOSED": [1, 1], "TOUR_MAIN_PURPOSE": [2, 5]})
        membership = pd.DataFrame({"TOUR_ID": ["t1", "t2"], "PERSON_KEY": ["p", "p"],
                                   "CANONICAL_PURPOSE": [7, 2]})
        rows = purpose_qa.main_scope_rows(tours, membership)
        counts = {(r["population"], r["metric"]): r["count"] for r in rows}
        population = "closed tours of multi-tour persons"
        self.assertEqual(counts[(population, "main absent despite own-tour activity")], 2)
        self.assertEqual(counts[(population, "main present in another personal tour")], 1)

    def test_official_semantic_harmonization(self):
        raw = pd.Series([6, 7, 11, 12, 13, 14, 15, 16, 8, 9, 10])
        actual = purpose_qa.harmonize_first_purpose(raw).tolist()
        self.assertEqual(actual[:8], [7, 6, 3, 3, 7, 6, 6, 6])
        self.assertTrue(all(pd.isna(value) for value in actual[8:]))

    def test_add_columns_and_reject_existing_change(self):
        old = pd.DataFrame({"CHOICE_ID": ["a"], "SOURCE_ROW_ID": ["10"],
                            "HOUSEHOLD_ACCOMPANIED": ["1"], "CHOICE": ["2"]})
        new = old.copy()
        new["TOUR_MAIN_PURPOSE"] = pd.Series([1], dtype="Int64")
        new["TOUR_LONGEST_DRIVER_TRIP_HWZWECK2"] = pd.Series([pd.NA], dtype="Int64")
        self.assertEqual(vehicle_choice.compare_against_previous(old, new)["existing_column_equality"], "PASS")
        new.loc[0, "HOUSEHOLD_ACCOMPANIED"] = "0"
        with self.assertRaises(AssertionError):
            vehicle_choice.compare_against_previous(old, new)

    def test_load_inputs_never_reads_prior_output(self):
        paths = replace(vehicle_choice.resolve_paths(), model_input=Path("nonexistent.csv"))
        with patch.object(vehicle_choice, "read_csv_strings", return_value=pd.DataFrame()) as reader:
            self.assertNotIn("previous_model", vehicle_choice.load_inputs(paths))
        self.assertNotIn(paths.model_input, [call.args[0] for call in reader.call_args_list])

    def test_saved_design_passthrough_and_no_row_filter(self):
        root = vehicle_choice.ROOT / "data_processed" / "model_input"
        source = pd.read_csv(root / "allocation_candidate_model_input.csv", dtype=str, keep_default_na=False)
        final = pd.read_csv(root / "single_driver_mnl_design_matrix.csv", dtype=str, keep_default_na=False)
        retained = source.loc[source.SINGLE_ACTIVE_DRIVER_HH.eq("1")].reset_index(drop=True)
        purpose = [purpose_qa.FIRST, purpose_qa.MAIN, purpose_qa.LONGEST]
        self.assertEqual(len(final), len(retained))
        self.assertEqual(final.CHOICE_ID.tolist(), retained.CHOICE_ID.tolist())
        self.assertEqual(final.columns[189:192].tolist(), purpose)
        self.assertEqual(len(final.columns), 208)
        self.assertTrue(final[purpose].equals(retained[purpose]))
        for raw, prefix in [(purpose[1], "MAIN_PURPOSE"), (purpose[2], "LONGEST_PURPOSE")]:
            names = [f"{prefix}_{label}" for label in
                     ["WORK", "BUSINESS", "EDUCATION", "SHOPPING", "ERRAND", "LEISURE", "ESCORT", "MISSING"]]
            block = final[names].astype(int)
            self.assertTrue(block.sum(axis=1).eq(1).all())
            codes = pd.to_numeric(retained[raw], errors="coerce")
            for code, name in enumerate(names[:-1], 1):
                self.assertTrue(block[name].eq(codes.eq(code).astype(int)).all())
            self.assertTrue(block[names[-1]].eq(retained[raw].eq("").astype(int)).all())


if __name__ == "__main__":
    unittest.main()
