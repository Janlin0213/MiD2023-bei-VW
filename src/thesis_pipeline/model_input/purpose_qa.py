"""Official MiD 2023 purpose harmonization and descriptive comparison QA.

Source: MiD2023_Codepläne_B1_Standard_v1.1.xlsx, Wege, W_ZWECK/zweck and
hwzweck2. Stored raw variables are never recoded by this module.
"""
from __future__ import annotations

import pandas as pd
from src.thesis_pipeline.model_input.tour_features import VEHICLE_QA_COLUMNS

FIRST = "TOUR_FIRST_PURPOSE"
MAIN = "TOUR_MAIN_PURPOSE"
LONGEST = "TOUR_LONGEST_DRIVER_TRIP_HWZWECK2"
PURPOSES = (FIRST, MAIN, LONGEST)
FIRST_TO_HWZWECK2 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 6,
                     11: 3, 12: 3, 13: 7, 14: 6, 15: 6, 16: 6}
LABELS = {1: "Work", 2: "Business", 3: "Education", 4: "Shopping",
          5: "Errand", 6: "Leisure", 7: "Escort / accompaniment"}
FLAGS = ["TOUR_NO_ELIGIBLE_HH_DRIVER_TRIP_FLAG", "TOUR_NO_VALID_DRIVER_DURATION_FLAG",
         "TOUR_LONGEST_PURPOSE_MISSING_FLAG", "TOUR_LONGEST_DURATION_TIE_FLAG",
         "TOUR_LONGEST_TIE_COMMON_PURPOSE_FLAG", "TOUR_LONGEST_TIE_UNRESOLVED_FLAG",
         "TOUR_LONGEST_RETURN_HOME_FLAG", "TOUR_LONGEST_IS_CLOSING_TRIP_FLAG",
         "TOUR_LONGEST_NOT_CLOSING_TRIP_FLAG", *VEHICLE_QA_COLUMNS[2:]]


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip(), errors="coerce")


def harmonize_first_purpose(series: pd.Series) -> pd.Series:
    """Map comparable first-purpose semantics to official hwzweck2 categories.

    Codes 8 (home), 9 (return), and 10 (other) remain non-comparable.
    """
    return numeric(series).map(FIRST_TO_HWZWECK2).astype("Int64")


def attach_tour_diagnostics(frame: pd.DataFrame, tours: pd.DataFrame) -> pd.DataFrame:
    if frame["CHOICE_ID"].duplicated().any() or frame["SOURCE_ROW_ID"].duplicated().any():
        raise AssertionError("Choice keys must be unique for purpose QA.")
    columns = ["TOUR_START_SOURCE_ROW_ID", "TOUR_ID", "TOUR_CLOSED", *PURPOSES, *FLAGS,
               *VEHICLE_QA_COLUMNS[:2]]
    merged = frame[["CHOICE_ID", "SOURCE_ROW_ID", "CHOICE"]].merge(
        tours[columns], left_on="SOURCE_ROW_ID", right_on="TOUR_START_SOURCE_ROW_ID",
        how="left", validate="one_to_one", sort=False,
    )
    if len(merged) != len(frame) or merged["TOUR_ID"].isna().any():
        raise AssertionError("Choice occasion did not match exactly one reconstructed tour.")
    for column in PURPOSES:
        allowed = range(1, 17) if column == FIRST else range(1, 8)
        left = numeric(frame[column]).where(lambda s: s.isin(allowed)).astype("Int64").reset_index(drop=True)
        right = numeric(merged[column]).where(lambda s: s.isin(allowed)).astype("Int64").reset_index(drop=True)
        if not left.equals(right):
            raise AssertionError(f"Purpose field {column} differs from tour source.")
    return merged


def vehicle_choice_rows(frame: pd.DataFrame, detail: pd.DataFrame, name: str) -> list[dict]:
    """Focal-car QA; unknown trip identities never count as confirmed matches."""
    if frame.TOUR_ID.duplicated().any():
        raise AssertionError("Vehicle QA requires one focal occasion per reconstructed tour.")
    chosen = pd.to_numeric(frame.CHOICE, errors="coerce")
    if not chosen.isin([1, 2]).all():
        raise AssertionError("Focal choice must identify one of the recorded two cars.")
    work = detail.merge(frame[["TOUR_ID", "CHOICE"]], on="TOUR_ID", validate="many_to_one")
    driver = work.QA_IDENTIFIED_HH_DRIVER_FLAG.eq(1)
    work["match"] = work.QA_CAR_ID.eq(work.CHOICE.astype("string").str.replace(r"\.0$", "", regex=True)).fillna(False)
    matched = work.loc[driver].groupby("TOUR_ID").match.all()
    other = work.loc[driver].groupby("TOUR_ID").match.apply(lambda values: bool((~values).any()))
    uncertain = work.groupby("TOUR_ID").QA_UNCERTAIN_DRIVER_VEHICLE_FLAG.max().eq(1)
    closing = work.loc[driver & numeric(work.TOUR_CLOSING_TRIP_FLAG).eq(1)].set_index("TOUR_ID").match
    ids = frame.TOUR_ID
    all_match = ids.map(matched).fillna(False)
    other_car = ids.map(other).fillna(False)
    unknown = ids.map(uncertain).fillna(False)
    chosen_return = ids.map(closing).fillna(False)
    closed = numeric(frame.TOUR_CLOSED).eq(1)
    n = len(frame)
    metrics = {
        "all identified trips match focal car": int(all_match.sum()),
        "complete focal-car match": int((all_match & ~unknown).sum()),
        "identified other household car": int(other_car.sum()),
        "incomplete driver/vehicle identity": int(unknown.sum()),
        "switch together with incomplete identity": int((other_car & unknown).sum()),
        "closed personal tour": int(closed.sum()),
        "closing trip directly drives chosen car": int((closed & chosen_return).sum()),
        "closed tour without direct chosen-car return": int((closed & ~chosen_return).sum()),
    }
    return [dict(section="vehicle_choice", population=name, metric=k, count=v,
                 denominator=int(closed.sum()) if "closing trip" in k or k.startswith("closed tour without") else n,
                 share=v / (int(closed.sum()) if "closing trip" in k or k.startswith("closed tour without") else n))
            for k, v in metrics.items()]


def main_scope_rows(tours: pd.DataFrame, membership: pd.DataFrame) -> list[dict]:
    """Descriptive compatibility, not a recreation of MiD's priority algorithm."""
    activities = membership[["TOUR_ID", "PERSON_KEY", "CANONICAL_PURPOSE"]].copy()
    activities["code"] = harmonize_first_purpose(activities.CANONICAL_PURPOSE)
    activities = activities.loc[activities.code.notna(), ["TOUR_ID", "PERSON_KEY", "code"]].drop_duplicates()
    closed = tours.loc[numeric(tours.TOUR_CLOSED).eq(1),
                       ["TOUR_ID", "PERSON_KEY", MAIN]].copy()
    closed["main_code"] = numeric(closed[MAIN])
    matches = activities.merge(closed[["TOUR_ID", "main_code"]], on="TOUR_ID")
    matches["same"] = matches.code.eq(matches.main_code)
    has_activity = activities.groupby("TOUR_ID").size()
    own_match = matches.groupby("TOUR_ID").same.any()
    closed["has_activity"] = closed.TOUR_ID.map(has_activity).fillna(0).gt(0)
    closed["own_match"] = closed.TOUR_ID.map(own_match).fillna(False)
    person_tour_counts = tours.groupby("PERSON_KEY").TOUR_ID.nunique()
    closed["multiple"] = closed.PERSON_KEY.map(person_tour_counts).gt(1)
    exceptions = closed.loc[closed.has_activity & ~closed.own_match]
    other = activities.merge(exceptions[["TOUR_ID", "PERSON_KEY", "main_code"]].rename(
        columns={"TOUR_ID": "exception_tour"}), on="PERSON_KEY")
    other_tour_ids = set(other.loc[other.TOUR_ID.ne(other.exception_tour) &
                                   other.code.eq(other.main_code), "exception_tour"])
    rows = []
    for label, subset in [("closed tours", closed),
                          ("closed tours of multi-tour persons", closed.loc[closed.multiple]),
                          ("closed tours of single-tour persons", closed.loc[~closed.multiple])]:
        denominator = len(subset)
        metrics = {
            "closed tours with substantive own-tour activity": int(subset.has_activity.sum()),
            "main represented in own-tour activities": int(subset.own_match.sum()),
            "main absent despite own-tour activity": int((subset.has_activity & ~subset.own_match).sum()),
            "main present in another personal tour": int(subset.TOUR_ID.isin(other_tour_ids).sum()),
            "no substantive own-tour activity": int((~subset.has_activity).sum()),
        }
        for metric, count in metrics.items():
            rows.append(dict(section="main_scope", population=label, metric=metric, count=count,
                             denominator=denominator, share=count / denominator if denominator else None))
    return rows


def summarize_population(frame: pd.DataFrame, name: str) -> list[dict]:
    n = len(frame)
    rows: list[dict] = []
    def add(section: str, metric: str, count: int, denominator: int = n, **extra: object) -> None:
        rows.append(dict(section=section, population=name, metric=metric, count=count,
                         denominator=denominator, share=count / denominator if denominator else None,
                         **extra))
    add("coverage", "total observations", n)
    add("coverage", "closed tours", int(numeric(frame["TOUR_CLOSED"]).eq(1).sum()))
    for column in PURPOSES:
        valid = numeric(frame[column]).isin(range(1, 17) if column == FIRST else range(1, 8))
        add("coverage", f"{column} observed", int(valid.sum()))
        add("coverage", f"{column} missing", int((~valid).sum()))
    for flag in FLAGS:
        add("diagnostic", flag, int(numeric(frame[flag]).eq(1).sum()))
    first_raw = numeric(frame[FIRST])
    first = harmonize_first_purpose(frame[FIRST])
    main = numeric(frame[MAIN]).where(lambda s: s.isin(range(1, 8)))
    longest = numeric(frame[LONGEST]).where(lambda s: s.isin(range(1, 8)))
    for code in (8, 9, 10):
        add("harmonization", f"first-purpose non-comparable code {code}", int(first_raw.eq(code).sum()))
    pairs = [(LONGEST, MAIN, longest, main), (LONGEST, FIRST, longest, first),
             (MAIN, FIRST, main, first)]
    for left_name, right_name, left, right in pairs:
        comparable = left.notna() & right.notna()
        agree = comparable & left.eq(right)
        pair = f"{left_name} vs harmonized {right_name}" if right_name == FIRST else f"{left_name} vs {right_name}"
        add("pairwise", pair + " comparable", int(comparable.sum()))
        add("pairwise", pair + " agree", int(agree.sum()), int(comparable.sum()))
        add("pairwise", pair + " disagree", int((comparable & ~agree).sum()), int(comparable.sum()))
    all_valid = first.notna() & main.notna() & longest.notna()
    add("three_way", "all comparable", int(all_valid.sum()))
    patterns = {
        "all three agree": first.eq(main) & main.eq(longest),
        "first=main, longest differs": first.eq(main) & main.ne(longest),
        "main=longest, first differs": main.eq(longest) & first.ne(main),
        "first=longest, main differs": first.eq(longest) & main.ne(first),
        "all three differ": first.ne(main) & main.ne(longest) & first.ne(longest),
    }
    for label, mask in patterns.items():
        add("three_way", label, int((all_valid & mask).sum()), int(all_valid.sum()))
    return rows


def disagreement_crosstab(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    first = harmonize_first_purpose(frame[FIRST])
    main = numeric(frame[MAIN])
    longest = numeric(frame[LONGEST])
    tables = []
    for pair, left, right in [("longest_vs_main", longest, main),
                              ("longest_vs_first", longest, first),
                              ("main_vs_first", main, first)]:
        comparable = left.isin(LABELS) & right.isin(LABELS)
        tmp = pd.DataFrame({"left": left[comparable], "right": right[comparable]})
        tab = tmp.value_counts().rename("count").reset_index()
        tab["population"] = name
        tab["pair"] = pair
        tab["left_label"] = tab["left"].map(LABELS)
        tab["right_label"] = tab["right"].map(LABELS)
        tab["agreement"] = tab["left"].eq(tab["right"])
        tables.append(tab)
    return pd.concat(tables, ignore_index=True)
