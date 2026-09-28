"""Read-only baseline versus extension description, with full sensitivity context.

Shares use valid-category weight sums; moments use valid-value weight sums.
Weighted median is the first sorted value reaching half the valid weight.
SMD = (extension mean - baseline mean) / sqrt((weighted population variances)/2).
No inference, row filtering, weight normalization, or model changes occur.
Numeric purpose codes are retained: the broad-purpose indicator labels are not
safe labels for the detailed TOUR_FIRST_PURPOSE codes.
"""
from pathlib import Path
import hashlib
import sys
import warnings
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            sys.path.insert(0, str(candidate))
            break

from src.thesis_pipeline.paths import MODEL_INPUT_DIR, RECONSTRUCTION_DIR
from src.thesis_pipeline.sensitivity.decision_timing import (
    check_fingerprints,
    fingerprints,
    ids,
    overlap_equal,
    read,
    reject,
)

MODEL_DIR = MODEL_INPUT_DIR
RECON = RECONSTRUCTION_DIR
OUT = MODEL_DIR / "sensitivity/comparison"
FULL = OUT / "decision_timing_sample_comparison.csv"
KEY = OUT / "decision_timing_sample_comparison_key_results.csv"
BANDS = ["before 06:00", "06:00–08:59", "09:00–11:59", "12:00–14:59", "15:00–17:59", "18:00–20:59", "21:00 or later"]


def weighted_stats(values, weights):
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    w = np.asarray(weights,dtype=float)
    valid = np.isfinite(x)
    x,w = x[valid],w[valid]
    if not len(x): return dict(n=0, mean=np.nan, median=np.nan, variance=np.nan, unweighted_mean=np.nan, unweighted_median=np.nan)
    order = np.argsort(x,kind="stable")
    cumulative = np.cumsum(w[order])
    mean = np.sum(w*x)/np.sum(w)
    return dict(n=len(x),mean=mean,median=x[order][np.searchsorted(cumulative,cumulative[-1]/2,side="left")],
                variance=np.sum(w*(x-mean)**2)/np.sum(w),unweighted_mean=np.mean(x),unweighted_median=np.median(x))


def main():
    before = fingerprints()
    # Include all accepted sensitivity model/Phase-4 datasets in the read-only guard.
    protected = [*MODEL_DIR.joinpath("sensitivity").glob("*.csv"),
        *RECON.joinpath("phase4_sensitivity").glob("mnl*.csv")]
    protected_hashes = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    baseline = read(MODEL_DIR / "mnl_vehicle_choice_model_input.csv")
    sensitivity = read(MODEL_DIR / "sensitivity/mnl_vehicle_choice_model_input_decision_timing_sensitivity.csv")
    extension = overlap_equal(baseline,sensitivity)
    if extension.empty: raise AssertionError("Baseline must be a proper subset.")
    samples = [baseline.copy(),extension.copy(),sensitivity.copy()]
    for name, frame, expected in zip(["baseline","extension","sensitivity"],samples,[30816,7327,38143]):
        ids(frame)
        if len(frame)!=expected: warnings.warn(f"{name} count {len(frame)} differs from accepted {expected}; verify intentional upstream change.")
        w = pd.to_numeric(frame.W_GEW,errors="coerce")
        reject(frame,w.isna() | ~np.isfinite(w) | w.le(0),"Weights must be finite and positive.")
        frame["_weight"] = w
    phase4 = read(RECON / "phase4_sensitivity/mnl_vehicle_choice_wide_tour_features_decision_timing_sensitivity.csv",
                  ["CHOICE_ID","SOURCE_ROW_ID","TOUR_MATCH_FOUND","TOUR_CLOSED","TOUR_OPEN_END_FLAG"])
    ids(phase4)
    if set(phase4.CHOICE_ID)!=set(sensitivity.CHOICE_ID): raise AssertionError("Phase 4 universe differs.")
    state = read(RECON / "phase3_sensitivity/vehicle_choice_occasions_decision_timing_sensitivity.csv",
                 ["CHOICE_ID","SOURCE_ROW_ID","PREDEPARTURE_STATE_PATTERN","SENSITIVITY_EXTENSION_FLAG"])
    ids(state)
    for i,frame in enumerate(samples):
        for metadata in [phase4,state]:
            frame = frame.merge(metadata,on=["CHOICE_ID","SOURCE_ROW_ID"],how="left",validate="one_to_one",indicator=True)
            reject(frame,frame._merge.ne("both"),"Unmatched structural metadata.")
            frame = frame.drop(columns="_merge")
        reject(frame,frame.TOUR_MATCH_FOUND.ne("1"),"Tour match must equal 1.")
        expected_extension = ~frame.CHOICE_ID.isin(baseline.CHOICE_ID)
        reject(frame,pd.to_numeric(frame.SENSITIVITY_EXTENSION_FLAG).ne(expected_extension.astype(int)),"Extension metadata contradiction.")
        reject(frame,expected_extension & ~frame.PREDEPARTURE_STATE_PATTERN.isin(["HOME_AWAY","AWAY_HOME"]),"Invalid extension state.")
        reject(frame,~expected_extension & frame.PREDEPARTURE_STATE_PATTERN.ne("HOME_HOME"),"Invalid baseline state.")
        frame["departure_band"] = pd.cut(pd.to_numeric(frame.START_MIN,errors="coerce"),
            [-np.inf,360,540,720,900,1080,1260,np.inf],right=False,labels=BANDS).astype("string").fillna("")
        for feature in ["POWERTRAIN","SEGMENT","HOLDER","STATUS"]:
            a,b=frame[f"{feature}_1"],frame[f"{feature}_2"]
            frame[f"{feature.lower()}_pair"] = np.where(a.eq("")|b.eq(""),"",np.where(a.eq(b),"same","different"))
        frame["absolute_vehicle_age_difference"] = (pd.to_numeric(frame.VEHICLE_AGE_1,errors="coerce")-pd.to_numeric(frame.VEHICLE_AGE_2,errors="coerce")).abs()
        samples[i]=frame
    rows=[]
    def add(section,variable,statistic,values,ns=None,category="",weighting="UNWEIGHTED",notes="",pp=False):
        a,b,c=values
        rows.append(dict(section=section,variable=variable,category=category,statistic=statistic,
            baseline_value=a,extension_value=b,sensitivity_value=c,difference=(b-a)*(100 if pp else 1),
            baseline_valid_n=ns[0] if ns else len(samples[0]),extension_valid_n=ns[1] if ns else len(samples[1]),
            sensitivity_valid_n=ns[2] if ns else len(samples[2]),weighting=weighting,
            notes=notes+(" Difference is extension minus baseline in percentage points; values are proportions." if pp else " Difference is extension minus baseline.")))
    for variable,key in [("occasions",None),("households","H_ID"),("persons","HP_ID")]:
        add("STRUCTURE",variable,"count",[len(f) if key is None else f[key].nunique() for f in samples])
    existing = samples[1].H_ID.isin(baseline.H_ID)
    for variable,value in [("new households",len(set(extension.H_ID)-set(baseline.H_ID))),
        ("existing baseline households with extension",len(set(extension.H_ID)&set(baseline.H_ID))),
        ("extension occasions from existing households",int(existing.sum())),
        ("extension occasion share from existing households",float(existing.mean()))]:
        add("STRUCTURE",variable,"share" if "share" in variable else "count",[np.nan,value,np.nan])
    add("VALIDATION","tour matches","count",[len(f) for f in samples],notes="All matched / all occasions; match rate 100%.")
    for col in ["TOUR_CLOSED","TOUR_OPEN_END_FLAG"]:
        add("STRUCTURE",col,"count",[int(f[col].eq("1").sum()) for f in samples])
    def continuous(section,col):
        stats=[weighted_stats(f[col],f._weight) for f in samples]
        ns=[s["n"] for s in stats]
        for stat in ["mean","median","unweighted_mean","unweighted_median"]:
            weighted=not stat.startswith("unweighted")
            add(section,col,"weighted_"+stat if weighted else stat,[s[stat] for s in stats],ns,
                weighting="W_GEW-WEIGHTED" if weighted else "UNWEIGHTED",
                notes="Valid-value denominators; missing values retained in sample. Weighted median: first value reaching half cumulative weight.")
        pooled=np.sqrt((stats[0]["variance"]+stats[1]["variance"])/2)
        smd=(stats[1]["mean"]-stats[0]["mean"])/pooled if pooled>0 else np.nan
        add(section,col,"standardized_mean_difference",[np.nan,smd,np.nan],ns,weighting="W_GEW-WEIGHTED",
            notes="Extension value is (extension mean-baseline mean)/sqrt((baseline variance+extension variance)/2); weighted population variances. Descriptive only.")
    def categorical(section,col):
        categories=sorted(set().union(*(set(f[col].dropna())-{ ""} for f in samples)),key=str)
        ns=[int(f[col].notna().sum()-f[col].eq("").sum()) for f in samples]
        for category in categories:
            counts=[int(f[col].eq(category).sum()) for f in samples]
            shares=[]
            for f in samples:
                valid=f[col].notna() & f[col].ne("")
                denominator=f.loc[valid,"_weight"].sum()
                shares.append(f.loc[f[col].eq(category),"_weight"].sum()/denominator if denominator>0 else np.nan)
            add(section,col,"count",counts,ns,category=category)
            add(section,col,"weighted_share",shares,ns,category=category,weighting="W_GEW-WEIGHTED",pp=True,
                notes="Denominator: sum of raw W_GEW with valid category; household/person characteristics are occasion-weighted.")
    categorical("CHOICE","CHOICE")
    continuous("TEMPORAL","START_MIN")
    for col in ["departure_band","DAY_TYPE","saison"]: categorical("TEMPORAL",col)
    for col in ["TOUR_DISTANCE_KM","TOUR_TRAVEL_TIME_MIN","TOUR_N_NONHOME_STOPS","TOUR_N_DISTINCT_PURPOSES"]: continuous("TOUR_DEMAND",col)
    categorical("TOUR_DEMAND","TOUR_FIRST_PURPOSE")
    for col in baseline.columns:
        if col.startswith("TOUR_HAS_"): categorical("TOUR_DEMAND",col)
    for col in ["household_type","hhgr_gr","oek_status","RegioStaR4","XMStadt","quali_opnv","min_bab","min_ozmz"]: categorical("HOUSEHOLD",col)
    for col in ["HP_SEX","alter_gr5","erwerb","P_VPED","P_VRAD","carsharing","mobein"]: categorical("PERSON",col)
    for col in ["powertrain_pair","segment_pair","holder_pair","status_pair"]: categorical("VEHICLE_PAIR",col)
    continuous("VEHICLE_PAIR","absolute_vehicle_age_difference")
    categorical("STATE","PREDEPARTURE_STATE_PATTERN")
    result=pd.DataFrame(rows)
    # Compact selection: pre-specified central results plus largest absolute
    # composition differences within explicitly selected blocks (descriptive).
    selected=[]
    def pick(variable,statistic,category=None,largest=False,n=1):
        part=result.loc[result.variable.eq(variable)&result.statistic.eq(statistic)]
        if category is not None: part=part.loc[part.category.eq(category)]
        if largest: part=part.loc[part.difference.abs().sort_values(ascending=False).index]
        selected.extend(part.head(n).index)
    for col in ["occasions","households","persons"]: pick(col,"count")
    pick("extension occasion share from existing households","share")
    pick("CHOICE","weighted_share",n=2)
    pick("START_MIN","weighted_mean"); pick("START_MIN","weighted_median")
    pick("departure_band","weighted_share",largest=True)
    for col in ["TOUR_DISTANCE_KM","TOUR_TRAVEL_TIME_MIN","TOUR_N_NONHOME_STOPS"]: pick(col,"weighted_mean")
    pick("TOUR_FIRST_PURPOSE","weighted_share",largest=True,n=2)
    for col in ["household_type","alter_gr5","erwerb"]: pick(col,"weighted_share",largest=True)
    for col in ["segment_pair","powertrain_pair"]: pick(col,"weighted_share",category="different")
    pick("absolute_vehicle_age_difference","weighted_mean")
    for cat in ["HOME_AWAY","AWAY_HOME"]: pick("PREDEPARTURE_STATE_PATTERN","weighted_share",category=cat)
    key=result.loc[list(dict.fromkeys(selected))].copy()
    if not 15<=len(key)<=25: raise AssertionError("Key table must stay compact.")
    if [len(f) for f in samples]!=[len(baseline),len(extension),len(sensitivity)]: raise AssertionError("Rows dropped.")
    # Verify share totals and reproduce former QA directly from model values.
    shares=result.loc[result.statistic.eq("weighted_share")]
    for col in ["baseline_value","extension_value","sensitivity_value"]:
        if not np.allclose(shares.groupby("variable")[col].sum(),1): raise AssertionError("Weighted shares do not sum to one.")
    check_fingerprints(before)
    for p,digest in protected_hashes.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise AssertionError(f"Accepted input modified: {p}")
    OUT.mkdir(parents=True,exist_ok=True)
    result.to_csv(FULL,index=False); key.to_csv(KEY,index=False)
    if len(read(FULL))!=len(result) or len(read(KEY))!=len(key): raise AssertionError("Comparison serialization failed.")
    check_fingerprints(before)
    for p,digest in protected_hashes.items():
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise AssertionError(f"Accepted input modified: {p}")
    print("DECISION-TIMING SAMPLE COMPARISON")
    print(key[["variable","category","statistic","baseline_value","extension_value","difference"]].to_string(index=False))
    print(f"Full sensitivity: {len(sensitivity):,} occasions; {sensitivity.H_ID.nunique():,} households; {sensitivity.HP_ID.nunique():,} persons")
    print(f"Tour matching: {len(sensitivity):,} / {len(sensitivity):,}")
    print("OUTPUTS:"); print(FULL); print(KEY)
    print("Accepted baseline and sensitivity reconstruction/model-input files were not modified.")
    return result

if __name__ == "__main__":
    main()
