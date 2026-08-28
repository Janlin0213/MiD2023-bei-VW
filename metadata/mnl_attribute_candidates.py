"""Canonical candidates for the baseline household vehicle-choice MNL input.

This registry deliberately contains one preferred representation per modelling
construct. It is not the selected-raw extraction universe. Variables kept only
for reconstruction, QA, realised-choice leakage, closed-tour sensitivity, or
later route/OD extensions are intentionally absent.

The names below are final model-input names. Alternative-specific vehicle
features are listed without ``_1``/``_2``; the downstream builder pivots them
by the authoritative ``H_ID + A_ID`` key.
"""


MNL_ATTRIBUTE_CANDIDATES = {
    "tour_and_departure_context": {
        "time": [
            # START_MIN is the accepted continuous departure time. DAY_TYPE
            # replaces ST_WOTAG + feiertag; saison is the only season encoding.
            "START_MIN",
            "DAY_TYPE",
            "saison",
        ],
        "tour_demand": [
            # Tour totals supersede wegkm_imp/wegmin_imp2 and every LEG_* field.
            "TOUR_DISTANCE_KM",
            "TOUR_TRAVEL_TIME_MIN",
        ],
        "tour_complexity": [
            # Counts replace TOUR_N_TRIPS and deterministic stop/purpose flags.
            "TOUR_N_NONHOME_STOPS",
            "TOUR_N_DISTINCT_PURPOSES",
        ],
        "purpose": [
            # TOUR_FIRST_PURPOSE replaces W_ZWECK/zweck/hwzweck*/LEG_PURPOSE.
            # OTHER is retained because Phase 6 implements it for purpose 10.
            "TOUR_FIRST_PURPOSE",
            "TOUR_HAS_WORK",
            "TOUR_HAS_BUSINESS",
            "TOUR_HAS_EDUCATION",
            "TOUR_HAS_SHOPPING",
            "TOUR_HAS_ERRAND",
            "TOUR_HAS_LEISURE",
            "TOUR_HAS_ESCORT",
            "TOUR_HAS_OTHER_PURPOSE",
        ],
        "departure_accompaniment_and_weather": [
            # W_ANZBEGL is the direct count; anzbegl and anzpers are dropped.
            # HOUSEHOLD_ACCOMPANIED is the binary canonicalisation of W_BEGL_HH.
            "W_ANZBEGL",
            "HOUSEHOLD_ACCOMPANIED",
            "P_STWETTER",
        ],
    },
    "household": {
        "composition_and_housing": [
            # household_type is the accepted thesis life-stage construct;
            # hhgr_gr remains because it is explicitly size, not household type.
            "household_type",
            "hhgr_gr",
            "H_MIETE",
        ],
        "socioeconomic_context": [
            # Official analytical status replaces hheink_gr2/aq_eink_gr.
            "oek_status",
        ],
        "mobility_resources": [
            # Direct separate counts replace combined anzpedrad and mobtyp.
            "H_CS",
            "H_ANZPED",
            "H_ANZRAD",
        ],
        "spatial_and_built_environment": [
            # RegioStaR4 is the sole RegioStaR resolution. XMStadt is preferred
            # over highly overlapping MSIndex for its direct minute-city scale.
            "RegioStaR4",
            "XMStadt",
            "quali_opnv",
            "min_bab",
            "min_ozmz",
        ],
    },
    "person": {
        "socio_demographics": [
            # alter_gr5 is the sole age representation.
            "HP_SEX",
            "alter_gr5",
        ],
        "employment": [
            # Official 0/1 status replaces P_TAET/taet/taet_diff/P_BKAT.
            "erwerb",
        ],
        "personal_mobility_resources": [
            # Separate availability variables replace combined vpedrad.
            "P_VPED",
            "P_VRAD",
        ],
        "carsharing_and_mobility_limitation": [
            # Binary carsharing replaces carsharing_diff.
            "carsharing",
            "mobein",
        ],
    },
    "vehicle_alternative_specific": {
        "core_vehicle_attributes": [
            # A_ANTRIEB is preferred over the three-level antrieb collapse so
            # hybrid, plug-in hybrid and BEV remain identifiable.
            "POWERTRAIN",
            # seg_kba_gr is the documented grouped segment representation.
            "SEGMENT",
            # Reporting year minus A_BAUJ replaces A_BAUJ/bauj_gr.
            "VEHICLE_AGE",
        ],
        "optional_vehicle_attributes": [
            # A_HALTER holder type, retained only as an optional specification.
            "HOLDER",
            # Derived from KBA segment and construction year; treat as an
            # alternative representation rather than automatically combining
            # it with SEGMENT + VEHICLE_AGE in the same specification.
            "STATUS",
        ],
    },
}


# Deliberately deferred extensions (not baseline registry entries):
# - TOUR_MAIN_PURPOSE and TOUR_ELAPSED_TIME_MIN: closed-tour sensitivity only.
# - route/OD mode distances and durations: later accessibility extension.
# - origin/destination spatial classifications: later trip-spatial extension.
# - HOME_CHARGING, DRIVING_EXPERIENCE_YEARS, P_ARB_ENTF, hoff1,
#   hind_opnv_anz, ENGINE_POWER_KW and PARKING: documented in the model
#   input QA as structurally unavailable for most rows in the accepted strict
#   universe and therefore omitted from the clean baseline.
