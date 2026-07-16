HH_SELECTED_COLS = [
    # ID / weights / sample diagnostics
    "H_ID",
    "H_GEW",
    "H_HOCH",
    "MODE",
    "BASISAUF",
    "M_CAR",
    "H_ART",

    # household composition
    "H_GR",
    "hhgr_gr",
    "HP_ALTER_1",
    "HP_ALTER_2",
    "HP_ALTER_3",
    "HP_ALTER_4",
    "HP_ALTER_5",
    "HP_ALTER_6",

    # housing / socio-economic context
    "H_MIETE",
    "hheink_gr2",
    "aq_eink_gr",
    "oek_status",

    # mobility tools / vehicle ownership
    "H_ANZAUTO",
    "anzauto_gr3",
    "H_CS",
    "mobtyp",
    "anzmotmop",
    "H_ANZPED",
    "H_ANZRAD",
    "anzpedrad",

    # car dependence / EV charging context
    "A_LADEN",
    "H_NOT",

    # spatial context / built environment
    "BLAND",
    "RegioStaR2",
    "RegioStaR4",
    "RegioStaR7",
    "RegioStaR17",
    "XMStadt",
    "MSIndex",
    "bus28",
    "tram28",
    "bahn28",
    "min_bab",
    "min_ozmz",
    "haustyp",
    "wohnlage",
    "quali_nv",
    "quali_opnv",
]

PERSON_SELECTED_COLS = [
 # ID / weights / survey diagnostics
 "HP_ID",
 "H_ID",
 "P_ID",
 "P_GEW",
 "P_HOCH",
 "MODE",
 "MODE_HH",
 "BASISAUF",
 "PROXY",
 "PROXY_01",
 "INT_TYP",

 # module flags
 "M_HOFF",
 "M_OPNV",
 "M_MOB",

 # Stichtag / temporal context
 "ST_WOTAG",
 "arbwo",
 "kernwo",
 "feiertag",
 "saison",
 "P_STWETTER",

 # socio-demographics
 "HP_SEX",
 "HP_ALTER",
 "alter_gr5",
 "alter_gr6",

 # licence and car availability
 "P_FS_PKW",
 "P_FSJAHR",
 "fsjahr_gr",
 "fsalter",
 "P_VAUTO",
 "P_STKFZ",

 # employment / occupation
 "P_TAET",
 "taet",
 "taet_diff",
 "P_BKAT",
 "erwerb",

 # work / commute / home office
 "P_ARB_ANZ",
 "P_ARB_ENTF",
 "arb_entf_gr2",
 "arb_vm2",
 "arb_vm2_diff",
 "hoff1",
 "hoff1_diff",
 "P_HOFF2",
 "hoff2_gr",
 "P_STARB1",
 "starb2",

 # personal mobility tool availability
 "P_VPED",
 "P_VRAD",
 "vpedrad",

 # usual mobility behaviour
 "P_NUTZ_AUTO",
 "P_NUTZ_RAD",
 "P_NUTZ_OPNV",
 "P_NUTZ_FUSS",
 "P_NUTZ_CS",

 # public transport ticket / barriers / perceived accessibility
 "P_FKARTE",
 "hind_opnv",
 "hind_opnv_anz",
 "P_ERR_ARB",
 "P_ERR_EINK",
 "P_ERR_BESU",
 "P_ERR_FREI",

 # carsharing
 "carsharing",
 "carsharing_diff",

 # mobility limitation / participation
 "mobein",
 "verzicht",

 # person-day mobility summary: descriptive / filtering, not main explanatory variables
 "mobil",
 "mobil_diff",
 "anzwege1",
 "anzwege3",
 "perskm1",
 "perskm2",
 "persmin1",
 "persmin2",

 # derived person mobility profiles: descriptive / sensitivity
 "pergrup1",
 "pergrup2",
 "seg_vm",
 "multimodal",
 "intermodal",
 "intermodal2",
]

WEGE_SELECTED_COLS = [
 # IDs / weights / filters
 "HP_ID", "H_ID", "P_ID", "W_ID",
 "W_GEW", "W_HOCH", "W_GEW_PKM", "W_HOCH_PKM",
 "MODE", "BASISAUF", "PROXY_01", "W_RBW",

 # Time
 "ST_MONAT", "ST_JAHR", "ST_WOTAG", "feiertag", "saison",
 "W_SZ", "W_AZ", "W_FOLGETAG",
 "sz_gr1", "sz_gr2", "az_gr1", "az_gr2",
 "wegmin", "wegmin_imp2", "wegmin_gr", "wegmin_imp2_gr",

 # Purpose
 "W_ZWECK", "zweck", "hwzweck1", "hwzweck2",
 "W_ZWD", "W_ZWDE", "W_ZWDP", "W_ZWDF",

 # Distance / speed
 "wegkm", "wegkm_imp", "wegkm_gr", "wegkm_imp_gr",
 "tempo", "tempo_imp",

 # Modes and vehicle allocation
 "W_VM_A", "W_VM_B", "W_VM_C", "W_VM_F", "W_VM_D", "W_VM_E",
 "W_VM_G", "W_VM_H", "W_VM_I", "W_VM_J", "W_VM_K", "W_VM_L",
 "W_VM_M", "W_VM_N", "W_VM_O", "W_VM_P", "W_VM_Q", "W_VM_R",
 "W_VM_S", "W_VM_T", "W_VM_U", "W_VM_Z",
 "hvm", "hvm_diff1", "hvm_diff2", "hvm_oev", "hvm_imp",
 "vm_kombi", "weg_intermod", "weg_intermod2",
 "W_FMF", "pkw_fmf", "W_WAUTO",

 # Accompaniment
 "W_ANZBEGL", "anzbegl", "anzpers", "W_BEGL_HH",

 # Weather
 "P_STWETTER",

 # Route alternatives / accessibility
 "auto_dist", "auto_dauer", "auto_dauer_vgl",
 "rad_dist", "rad_dauer",
 "opnv_dist", "opnv_dist_transit", "opnv_dist_fuss",
 "opnv_dauer_brutto", "opnv_dauer_brutto_vgl",
 "opnv_dauer_netto", "opnv_dauer_transit",
 "opnv_dauer_fuss", "opnv_dauer_wartezeit",

 # Household/person convenience variables copied in Wege
 "H_GR", "H_ANZAUTO", "auto",
 "P_FS_PKW", "P_VAUTO", "P_STKFZ",

 # Spatial / built environment
 "BLAND", "RegioStaR17", "RegioStaR7", "RegioStaR4", "RegioStaR2",
 "RegioStaRGem7", "RegioStaRGem5",
 "XMStadt", "XMStadt_SO", "XMStadt_ZO",
 "MSIndex", "MSIndex_SO", "MSIndex_ZO",
 "bus28", "bus28_so", "bus28_zo",
 "tram28", "tram28_so", "tram28_zo",
 "bahn28", "bahn28_so", "bahn28_zo",
 "min_bab", "min_bab_so", "min_bab_zo",
 "min_ozmz", "min_ozmz_so", "min_ozmz_zo",
 "quali_nv", "quali_nv_so", "quali_nv_zo",
 "quali_opnv", "quali_opnv_so", "quali_opnv_zo",
]

AUTOS_SELECTED_COLS = [
 "H_ID", "A_ID", "A_GEW", "M_CAR", "H_ANZAUTO",
 "A_ANTRIEB", "antrieb", "A_HALTER",
 "A_BAUJ", "bauj_gr", "A_ERWJ", "erwj_gr",
 "A_JAHRESFL", "jahresfl_gr",
 "A_KW", "kw_gr", "A_PS", "ps_gr",
 "seg_kba", "seg_kba_gr", "status",
 "A_STELL", "A_LADEN", "anzladen",
]