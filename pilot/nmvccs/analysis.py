# -*- coding: utf-8 -*-
"""NMVCCS 记录机制审计：步骤1（先验+多数类基线）、步骤2（记录完整率梯度）、
步骤4（严重度分类器 AUROC/AP/top-decile）、步骤5（成本扫描）。"""
import os, re, json
import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(BASE, "tables")
os.makedirs(OUT, exist_ok=True)
CAT = os.path.join(RAW, "pub_extracted", "formats.sas7bcat")

def load(name, sub="pub_extracted"):
    df, meta = pyreadstat.read_sas7bdat(os.path.join(RAW, sub, name), catalog_file=CAT)
    return df

# ---------- 载入 ----------
crash = load("crash.sas7bdat")
gv = load("gv.sas7bdat")
cv = load("cv.sas7bdat")
env = load("env.sas7bdat")
pca = load("pca.sas7bdat")
occ = load("occ.sas7bdat")
intdr = load("intdr.sas7bdat", sub="interviews_extracted")

KEY = ["SCASEID", "VEHNO"]

# ---------- 步骤1：案件级伤情先验 ----------
sev_order = ["O - No injury", "C - Possible injury", "B - Non-incapacitating injury",
             "A - Incapacitating injury", "K - Killed"]
sev_rank = {s: i for i, s in enumerate(sev_order)}
unk_codes_crash = ["U - Injury, severity unknown", "Unknown if injured",
                   "No PAR obtained", "No Person in crash"]

prior = crash["INJSEVA"].value_counts(dropna=False)
prior_tbl = pd.DataFrame({"类别": prior.index, "案件数": prior.values})
prior_tbl["占比%"] = (prior_tbl["案件数"] / len(crash) * 100).round(2)
major = prior.idxmax()
maj_acc = prior.max() / len(crash)
step1 = {
    "n_cases": int(len(crash)),
    "distribution": prior_tbl.to_dict("records"),
    "majority_class": str(major),
    "majority_baseline_acc": round(float(maj_acc), 4),
}
prior_tbl.to_csv(os.path.join(OUT, "step1_case_severity_prior.csv"), index=False)

# ---------- 车级数据集 ----------
pca_k = pca.copy()
# 合并车、环境、警报字段
veh = pca_k.merge(gv[KEY + ["INJSEVV", "BODYTYPE", "CLASS", "MODELYR", "INTRANSP", "TOWPAR"]],
                  on=KEY, how="left")
veh = veh.merge(cv[KEY + ["ALCOHOL", "DRUGS", "BELTUSE", "TRAVELSP", "DLSTATUS"]],
                on=KEY, how="left")
env_feat = ["SPLIMIT", "SURCOND", "NATLGT", "ALIGN", "PROFILE", "RELJNCT", "TRAFFLOW"] + \
           [f"WEATHER{i}" for i in range(1, 9)]
veh = veh.merge(env[KEY + env_feat], on=KEY, how="left")
veh["WEATHER_any"] = veh[[f"WEATHER{i}" for i in range(1, 9)]].apply(
    lambda r: "One or more present" if any(str(x).startswith("Present") for x in r)
    else ("Unknown if present" if any("Unknown" in str(x) for x in r) else "None"), axis=1)

# 访谈完成情况（按车）
intdr_k = intdr[["SCASEID", "VEHNO", "DI_INTVRESULT", "INTERVIEWTYPE"]].copy()
intdr_k["interview_completed"] = intdr_k["DI_INTVRESULT"].astype(str).str.contains(
    "Complete|Partial", case=False, regex=True)
intdr_k["surrogate"] = intdr_k["INTERVIEWTYPE"].astype(str).str.contains("Surrogate", case=False)
intdr_small = intdr_k.groupby(KEY).agg(
    interview_completed=("interview_completed", "max"),
    surrogate=("surrogate", "max")).reset_index()
veh = veh.merge(intdr_small, on=KEY, how="left")
veh["interview_completed"] = veh["interview_completed"].fillna(False).astype(bool)
veh["surrogate"] = veh["surrogate"].fillna(False).astype(bool)

veh["sev_rank"] = veh["INJSEVV"].map(sev_rank)
veh["sev_known"] = veh["INJSEVV"].isin(sev_rank)

# ---------- 编码归类工具 ----------
def classify_code(label):
    """按编码手册标签把取值归类：unknown / not_reported / no_par / na / known。"""
    s = str(label).strip().lower()
    if s in ("nan", ""):
        return "na"
    # 无值标签的数值字段（如 SLEEP24HRS）：9999 = Unknown，8888 = No driver present
    if s in ("9999", "9999.0"):
        return "unknown"
    if s in ("8888", "8888.0"):
        return "na"
    if "no par obtained" in s:
        return "no_par"
    if "not reported" in s or "did not indicate" in s:
        return "not_reported"
    if "unknown" in s:
        return "unknown"
    if ("not applicable" in s or "not coded to this vehicle" in s or "not involved" in s
            or "no driver present" in s or "not collected" in s):
        return "na"
    return "known"

# 步骤2核心字段（车级）
FIELDS_STEP2 = [
    ("CRITREASON", "pca"), ("PREEVENT", "pca"), ("PREMOVE", "pca"), ("AVMAN", "pca"),
    ("FATIGUE", "pca"), ("INATTEN", "pca"), ("MEDS", "pca"), ("ILLNESS", "pca"),
    ("TIMEDRIVING", "pca"), ("SLEEP24HRS", "pca"),
    ("ALCOHOL", "cv"), ("DRUGS", "cv"), ("BELTUSE", "cv"), ("TRAVELSP", "cv"), ("DLSTATUS", "cv"),
    ("SURCOND", "env"), ("NATLGT", "env"),
]
groups = ["O - No injury", "C - Possible injury", "B - Non-incapacitating injury",
          "A - Incapacitating injury", "K - Killed"]
gshort = {"O - No injury": "O", "C - Possible injury": "C", "B - Non-incapacitating injury": "B",
          "A - Incapacitating injury": "A", "K - Killed": "K"}

rows, grad_rows = [], []
for var, src in FIELDS_STEP2:
    ser = veh[var].map(classify_code)
    n_tot = len(veh)
    overall = ser.value_counts(normalize=True)
    rec = {"字段": var, "总体未知率%": round(float((ser == "unknown").mean()) * 100, 2),
           "总体未报告率%": round(float((ser == "not_reported").mean()) * 100, 2),
           "总体无PAR率%": round(float((ser == "no_par").mean()) * 100, 2),
           "总体NA率%": round(float((ser == "na").mean()) * 100, 2)}
    rates = []
    for g in groups:
        mask = veh["INJSEVV"] == g
        sub = ser[mask]
        # 记录缺失率 = unknown + not_reported + no_par（在全部车辆中占比）
        miss = sub.isin(["unknown", "not_reported", "no_par"]).mean()
        na = (sub == "na").mean()
        rec[f"{gshort[g]}_缺失%"] = round(float(miss) * 100, 2)
        rec[f"{gshort[g]}_NA%"] = round(float(na) * 100, 2)
        rates.append(float(miss))
    rec["样本数O/C/B/A/K"] = "/".join(str(int((veh["INJSEVV"] == g).sum())) for g in groups)
    rows.append(rec)
    rho, p = spearmanr(list(range(5)), rates)
    grad_rows.append({"字段": var, "Spearman_rho(缺失率~严重度)": round(float(rho), 3),
                      "p值": round(float(p), 4),
                      "方向": "随严重度上升" if rho > 0 else ("随严重度下降" if rho < 0 else "无梯度")})

step2_tbl = pd.DataFrame(rows)
step2_tbl.to_csv(os.path.join(OUT, "step2_missingness_by_severity.csv"), index=False)
grad_tbl = pd.DataFrame(grad_rows)
grad_tbl.to_csv(os.path.join(OUT, "step2_gradient_tests.csv"), index=False)

# 访谈完成率梯度（记录机制的直接证据）
int_rows = []
for g in groups + ["U - Injury, severity unknown", "No PAR obtained"]:
    mask = veh["INJSEVV"] == g
    n = int(mask.sum())
    if n == 0:
        continue
    int_rows.append({
        "严重度组": g if g in gshort else g,
        "车辆数": n,
        "访谈完成率%": round(float(veh.loc[mask, "interview_completed"].mean()) * 100, 2),
        " surrogate访谈占比%": round(float(veh.loc[mask, "surrogate"].mean()) * 100, 2),
    })
int_tbl = pd.DataFrame(int_rows)
int_tbl.to_csv(os.path.join(OUT, "step2_interview_completion_by_severity.csv"), index=False)

# ---------- 步骤4：严重度分类（K/A vs 其他） ----------
model_df = veh[veh["sev_known"]].copy()  # 仅已知严重度
model_df = model_df[~model_df["INJSEVV"].isin(["No Person in vehicle", "Died prior to crash"])]
model_df["y"] = model_df["INJSEVV"].isin(["K - Killed", "A - Incapacitating injury"]).astype(int)

# 特征组
scene_feats = ["BODYTYPE", "CLASS", "MODELYR", "INTRANSP", "ALCOHOL", "DRUGS", "BELTUSE",
               "TRAVELSP", "DLSTATUS", "SPLIMIT", "SURCOND", "NATLGT", "ALIGN",
               "PROFILE", "RELJNCT", "TRAFFLOW", "WEATHER_any"]
deep_feats = ["PREMOVE", "AVMAN", "RIGHTOFWAY", "FATIGUE", "INATTEN", "MEDS", "ILLNESS",
              "TIMEDRIVING", "CONVERSE", "TOOCLOSE", "MISJUDGE", "FALSEASMPT", "ILLMAN",
              "AGGRACT", "INEVASION", "DRPERROR", "UPSET", "HURRY", "HRSAWAKE", "SLEEP24HRS",
              "ACCTYPE", "PREEVENT", "CRITREASON"]

def prep(Xdf):
    """类别列转字符串（保留 unknown 等编码为独立类别）；数值列转 float 且缺失指示。"""
    X = Xdf.copy()
    cat_cols, num_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and c in ("MODELYR", "SPLIMIT", "HRSAWAKE", "SLEEP24HRS"):
            v = pd.to_numeric(X[c], errors="coerce")
            X[c] = v
            X[c + "__miss"] = v.isna().astype(float)
            num_cols += [c, c + "__miss"]
        else:
            X[c] = X[c].astype(str)
            cat_cols.append(c)
    return X, cat_cols, num_cols

def run_model(feats, name):
    d = model_df.dropna(subset=feats, how="any") if False else model_df
    X, cat_cols, num_cols = prep(d[feats])
    y = d["y"].values
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    oof = np.zeros(len(d))
    # Vehicles from the same crash share crash-level fields and outcomes, so
    # folds are grouped by case (SCASEID) to keep each crash in one fold.
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y, groups=d["SCASEID"].values):
        pipe_feats = [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
                      ("num", StandardScaler(), num_cols)]
        from sklearn.pipeline import Pipeline
        pipe = Pipeline([("pre", ColumnTransformer(pipe_feats)),
                         ("lr", LogisticRegression(max_iter=2000, C=0.5))])
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    auc = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    n = len(y); k = int(np.ceil(n * 0.1))
    top = np.argsort(-oof)[:k]
    top_precision = float(y[top].mean())
    tdc = float(y[top].sum() / max(1, y.sum()))
    prev = float(y.mean())
    res = {"模型": name, "n": int(n), "正例(严重K/A)数": int(y.sum()), "先验prev": round(prev, 4),
           "AUROC": round(float(auc), 4), "AP": round(float(ap), 4),
           "top_decile_coverage": round(tdc, 4),
           "top_decile_lift": round(top_precision / prev, 2)}
    return res, oof, y

res_m1, oof_m1, y1 = run_model(scene_feats, "M1 PAR/现场字段")
res_m2, oof_m2, y2 = run_model(scene_feats + deep_feats, "M2 深度调查字段")
step4_tbl = pd.DataFrame([res_m1, res_m2])
step4_tbl.to_csv(os.path.join(OUT, "step4_severity_models.csv"), index=False)

# ---------- 步骤5：成本扫描（用 M2 的 OOF 概率；M1 同法） ----------
def cost_scan(oof, y, tag):
    n = len(y); prev = y.mean()
    recs = []
    for c in range(1, 51):
        t = 1.0 / (c + 1)
        pred = (oof >= t).astype(int)
        fn = int(((pred == 0) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        cost = (c * fn + fp) / n
        # For an FN:FP ratio of c:1, marking every row severe incurs only
        # false-positive cost, whereas marking no row severe incurs c times
        # the severe prevalence in false-negative cost.
        cost_all = 1 - prev         # 全标严重：所有非严重案例误标
        cost_none = c * prev         # 全不标：所有严重案例漏标
        recs.append({"c_FN:FP": f"{c}:1", "阈值": round(t, 4), "FN": fn, "FP": fp,
                     "模型期望成本": round(float(cost), 5),
                     "全标严重成本": round(float(cost_all), 5),
                     "全不标成本": round(float(cost_none), 5),
                     "优于全标": cost < cost_all, "优于全不标": cost < cost_none,
                     "优于两者": (cost < cost_all) and (cost < cost_none)})
    t = pd.DataFrame(recs)
    t.to_csv(os.path.join(OUT, f"step5_cost_scan_{tag}.csv"), index=False)
    wins = t[t["优于两者"]]["c_FN:FP"].tolist()
    return {"tag": tag, "window": (wins[0], wins[-1]) if wins else None, "table": t}

cs_m2 = cost_scan(oof_m2, y2, "M2")
cs_m1 = cost_scan(oof_m1, y1, "M1")

step5_summary = {
    "M2": {"成本比窗口": cs_m2["window"],
           "细节": cs_m2["table"][cs_m2["table"]["c_FN:FP"].isin(["1:1","5:1","10:1","20:1","50:1"])].to_dict("records")},
    "M1": {"成本比窗口": cs_m1["window"]},
}

summary = {"step1": step1, "step4": step4_tbl.to_dict("records"), "step5": step5_summary,
           "n_vehicles_total": int(len(veh)),
           "n_vehicles_known_sev": int(veh["sev_known"].sum())}
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

print("=== STEP1 ==="); print(prior_tbl.to_string(index=False))
print(f"majority={major}, acc={maj_acc:.4f}")
print("\n=== STEP2 missingness ==="); print(step2_tbl.to_string(index=False))
print("\n=== STEP2 gradient ==="); print(grad_tbl.to_string(index=False))
print("\n=== STEP2 interview ==="); print(int_tbl.to_string(index=False))
print("\n=== STEP4 ==="); print(step4_tbl.to_string(index=False))
print("\n=== STEP5 ==="); print(json.dumps(step5_summary, ensure_ascii=False, indent=1, default=str))
