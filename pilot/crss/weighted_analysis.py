# -*- coding: utf-8 -*-
"""
CRSS 2024 加权复核：在抽样权重 WEIGHT 下重算先验/基线、加权逻辑回归 OOF 指标与成本窗口。
手册依据（CRSS Analytical User's Manual 2016-2024, DOT HS 813 796）：
  "The final CRSS weight variable that incorporates the above steps is called WEIGHT"
  （accident 表 Case Weight，"used to produce national estimates from the data"）。
  未加权结果来自 analysis.py（summary.json / step5_cost_scan.csv），此处逐项对照。
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "CRSS2024/CRSS2024CSV"
SEV_ORDER = [0, 1, 2, 3, 4]

use_acc = ["CASENUM", "WEIGHT", "REGION", "REGIONNAME", "URBANICITY", "URBANICITYNAME",
           "MONTH", "DAY_WEEK", "HOUR", "WEATHER", "WEATHERNAME", "LGT_COND", "LGT_CONDNAME",
           "MAN_COLL", "MAN_COLLNAME", "HARM_EV", "HARM_EVNAME", "REL_ROAD", "REL_ROADNAME",
           "TYP_INT", "TYP_INTNAME", "WRK_ZONE", "WRK_ZONENAME", "INT_HWY", "INT_HWYNAME",
           "PEDS", "PERNOTMVIT", "VE_FORMS", "PERMVIT", "MAX_SEV", "MAX_SEVNAME"]
use_veh = ["CASENUM", "VEH_NO", "BODY_TYP", "BODY_TYPNAME", "MOD_YEAR", "MOD_YEARNAME",
           "HIT_RUN", "SPEEDREL", "SPEEDRELNAME", "VSPD_LIM", "VSPD_LIMNAME", "VSURCOND",
           "VSURCONDNAME", "VALIGN", "VALIGNNAME", "VPROFILE", "VPROFILENAME", "VNUM_LAN",
           "VNUM_LANNAME", "GVWR_TO", "GVWR_TONAME", "TOWED", "TOWEDNAME"]
use_per = ["CASENUM", "VEH_NO", "PER_NO", "PER_TYP", "INJ_SEV", "INJ_SEVNAME", "AGE", "AGENAME",
           "SEX", "SEXNAME", "REST_USE", "REST_USENAME", "DRINKING", "DRINKINGNAME", "DRUGS",
           "DRUGSNAME", "ALC_RES", "ALC_RESNAME", "HOSPITAL", "HOSPITALNAME", "EJECTION",
           "EJECTIONNAME"]

acc = pd.read_csv(f"{DATA}/accident.csv", usecols=use_acc)
veh = pd.read_csv(f"{DATA}/vehicle.csv", usecols=use_veh)
per = pd.read_csv(f"{DATA}/person.csv", usecols=use_per)

def classify(name):
    if pd.isna(name):
        return "blank"
    s = str(name)
    if "Reported as Unknown" in s:
        return "reported_as_unknown"
    if "Not Reported" in s:
        return "not_reported"
    if "Not Applicable" in s:
        return "not_applicable"
    return "valid"

# ---------- (1) 加权先验与多数类基线 ----------
print("===== (1) 严重度先验与多数类基线：未加权 vs 加权 =====")
acc["sev"] = acc["MAX_SEV"]
n = len(acc)
t = acc.groupby(["sev", "MAX_SEVNAME"]).agg(n=("CASENUM", "count"), w=("WEIGHT", "sum")).reset_index()
t["pct"] = (t.n / n * 100).round(2)
t["wtd_pct"] = (t.w / t.w.sum() * 100).round(2)
print(t[["sev", "MAX_SEVNAME", "n", "pct", "wtd_pct"]].to_string(index=False))
maj_uw = (acc.groupby("sev").size() / n).max()
maj_w = (acc.groupby("sev").WEIGHT.sum() / acc.WEIGHT.sum()).max()
print(f"多数类基线(全样本, 5类): 未加权 {maj_uw:.4f} / 加权 {maj_w:.4f}")

# ---------- 特征工程（与 analysis.py 步骤4完全一致） ----------
feat = acc[acc.sev.isin(SEV_ORDER)].copy()
feat["fatal"] = (feat.sev == 4).astype(int)
v = veh.copy()
v["age"] = 2024 - v.MOD_YEAR.where(v.MOD_YEAR.between(1900, 2024))
v["is_moto"] = v.BODY_TYP.between(80, 89).astype(int)
v["is_heavy"] = v.GVWR_TO.ge(3).astype(int)
v["speeding"] = v.SPEEDREL.between(1, 4).astype(int)
v["spd_miss"] = (~v.VSPD_LIM.between(1, 97)).astype(int)
va = v.groupby("CASENUM").agg(
    n_veh=("VEH_NO", "count"),
    moto_any=("is_moto", "max"), heavy_any=("is_heavy", "max"),
    speeding_any=("speeding", "max"),
    veh_age_mean=("age", "mean"), spd_lim_med=("VSPD_LIM", "median"),
    spdlim_miss_any=("spd_miss", "max"))
feat = feat.merge(va, left_on="CASENUM", right_index=True, how="left")
feat["hour_bin"] = pd.cut(feat.HOUR, [-1, 5, 11, 17, 23],
                          labels=["0-5", "6-11", "12-17", "18-23"])
cat_cols = ["URBANICITYNAME", "REGIONNAME", "MONTH", "DAY_WEEK", "hour_bin",
            "WEATHERNAME", "LGT_CONDNAME", "MAN_COLLNAME", "HARM_EVNAME",
            "REL_ROADNAME", "TYP_INTNAME", "WRK_ZONENAME"]
num_cols = ["PEDS", "PERNOTMVIT", "VE_FORMS", "PERMVIT",
            "n_veh", "moto_any", "heavy_any", "speeding_any", "veh_age_mean",
            "spd_lim_med", "spdlim_miss_any"]
for c in cat_cols:
    feat[c] = feat[c].map(classify).where(feat[c].map(classify).eq("valid"), feat[c].astype(str))
    feat[c] = feat[c].replace({"reported_as_unknown": "Unknown", "not_reported": "NotReported"})

X = feat[cat_cols + num_cols]
y = feat["fatal"].values
w = feat["WEIGHT"].values  # 分析权重
pi_uw = y.mean()
pi_w = np.average(y, weights=w)
print(f"\n[建模样本] {len(feat)} crashes, fatal n={feat.fatal.sum()}")
print(f"fatal 先验: 未加权 {pi_uw:.4%} / 加权 {pi_w:.4%}")
print(f"二值多数类基线(非fatal): 未加权 {max(pi_uw,1-pi_uw):.4f} / 加权 {max(pi_w,1-pi_w):.4f}")
print(f"权重: min={w.min():.2f} max={w.max():.2f} 总和={w.sum():,.0f}")

pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)])
model = Pipeline([("pre", pre),
                  ("lr", LogisticRegression(max_iter=3000, C=1.0))])

# ---------- (2) 加权逻辑回归，5 折 OOF（fold 划分与 analysis.py 相同） ----------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_w = np.zeros(len(feat))
for tr, te in skf.split(X, y):
    model.fit(X.iloc[tr], y[tr], lr__sample_weight=w[tr])
    oof_w[te] = model.predict_proba(X.iloc[te])[:, 1]

auroc_w = roc_auc_score(y, oof_w, sample_weight=w)
ap_w = average_precision_score(y, oof_w, sample_weight=w)
k = int(0.1 * len(feat))
top = np.argsort(-oof_w)[:k]
tdc_w = (w[top] * y[top]).sum() / (w * y).sum()  # 被标中 fatal 加权质量 / 全部 fatal 加权质量
# 稳健性：按权重累计 10% 预算（flag 至累计权重达总权重 10%）
o = np.argsort(-oof_w)
cw = np.cumsum(w[o]) / w.sum()
topw = o[cw <= 0.1]
tdc_w2 = (w[topw] * y[topw]).sum() / (w * y).sum()
print(f"\n[步骤4·加权] 5折OOF: AUROC={auroc_w:.4f}  AP={ap_w:.4f}  top-decile coverage={tdc_w:.4f}")
print(f"[稳健性] 按权重预算10%的覆盖: {tdc_w2:.4f} (flag {len(topw)} 起)")

# ---------- (3) 加权成本扫描 1:1–50:1 ----------
print("\n===== (3) 加权成本扫描（阈值 1/(c+1)，加权期望成本） =====")
res = []
for c in range(1, 51):
    thr = 1.0 / (c + 1)
    flag = oof_w >= thr
    fn = ((~flag) & (y == 1))
    fp = (flag & (y == 0))
    model_cost = (c * (w * fn).sum() + (w * fp).sum()) / w.sum()
    cost_all = 1 - pi_w
    cost_none = c * pi_w
    res.append({"c": c, "thr": round(thr, 4), "model": round(model_cost, 5),
                "flag_all": round(cost_all, 5), "flag_none": round(cost_none, 5),
                "beats_both": bool(model_cost < cost_all and model_cost < cost_none)})
r = pd.DataFrame(res)
print(r[r.c.isin([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50])].to_string(index=False))
win = r[r.beats_both].c
print(f"加权窗口: c ∈ [{win.min() if len(win) else '-'}, {win.max() if len(win) else '-'}]" +
      (f" (连续 {len(win)} 个)" if len(win) else ""))
flag = oof_w >= 0.5
print(f"c=1 thr=0.5: flagged={flag.sum()}, 加权召回={(w[flag]*y[flag]).sum()/(w*y).sum():.3f}")

# ---------- (4) 与未加权结果逐项对照 ----------
print("\n===== (4) 未加权 vs 加权 对照 =====")
unw = json.load(open("summary.json"))
cmp_rows = [
    ("fatal 先验", unw["prevalence"], pi_w),
    ("多数类基线(非fatal)", 1 - unw["prevalence"], 1 - pi_w),
    ("AUROC (5折OOF)", unw["auroc"], auroc_w),
    ("Average Precision", unw["ap"], ap_w),
    ("先验 AP 参照", unw["prevalence"], pi_w),
    ("Top-decile coverage", unw["top_decile_cov"], tdc_w),
]
cmp_tab = pd.DataFrame(cmp_rows, columns=["指标", "未加权", "加权"])
cmp_tab["未加权"] = cmp_tab["未加权"].round(4)
cmp_tab["加权"] = cmp_tab["加权"].round(4)
print(cmp_tab.to_string(index=False))
print(f"成本窗口: 未加权 {unw['cost_window']} vs 加权 [{win.min()}, {win.max()}]")

# ---------- 保存 ----------
r.to_csv("weighted_step5_cost_scan.csv", index=False)
feat[["CASENUM", "fatal"]].assign(oof=oof_w).to_csv("weighted_oof_predictions.csv", index=False)
wsum = {"n_crashes": int(n),
        "fatal_prev_unweighted": float(pi_uw), "fatal_prev_weighted": float(pi_w),
        "maj_baseline_unweighted": float(max(pi_uw, 1 - pi_uw)),
        "maj_baseline_weighted": float(max(pi_w, 1 - pi_w)),
        "auroc_weighted": float(auroc_w), "ap_weighted": float(ap_w),
        "ap_baseline_weighted": float(pi_w),
        "top_decile_cov_weighted": float(tdc_w),
        "top_decile_cov_weighted_wbudget": float(tdc_w2),
        "cost_window_unweighted": unw["cost_window"],
        "cost_window_weighted": [int(win.min()), int(win.max())] if len(win) else None}
json.dump(wsum, open("weighted_summary.json", "w"), indent=2)
cmp_tab.to_csv("weighted_vs_unweighted_comparison.csv", index=False)
print("\nsaved: weighted_step5_cost_scan.csv, weighted_oof_predictions.csv, "
      "weighted_summary.json, weighted_vs_unweighted_comparison.csv")
