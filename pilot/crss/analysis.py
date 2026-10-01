# -*- coding: utf-8 -*-
"""
CRSS 2024 记录机制审计：严重度先验、记录完整率梯度、严重度分类器与成本扫描。
数据：NHTSA CRSS 2024 CSV (https://static.nhtsa.gov/nhtsa/downloads/CRSS/2024/CRSS2024CSV.zip)
编码约定（CRSS Analytical User's Manual 2016-2024, DOT HS 813 796）：
  8/98/998/9998 = Not Reported（警官未填报）；9/99/999/9999 = Reported as Unknown（填报为未知）。
  注意 person/crash 级 INJ_SEV/MAX_SEV 的 9 标签为 "Unknown/Not Reported"（二者合并），如实报告。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "CRSS2024/CRSS2024CSV"
SEV_ORDER = [0, 1, 2, 3, 4]
SEV_LABEL = {0: "无伤情(O)", 1: "可能受伤(C)", 2: "轻伤疑似(B)", 3: "重伤疑似(A)", 4: "死亡(K)"}

# ---------- 读取 ----------
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

# ---------- 步骤1：严重度先验（官方 MAX_SEV 为主，person 聚合交叉校验） ----------
sev_rank = {0: 0, 1: 1, 2: 2, 3: 3, 5: 3.5, 6: 3.6, 4: 4, 9: 9}
per["sev_r"] = per.INJ_SEV.map(sev_rank)
gsev = per.groupby("CASENUM")["sev_r"].max()
acc["sev_agg"] = acc.CASENUM.map(gsev).fillna(9).astype(int)
# 与官方 MAX_SEV 对照
chk = (acc["sev_agg"].astype(float) == acc["MAX_SEV"].astype(float)).mean()
print(f"[校验] person聚合严重度 == accident.MAX_SEV 一致率: {chk:.4f}")
print(pd.crosstab(acc.MAX_SEV, acc.sev_agg).to_string())
# 二值 fatal 标签两种口径完全一致
f1 = acc.MAX_SEV.eq(4)
f2 = acc.CASENUM.map(per.assign(f=per.INJ_SEV.eq(4).astype(int)).groupby("CASENUM")["f"].max()).fillna(0).astype(int).eq(1)
print(f"[校验] fatal 标签(MAX_SEV==4 vs person-any-INJ_SEV==4) 一致: {(f1==f2).all()}")

acc["sev"] = acc["MAX_SEV"]  # 分组采用官方 crash 级严重度
n = len(acc)
print("\n===== 步骤1 严重度先验（crash 级 MAX_SEV, 未加权样本占比 / 加权全国估计占比） =====")
out1 = pd.DataFrame({"n": acc.groupby(["sev", "MAX_SEVNAME"]).size()}).reset_index()
out1["pct"] = (out1.n / n * 100).round(2)
w = acc.groupby(["sev", "MAX_SEVNAME"]).WEIGHT.sum().reset_index(name="w")
out1 = out1.merge(w, on=["sev", "MAX_SEVNAME"])
out1["wtd_pct"] = (out1.w / out1.w.sum() * 100).round(2)
print(out1[["sev", "MAX_SEVNAME", "n", "pct", "wtd_pct"]].to_string(index=False))
maj = (acc.groupby("sev").size() / n).max()
print(f"多数类(无伤情)基线准确率(未加权): {maj:.4f}; (加权): "
      f"{(acc.groupby('sev').WEIGHT.sum()/acc.WEIGHT.sum()).max():.4f}")

# ---------- 步骤2：字段级记录完整率按严重度分组 ----------
def classify(name):
    """把 NAME 解码值分为 valid / reported_as_unknown / not_reported / not_applicable / blank"""
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

def miss_table(df, field, name_col, level_label):
    """对给定字段，返回按严重度分组的各类编码占比（行=严重度）"""
    d = df[["CASENUM", field, name_col]].copy()
    d["cat"] = d[name_col].map(classify)
    # 聚合到 crash 级：该层记录中 missing(unknown+not reported) 占比
    g = d.groupby(["CASENUM", "cat"]).size().unstack(fill_value=0)
    for c in ["valid", "reported_as_unknown", "not_reported", "not_applicable", "blank"]:
        if c not in g.columns:
            g[c] = 0
    g["tot"] = g.sum(axis=1)
    g["miss_share"] = (g["reported_as_unknown"] + g["not_reported"]) / g["tot"]
    g["unk_share"] = g["reported_as_unknown"] / g["tot"]
    g["nr_share"] = g["not_reported"] / g["tot"]
    j = acc.set_index("CASENUM")["sev"]
    g = g.join(j).dropna(subset=["sev"])
    return g

ACC_FIELDS = [("WEATHER", "WEATHERNAME"), ("LGT_COND", "LGT_CONDNAME"),
              ("MAN_COLL", "MAN_COLLNAME"), ("HARM_EV", "HARM_EVNAME"),
              ("REL_ROAD", "REL_ROADNAME"), ("TYP_INT", "TYP_INTNAME"),
              ("WRK_ZONE", "WRK_ZONENAME"), ("INT_HWY", "INT_HWYNAME")]
VEH_FIELDS = [("MOD_YEAR", "MOD_YEARNAME"), ("BODY_TYP", "BODY_TYPNAME"),
              ("GVWR_TO", "GVWR_TONAME"), ("SPEEDREL", "SPEEDRELNAME"),
              ("VSPD_LIM", "VSPD_LIMNAME"), ("VSURCOND", "VSURCONDNAME"),
              ("VALIGN", "VALIGNNAME"), ("VPROFILE", "VPROFILENAME"),
              ("VNUM_LAN", "VNUM_LANNAME"), ("TOWED", "TOWEDNAME")]
PER_FIELDS = [("AGE", "AGENAME"), ("SEX", "SEXNAME"), ("REST_USE", "REST_USENAME"),
              ("DRINKING", "DRINKINGNAME"), ("DRUGS", "DRUGSNAME"),
              ("ALC_RES", "ALC_RESNAME"), ("HOSPITAL", "HOSPITALNAME"),
              ("EJECTION", "EJECTIONNAME")]

print("\n===== 步骤2 缺失编码占比按严重度分组（%）: unknown+not_reported 合计 =====")
rows = []
for f, nm, lvl, df in ([(f, nm, "accident", acc) for f, nm in ACC_FIELDS] +
                       [(f, nm, "vehicle", veh) for f, nm in VEH_FIELDS] +
                       [(f, nm, "person", per) for f, nm in PER_FIELDS]):
    g = miss_table(df, f, nm, lvl)
    row = {"level": lvl, "field": f}
    for s in SEV_ORDER:
        sub = g[g.sev == s]
        row[f"S{s}"] = round(sub.miss_share.mean() * 100, 2) if len(sub) else np.nan
    # 单调性：Spearman（严重度序 vs 缺失率）
    vals = [row[f"S{s}"] for s in SEV_ORDER]
    rho, p = stats.spearmanr(SEV_ORDER, vals)
    row["spearman_rho"] = round(rho, 3)
    row["p"] = "<0.05" if p < 0.05 else round(p, 3)
    rows.append(row)
tab2 = pd.DataFrame(rows)
print(tab2.to_string(index=False))

# 细分：unknown vs not_reported 分列（每严重度组，选代表性字段）
print("\n--- 细分：Reported-as-Unknown 与 Not-Reported 分列（person级，每crash均值%） ---")
rows2 = []
for f, nm in [("DRINKING", "DRINKINGNAME"), ("DRUGS", "DRUGSNAME"), ("AGE", "AGENAME"),
              ("ALC_RES", "ALC_RESNAME"), ("REST_USE", "REST_USENAME")]:
    g = miss_table(per, f, nm, "person")
    row = {"field": f}
    for s in SEV_ORDER:
        sub = g[g.sev == s]
        row[f"S{s}_unk"] = round(sub.unk_share.mean() * 100, 2)
        row[f"S{s}_nr"] = round(sub.nr_share.mean() * 100, 2)
    rows2.append(row)
print(pd.DataFrame(rows2).to_string(index=False))

# BAC 检测率（调查深度代理）：ALC_RES 中实际给出数值结果的比例
per["bac_given"] = (~per.ALC_RESNAME.astype(str).str.contains(
    "Test Not Given|Not Reported|Reported as Unknown|Results Unknown|Positive Reading", na=False)).astype(int)
bac = per.groupby("CASENUM")["bac_given"].max().rename("bac_given").to_frame()
bac = bac.join(acc.set_index("CASENUM")["sev"])
bacr = bac.groupby("sev")["bac_given"].mean().reindex(SEV_ORDER)
print("\n--- 每 crash 至少一人有 BAC 数值结果的占比（按严重度） ---")
print((bacr * 100).round(2).rename(index=SEV_LABEL).to_string())

# ---------- 步骤4：fatal vs non-fatal 逻辑回归 ----------
# 现场/记录时可得字段（排除 *_IM 插补产物与伤亡结果字段）
feat = acc.copy()
feat = feat[feat.sev.isin(SEV_ORDER)].copy()   # 仅保留有序严重度 0-4（5/6/8/9 剔除）
feat["fatal"] = (feat.sev == 4).astype(int)
print(f"\n[步骤4] 建模样本 {len(feat)} crashes, fatal {feat.fatal.sum()} ({feat.fatal.mean()*100:.2f}%)")

# 车辆级聚合特征（仅 Motor Vehicle In-Transport；该表 2024 年 UNITTYPENAME 单一值）
v = veh.copy()
v["age"] = 2024 - v.MOD_YEAR.where(v.MOD_YEAR.between(1900, 2024))
v["is_moto"] = v.BODY_TYP.between(80, 89).astype(int)
v["is_heavy"] = v.GVWR_TO.ge(3).astype(int)       # GVWR class>=3 (>10,000 lbs)
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
# 缺失类别化：Unknown/Not Reported -> "Missing" 类（保持 valid 类别语义）
for c in cat_cols:
    feat[c] = feat[c].map(classify).where(feat[c].map(classify).eq("valid"), feat[c].astype(str))
    feat[c] = feat[c].replace({"reported_as_unknown": "Unknown", "not_reported": "NotReported"})

X = feat[cat_cols + num_cols]
y = feat["fatal"].values
pre = ColumnTransformer([
    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)])
model = Pipeline([("pre", pre),
                  ("lr", LogisticRegression(max_iter=3000, C=1.0))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(feat))
for tr, te in skf.split(X, y):
    model.fit(X.iloc[tr], y[tr])
    oof[te] = model.predict_proba(X.iloc[te])[:, 1]

auroc = roc_auc_score(y, oof)
ap = average_precision_score(y, oof)
k = int(0.1 * len(feat))
top = np.argsort(-oof)[:k]
tdc = y[top].sum() / y.sum()
print(f"[步骤4] 5折CV OOF: AUROC={auroc:.4f}  AP={ap:.4f}  top-decile coverage={tdc:.4f}")

# ---------- 步骤5：成本扫描 ----------
print("\n===== 步骤5 FN:FP 成本扫描（OOF 概率, 阈值 1/(c+1)） =====")
pi = y.mean()
res = []
for c in range(1, 51):
    thr = 1.0 / (c + 1)
    flag = oof >= thr
    fn = ((flag == 0) & (y == 1)).sum()
    fp = ((flag == 1) & (y == 0)).sum()
    model_cost = (c * fn + fp) / len(y)
    cost_all = 1 - pi          # 全标严重：所有非严重案例误标
    cost_none = c * pi         # 全不标：所有严重案例漏标
    res.append({"c": c, "thr": round(thr, 4), "model": round(model_cost, 5),
                "flag_all": round(cost_all, 5), "flag_none": round(cost_none, 5),
                "beats_both": model_cost < cost_all and model_cost < cost_none})
r = pd.DataFrame(res)
print(r[r.c.isin([1, 2, 3, 5, 10, 15, 20, 30, 40, 50])].to_string(index=False))
win = r[r.beats_both].c
print(f"模型同时优于两个常数规则的成本比窗口: c ∈ [{win.min() if len(win) else '-'}, {win.max() if len(win) else '-'}]" +
      (f" (连续 {len(win)} 个)" if len(win) else ""))
# 无成本阈值下的混淆概况（c=1, thr=0.5）
flag = oof >= 0.5
print(f"c=1 thr=0.5: flagged={flag.sum()}, 其中真实fatal={y[flag].sum()}, 召回={y[flag].sum()/y.sum():.3f}")

# ---------- 保存中间结果 ----------
tab2.to_csv("step2_missing_by_sev.csv", index=False)
r.to_csv("step5_cost_scan.csv", index=False)
np.save("oof.npy", oof)
feat[["CASENUM", "fatal"]].assign(oof=oof).to_csv("oof_predictions.csv", index=False)
summary = {"n_crashes": int(n), "sev_check_agree": float(chk),
           "auroc": float(auroc), "ap": float(ap), "top_decile_cov": float(tdc),
           "prevalence": float(pi),
           "cost_window": [int(win.min()), int(win.max())] if len(win) else None}
json.dump(summary, open("summary.json", "w"), indent=2)
print("\nsaved: step2_missing_by_sev.csv, step5_cost_scan.csv, oof_predictions.csv, summary.json")
