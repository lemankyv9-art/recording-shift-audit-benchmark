# -*- coding: utf-8 -*-
"""
STATS19 跨年记录制度变化审计（2016 vs 2024）
执行审计协议: 1) 类别先验+多数类基线  2) 字段级记录完整率(-1/unknown)按严重度分组
3) 域分类器(2016 vs 2024)  + 2024 injury-based 与调整后严重度专项统计
数据来源: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
"""
import pandas as pd
import numpy as np
from scipy import stats

OUT = {}

# ---------- 载入 ----------
d24 = pd.read_csv("collision_2024.csv", dtype=str)
d16 = pd.read_csv("collision_2016.csv", dtype=str)
for d in (d16, d24):
    d["sev"] = d["collision_severity"].astype(int)
    d["sev_label"] = d["sev"].map({1: "fatal", 2: "serious", 3: "slight"})

# 统一"路口情境"字段: 2024年新规格把"不在路口"的结构性NA从0改记为-1,
# 需用路口字段把它从"缺失"中剔除, 得到跨年可比的"真实缺失"
def add_junction_context(d, year):
    d = d.copy()
    if year == 2024:
        ib = d["collision_injury_based"] == "1"
        ctx = np.where(ib, d["junction_detail"], d["junction_detail_historic"])
    else:
        ctx = d["junction_detail_historic"]
    d["_at_junction"] = (ctx != "0") & (ctx != "-1")  # 0=不在路口; -1=该情境字段自身缺失
    for col in ["junction_control", "second_road_class"]:
        d[f"_{col}_genuine_missing"] = (d[col] == "-1") & d["_at_junction"]
    return d
d16 = add_junction_context(d16, 2016)
d24 = add_junction_context(d24, 2024)
YEARS = {2016: d16, 2024: d24}  # 必须在增强之后绑定
d24c = None  # 稍后定义为 2024 中 IBRS=0 的可比子集


# ---------- 步骤1: 先验 + 多数类基线 ----------
print("=" * 70)
print("步骤1: 严重度先验分布与多数类基线")
prior_rows = []
for y, d in YEARS.items():
    n = len(d)
    vc = d["sev"].value_counts()
    row = {"year": y, "n": n,
           "fatal%": 100 * vc.get(1, 0) / n,
           "serious%": 100 * vc.get(2, 0) / n,
           "slight%": 100 * vc.get(3, 0) / n,
           "majority_class": "slight",
           "majority_baseline_acc%": 100 * vc.get(3, 0) / n}
    prior_rows.append(row)
    print(row)
prior = pd.DataFrame(prior_rows)
prior.to_csv("out_step1_priors.csv", index=False)

# ---------- 步骤2: -1 / unknown 占比按严重度分组 ----------
# 编码手册(2025 data guide): -1 = Data missing or out of range;
# 9/99 = unknown(self reported); 0 = Not at junction(结构性"不适用")
FIELDS = {
    "officer_attend":  ("did_police_officer_attend_scene_of_accident", {"missing": ["-1"], "unknown": []}),
    "trunk_road":      ("trunk_road_flag", {"missing": ["-1"], "unknown": []}),
    "urban_rural":     ("urban_or_rural_area", {"missing": ["-1"], "unknown": ["3"]}),  # 3=Unallocated
    "road_type":       ("road_type", {"missing": ["-1"], "unknown": ["9"]}),
    "speed_limit":     ("speed_limit", {"missing": ["-1"], "unknown": ["99"]}),
    "junction_detail": ("junction_detail_historic", {"missing": ["-1"], "unknown": ["99"], "na": ["0"]}),
    "junction_control":("junction_control", {"missing": ["-1"], "unknown": ["9"], "na": ["0"]}),
    "second_road_class":("second_road_class", {"missing": ["-1"], "unknown": ["9"], "na": ["0"]}),
    "light_conditions":("light_conditions", {"missing": ["-1"], "unknown": []}),
    "weather":         ("weather_conditions", {"missing": ["-1"], "unknown": ["9"]}),
    "road_surface":    ("road_surface_conditions", {"missing": ["-1"], "unknown": ["9"]}),
    "special_cond":    ("special_conditions_at_site", {"missing": ["-1"], "unknown": ["9"]}),
    "cway_hazards":    ("carriageway_hazards_historic", {"missing": ["-1"], "unknown": ["9"]}),
    "ped_cross_hc":    ("pedestrian_crossing_human_control_historic", {"missing": ["-1"], "unknown": ["9"]}),
    "ped_cross_pf":    ("pedestrian_crossing_physical_facilities_historic", {"missing": ["-1"], "unknown": ["9"]}),
    "first_road_class":("first_road_class", {"missing": ["-1"], "unknown": []}),
    "lsoa":            ("lsoa_of_accident_location", {"missing": ["-1"], "unknown": []}),
    "longitude":       ("longitude", {"missing": ["-1", ""], "unknown": []}),
}
# 跨年可比(剔除结构性NA)的路口字段: 用布尔列, True=情境内真实缺失
HARM_FIELDS = {"junction_control*": "_junction_control_genuine_missing",
               "second_road_class*": "_second_road_class_genuine_missing"}
print("=" * 70)
print("步骤2: 字段记录完整率按严重度分组(2016 / 2024)")
rec_rows = []
for short, (col, codes) in FIELDS.items():
    for y, d in YEARS.items():
        for sev in (1, 2, 3):
            dd = d[d["sev"] == sev]
            m = dd[col].isin(codes["missing"]).mean() * 100
            u = dd[col].isin(codes["unknown"]).mean() * 100 if codes["unknown"] else np.nan
            na = dd[col].isin(codes.get("na", [])).mean() * 100 if codes.get("na") else np.nan
            rec_rows.append({"year": y, "field": short, "sev": {1: "fatal", 2: "serious", 3: "slight"}[sev],
                             "missing%": round(m, 2), "unknown%": round(u, 2) if not np.isnan(u) else "",
                             "not_applicable%": round(na, 2) if not np.isnan(na) else ""})
for short, col in HARM_FIELDS.items():
    for y, d in YEARS.items():
        for sev in (1, 2, 3):
            dd = d[d["sev"] == sev]
            m = dd[col].mean() * 100
            rec_rows.append({"year": y, "field": short, "sev": {1: "fatal", 2: "serious", 3: "slight"}[sev],
                             "missing%": round(m, 2), "unknown%": "", "not_applicable%": ""})
rec = pd.DataFrame(rec_rows)
rec.to_csv("out_step2_recording_completeness.csv", index=False)
# 单调梯度检验: 对每年每字段, missing% 应随严重度上升而下降 (fatal<serious<slight -> 覆盖递增)
# Spearman: 严重度编码1/2/3 与 missing% 应为正相关(越严重编码越小missing越高)
grad_rows = []
for (y, f), g in rec.groupby(["year", "field"]):
    g = g.sort_values("sev")
    miss = g["missing%"].values.astype(float)
    rho, p = stats.spearmanr([1, 2, 3], miss)  # sev编码1=fatal,...; 期望 rho>0
    monotone = bool(miss[0] >= miss[1] >= miss[2])  # fatal>=serious>=slight
    grad_rows.append({"year": y, "field": f, "fatal%": miss[0], "serious%": miss[1], "slight%": miss[2],
                      "spearman_rho": round(rho, 3), "p": round(p, 4), "monotone_decreasing": monotone})
grad = pd.DataFrame(grad_rows)
grad.to_csv("out_step2_gradient_test.csv", index=False)
print(grad.to_string())

# 2024 IBRS 结构性缺测: 旧规格字段在 injury-based 记录上全部 -1
print("\n2024 injury-based 记录占比:", round(100 * (d24["collision_injury_based"] == "1").mean(), 2), "%")
ib = d24["collision_injury_based"] == "1"
for col in ["junction_detail_historic", "pedestrian_crossing_physical_facilities_historic",
            "carriageway_hazards_historic"]:
    print(f"  2024 P({col}=-1 | IBRS=1) =", round((d24.loc[ib, col] == '-1').mean(), 4),
          "| P(=-1 | IBRS=0) =", round((d24.loc[~ib, col] == '-1').mean(), 4))
# 因此 2024 与 2016 可比的子集是 IBRS=0(口径与2016相同)
d24c = d24[~ib].copy()
print("2024 IBRS=0 可比子集 n =", len(d24c))
# 在可比子集上重算 -1 表
rec2_rows = []
for short, (col, codes) in FIELDS.items():
    for sev in (1, 2, 3):
        dd = d24c[d24c["sev"] == sev]
        m = dd[col].isin(codes["missing"]).mean() * 100
        rec2_rows.append({"year": 2024, "field": short, "sev": sev, "missing%": round(m, 2)})
for short, col in HARM_FIELDS.items():
    for sev in (1, 2, 3):
        dd = d24c[d24c["sev"] == sev]
        rec2_rows.append({"year": 2024, "field": short, "sev": sev, "missing%": round(dd[col].mean() * 100, 2)})
rec2 = pd.DataFrame(rec2_rows)
grad2 = []
for f, g in rec2.groupby("field"):
    g = g.sort_values("sev")
    miss = g["missing%"].values.astype(float)
    rho, p = stats.spearmanr([1, 2, 3], miss)
    grad2.append({"year": 2024, "field": f, "fatal%": miss[0], "serious%": miss[1], "slight%": miss[2],
                  "spearman_rho": round(rho, 3), "p": round(p, 4),
                  "monotone_decreasing": bool(miss[0] >= miss[1] >= miss[2])})
grad2 = pd.DataFrame(grad2)
grad2.to_csv("out_step2_gradient_2024IBRS0.csv", index=False)
print(grad2.to_string())

# ---------- 步骤3: 域分类器 2016 vs 2024 ----------
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

NUM = ["number_of_vehicles", "day_of_week"]
CAT = ["police_force", "first_road_class", "road_type", "speed_limit",
       "junction_detail_historic", "junction_control", "second_road_class",
       "pedestrian_crossing_human_control_historic", "pedestrian_crossing_physical_facilities_historic",
       "light_conditions", "weather_conditions", "road_surface_conditions",
       "special_conditions_at_site", "carriageway_hazards_historic",
       "urban_or_rural_area", "did_police_officer_attend_scene_of_accident", "trunk_road_flag"]
# 派生小时/月
def prep(d):
    d = d.copy()
    d["hour"] = d["time"].str.slice(0, 2)
    d["month"] = d["date"].str.slice(3, 5)  # DD/MM/YYYY
    return d
a, b = prep(d16), prep(d24)
X = pd.concat([a[NUM + CAT + ["hour", "month"]], b[NUM + CAT + ["hour", "month"]]], ignore_index=True)
for c in NUM + CAT + ["hour", "month"]:
    X[c] = X[c].astype(str).fillna("NA")
y = np.r_[np.zeros(len(a)), np.ones(len(b))]
pipe = Pipeline([
    ("oh", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT + ["hour", "month"]),
                              ("num", "passthrough", NUM)])),
    ("lr", LogisticRegression(max_iter=2000, C=0.5)),
])
skf = StratifiedKFold(5, shuffle=True, random_state=0)
aucs = []
for tr, te in skf.split(X, y):
    pipe.fit(X.iloc[tr], y[tr])
    aucs.append(roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1]))
print("=" * 70)
print("步骤3: 域分类器 2016 vs 2024 (逻辑回归, 5折CV)")
print("fold AUROC:", [round(x, 4) for x in aucs], "mean =", round(np.mean(aucs), 4))
pd.DataFrame({"fold": range(1, 6), "auroc": aucs}).to_csv("out_step3_domain_auc.csv", index=False)

# ---------- 专项: 2024 injury-based 与调整后严重度 ----------
print("=" * 70)
print("专项: 2024 injury-based 记录与调整后严重度")
n_ib = int(ib.sum()); n_all = len(d24)
adj_s = d24["collision_adjusted_severity_serious"].astype(float)
adj_l = d24["collision_adjusted_severity_slight"].astype(float)
raw_serious_n = int((d24["sev"] == 2).sum())
print(f"2024 n={n_all}, IBRS记录 n={n_ib} ({100*n_ib/n_all:.2f}%)")
print(f"原始 serious: {raw_serious_n} ({100*raw_serious_n/n_all:.2f}%)")
print(f"调整后 serious(=Σadj_serious, fatal行=0): {adj_s.sum():.0f} ({100*adj_s.sum()/n_all:.2f}%)")
print(f"调整后 slight(=Σadj_slight): {adj_l.sum():.0f} ({100*adj_l.sum()/n_all:.2f}%)")
# 若全部按 IBRS 口径: 用调整后; 口径差 = 调整前后 serious 占比差
# 分警队口径: IBRS=1 vs 0 的原始 serious 占比
for grp, mask in [("IBRS=1", ib), ("IBRS=0", ~ib)]:
    print(f"  {grp}: n={mask.sum()}, raw serious% = {100*(d24['sev'][mask]==2).mean():.2f}")
spec = pd.DataFrame({
    "metric": ["n_collisions", "IBRS_n", "IBRS_share%", "raw_serious_n", "raw_serious_share%",
               "adjusted_serious_sum", "adjusted_serious_share%", "raw_serious_share%_IBRS1",
               "raw_serious_share%_IBRS0"],
    "value": [n_all, n_ib, round(100*n_ib/n_all, 2), raw_serious_n, round(100*raw_serious_n/n_all, 2),
              round(adj_s.sum(), 0), round(100*adj_s.sum()/n_all, 2),
              round(100*(d24['sev'][ib]==2).mean(), 2), round(100*(d24['sev'][~ib]==2).mean(), 2)],
})
spec.to_csv("out_ibrs_adjusted_severity.csv", index=False)
print("完成")
