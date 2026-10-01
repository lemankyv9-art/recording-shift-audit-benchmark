# -*- coding: utf-8 -*-
"""
跨期迁移检验（论文预测③）：记录漂移是否直接降低排序，还是只破坏校准。
2016 <-> 2024 双向迁移，逻辑回归预测 fatal vs 其他：
  (1) 各年内部 5 折 OOF 基准（AUROC / AP / top-decile coverage）
  (2) 2016->2024 迁移：排序指标 + 校准（截距偏移[CITL, slope固定1]、calibration slope、
      log loss[先验平滑 eps=0.01]）
  (3) 2024->2016 反向迁移，同上
  (4) 截距修复模拟：目标年抽 36 / 121 条开发集，30 个种子，仅更新截距、斜率固定 1，
      报告修复后 CITL 与 log loss 变化
字段与缺失约定沿用 ../analysis.py：
  - 现场可得字段 NUM+CAT+派生 hour/month；-1 与空值统一记 "NA"；
  - junction_control / second_road_class 的 0/-1 跨年口径漂移统一记 "NA"（0=不在路口）；
  - one-hot(handle_unknown="ignore", min_frequency=50) + LogisticRegression(C=0.5)。
随机种子固定 seed=42（OOF 折划分）与 seed=1000+s（开发集抽样）。
"""
import pathlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent
EPS = 0.01
SEED_OOF = 42

NUM = ["number_of_vehicles", "day_of_week"]
CAT = ["police_force", "first_road_class", "road_type", "speed_limit",
       "junction_detail_historic", "junction_control", "second_road_class",
       "pedestrian_crossing_human_control_historic", "pedestrian_crossing_physical_facilities_historic",
       "light_conditions", "weather_conditions", "road_surface_conditions",
       "special_conditions_at_site", "carriageway_hazards_historic",
       "urban_or_rural_area", "did_police_officer_attend_scene_of_accident", "trunk_road_flag"]
# 跨年 0<->-1 编码漂移的字段：0(不在路口)与-1(缺失)统一记 NA
HARMONIZE_NA = ["junction_control", "second_road_class"]


def load(year, fname):
    d = pd.read_csv(DATA / fname, dtype=str)
    d["y"] = (d["collision_severity"].astype(int) == 1).astype(int)
    return d


def prep(d):
    d = d.copy()
    for c in NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["hour"] = d["time"].astype(str).str.slice(0, 2)
    d["month"] = d["date"].astype(str).str.slice(3, 5)
    for c in CAT + ["hour", "month"]:
        s = d[c].astype(str)
        s = s.replace({"nan": "NA", "None": "NA", "": "NA", "-1": "NA"})
        if c in HARMONIZE_NA:
            s = s.replace({"0": "NA"})
        d[c] = s
    return d


def make_pipe():
    return Pipeline([
        ("oh", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50),
                                   CAT + ["hour", "month"]),
                                  ("num", "passthrough", NUM)])),
        ("lr", LogisticRegression(max_iter=2000, C=0.5)),
    ])


# ---------- 指标 ----------
def top_decile_coverage(y, p):
    """预测风险前 10% 覆盖的真实严重案例比例（recall@10%）。"""
    k = max(1, int(np.ceil(0.10 * len(y))))
    top = np.argsort(-p)[:k]
    return y[top].sum() / y.sum()


def z_clip(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def fit_intercept_only(y, z):
    """slope 固定 1，拟合截距 b0 使 sigmoid(b0+z) 的 log loss 最小（阻尼 Newton）。
    罕见结局 + 小样本下截距 ML 估计可能发散（开发集 0 事件 -> b0 -> -inf），
    故对概率截断 [1e-9, 1-1e-9]、单步限幅 ±5、截距投影至 [-15, 15]（|logit| 截断 1e-6 时有效界约 ±13.8）。"""
    b = 0.0
    for _ in range(500):
        q = np.clip(expit(b + z), 1e-9, 1 - 1e-9)
        g = q.sum() - y.sum()
        h = np.maximum((q * (1 - q)).sum(), 1e-12)
        step = np.clip(g / h, -5, 5)
        b = float(np.clip(b - step, -15, 15))
        if abs(step) < 1e-10:
            break
    return b


def citl(y, p):
    """calibration-in-the-large：slope 固定 1 时的截距偏移 b0。"""
    return fit_intercept_only(y, z_clip(p))


def cal_slope(y, p):
    """calibration slope：y ~ logit(p) 的逻辑回归斜率。"""
    z = z_clip(p).reshape(-1, 1)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(z, y)
    return float(lr.coef_[0][0])


def sm_logloss(y, p, pi, eps=EPS):
    """加性平滑 log loss：p* = (1-eps)*p + eps*pi，pi 为训练集先验。"""
    return log_loss(y, (1 - eps) * p + eps * pi, labels=[0, 1])


def block_metrics(y, p, pi):
    return {
        "n": len(y), "n_fatal": int(y.sum()),
        "prevalence": y.mean(),
        "auroc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "top_decile_coverage": top_decile_coverage(y, p),
        "citl": citl(y, p),
        "cal_slope": cal_slope(y, p),
        "logloss": sm_logloss(y, p, pi),
    }


# ---------- 载入与预处理 ----------
d16 = prep(load(2016, "collision_2016.csv"))
d24 = prep(load(2024, "collision_2024.csv"))
FEAT = NUM + CAT + ["hour", "month"]
YEARS = {2016: d16, 2024: d24}
print("载入:", {y: (len(d), int(d["y"].sum())) for y, d in YEARS.items()})

# ---------- (1) 各年内部 5 折 OOF 基准 ----------
oof_rows = {}
oof_pred = {}
for y, d in YEARS.items():
    X, yy = d[FEAT], d["y"].values
    pipe = make_pipe()
    p = cross_val_predict(pipe, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=SEED_OOF),
                          method="predict_proba")[:, 1]
    oof_pred[y] = p
    row = block_metrics(yy, p, pi=yy.mean())
    row["year"] = y
    oof_rows[y] = row
    print(f"OOF {y}:", {k: round(v, 5) for k, v in row.items() if isinstance(v, float)})
oof_df = pd.DataFrame(oof_rows).T
oof_df.to_csv(HERE / "out_oof_baseline.csv", index=False)

# ---------- (2)(3) 双向迁移 ----------
transfer_rows = []
models = {}
for src, dst in [(2016, 2024), (2024, 2016)]:
    ds, dd = YEARS[src], YEARS[dst]
    pipe = make_pipe()
    pipe.fit(ds[FEAT], ds["y"])
    models[src] = pipe
    p = pipe.predict_proba(dd[FEAT])[:, 1]
    pi = ds["y"].mean()  # 训练集先验，用于 log loss 平滑
    row = block_metrics(dd["y"].values, p, pi=pi)
    row.update({"direction": f"{src}->{dst}", "train_year": src, "eval_year": dst,
                "train_prior": pi, "auroc_drop_vs_oof": oof_rows[src]["auroc"] - row["auroc"],
                "ap_drop_vs_oof": oof_rows[src]["ap"] - row["ap"],
                "topdec_drop_vs_oof": oof_rows[src]["top_decile_coverage"] - row["top_decile_coverage"]})
    transfer_rows.append(row)
    print(f"迁移 {src}->{dst}:", {k: round(v, 5) for k, v in row.items() if isinstance(v, float)})
tr_df = pd.DataFrame(transfer_rows)
tr_df.to_csv(HERE / "out_transfer_metrics.csv", index=False)

# ---------- (4) 截距修复模拟 ----------
# 目标年抽 n_dev 条开发集，仅更新截距（slope 固定 1），在全目标年上评估修复效果
repair_rows = []
for src, dst in [(2016, 2024), (2024, 2016)]:
    dd = YEARS[dst]
    yy = dd["y"].values
    p = models[src].predict_proba(dd[FEAT])[:, 1]
    z = z_clip(p)
    base = block_metrics(yy, p, pi=YEARS[src]["y"].mean())
    for n_dev in (36, 121):
        for s in range(30):
            rng = np.random.default_rng(1000 + s)
            dev = rng.choice(len(yy), size=n_dev, replace=False)
            b0 = fit_intercept_only(yy[dev], z[dev])
            p_fix = expit(b0 + z)
            fix = block_metrics(yy, p_fix, pi=YEARS[src]["y"].mean())
            repair_rows.append({
                "direction": f"{src}->{dst}", "dev_n": n_dev, "seed": s,
                "dev_fatal": int(yy[dev].sum()),
                "b0_dev": b0,
                "citl_repaired": fix["citl"],
                "logloss_repaired": fix["logloss"],
                "citl_before": base["citl"], "logloss_before": base["logloss"],
            })
rep_df = pd.DataFrame(repair_rows)
rep_df.to_csv(HERE / "out_intercept_repair_seeds.csv", index=False)

summ = (rep_df.groupby(["direction", "dev_n"])
        .agg(dev_fatal_mean=("dev_fatal", "mean"),
             b0_mean=("b0_dev", "mean"), b0_sd=("b0_dev", "std"),
             citl_before=("citl_before", "first"),
             citl_repaired_mean=("citl_repaired", "mean"),
             citl_repaired_sd=("citl_repaired", "std"),
             citl_repaired_median=("citl_repaired", "median"),
             logloss_before=("logloss_before", "first"),
             logloss_repaired_mean=("logloss_repaired", "mean"),
             logloss_repaired_sd=("logloss_repaired", "std"),
             logloss_repaired_median=("logloss_repaired", "median"),
             seeds_logloss_improved=("logloss_repaired",
                                     lambda s: float((s < rep_df.loc[s.index, "logloss_before"]).mean())),
             seeds_citl_improved=("citl_repaired",
                                  lambda s: float((s.abs() < rep_df.loc[s.index, "citl_before"].abs()).mean())))
        .reset_index())
summ["citl_gain"] = summ["citl_repaired_mean"].abs() - summ["citl_before"].abs()
summ["logloss_gain"] = summ["logloss_before"] - summ["logloss_repaired_mean"]
summ.to_csv(HERE / "out_intercept_repair_summary.csv", index=False)
print(summ.to_string(index=False))
print("完成")
