# -*- coding: utf-8 -*-
"""提取 DfT《Contributory factors in fatal collisions: comparing STATS19 with
post-investigation recording, 2021 data》附带数据表 (ras_cfs_final.ods)，
输出整洁 CSV 与中间汇总。"""
import pandas as pd

SRC = "ras_cfs_final.ods"

# ---------- Number_overall: 77 因子 × 两来源记录率 ----------
num = pd.read_excel(SRC, sheet_name="Number_overall", engine="odf", header=2)
num = num.dropna(subset=["Contributory factor code"])
num = num[num["Contributory factor code"].astype(str).str.strip().ne("")]
# 去掉 "All forces with data" 汇总行以外的区域行（本表只有 All forces）
num["STATS19 percentage"] = num["STATS19 percentage"].astype(str).str.rstrip("%").astype(float)
num["Post-investigation percentage"] = (
    num["Post-investigation percentage"].astype(str).str.rstrip("%").astype(float)
)
num["pp_change"] = num["Post-investigation percentage"] - num["STATS19 percentage"]
num = num.rename(columns={
    "Contributory factor code": "code",
    "Contributory factor name": "factor",
    "STATS19 count": "stats19_n",
    "Post-investigation count": "post_n",
    "STATS19 percentage": "stats19_pct",
    "Post-investigation percentage": "post_pct",
})
num = num[["code", "factor", "stats19_n", "post_n", "stats19_pct", "post_pct", "pp_change"]]
num["code"] = num["code"].astype(str).str.strip()
num = num.sort_values("code")
num.to_csv("cf_77_factors_rates.csv", index=False)

# ---------- Change_overall: 逐因子 churn ----------
chg = pd.read_excel(SRC, sheet_name="Change_overall", engine="odf", header=2)
chg = chg.rename(columns={
    "Contributory factor code": "code",
    "Contributory factor name": "factor",
    "Count in STATS19": "n_stats19",
    "of which, remaining post-investigation": "n_both",
    "of which, removed post-investigation": "n_removed",
    "of which, added post investigation": "n_added",
    "Count post-investigation": "n_post",
    "Percentage of STATS19 collisions where factor removed": "pct_removed",
    "Percentage of post-investigation collisions where factor added": "pct_added",
})
chg = chg.dropna(subset=["code"])
chg["code"] = chg["code"].astype(str).str.strip()
chg = chg[chg["code"].ne("")]
for c in ["pct_removed", "pct_added"]:
    chg[c] = pd.to_numeric(chg[c], errors="coerce")  # '[x]' -> NaN
chg.to_csv("cf_77_factors_churn.csv", index=False)

allrow = chg[chg["code"] == "All"]
print("=== All factors row ===")
print(allrow.to_string(index=False))

# ---------- 校验总体数字 ----------
merged = num.merge(
    chg[["code", "n_stats19", "n_both", "n_removed", "n_added", "n_post", "pct_removed", "pct_added"]],
    on="code", how="left", suffixes=("", "_c"))
# 汇总行（All）
print("\n=== checks ===")
print("sum stats19_n (77 factors):", num["stats19_n"].sum())
print("sum post_n   (77 factors):", num["post_n"].sum())

# ---------- Top 10 增减 ----------
top_up = merged.nlargest(10, "pp_change")
top_down = merged.nsmallest(10, "pp_change")
top_up.to_csv("cf_top10_increase.csv", index=False)
top_down.to_csv("cf_top10_decrease.csv", index=False)

print("\n=== TOP 10 increase (pp) ===")
print(top_up[["code", "factor", "stats19_pct", "post_pct", "pp_change"]].to_string(index=False))
print("\n=== TOP 10 decrease (pp) ===")
print(top_down[["code", "factor", "stats19_pct", "post_pct", "pp_change"]].to_string(index=False))

# ---------- Spearman 等级相关（报告正文称 >0.9） ----------
from scipy import stats  # noqa
rho = num["stats19_pct"].rank().corr(num["post_pct"].rank())
print("\nSpearman rank corr (recomputed):", round(rho, 3))
