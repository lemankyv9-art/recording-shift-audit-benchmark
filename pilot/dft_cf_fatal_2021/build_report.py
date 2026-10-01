# -*- coding: utf-8 -*-
"""组装 report.md（中文实证报告）。"""
import os

import pandas as pd

d = os.path.dirname(os.path.abspath(__file__))
num = pd.read_csv(f"{d}/cf_77_factors_rates.csv", dtype={"code": str})
chg = pd.read_csv(f"{d}/cf_77_factors_churn.csv", dtype={"code": str})
allrow = chg[chg["code"] == "All"].iloc[0]

up = num.nlargest(10, "pp_change").copy()
down = num.nsmallest(10, "pp_change").copy()
top20 = pd.concat([up, down])
top20["方向"] = ["↑ 低估"] * 10 + ["↓ 高估"] * 10

m = num.merge(
    chg[chg["code"] != "All"][
        ["code", "n_stats19", "n_both", "n_removed", "n_added", "n_post", "pct_removed", "pct_added"]
    ],
    on="code", how="left",
)
m["pct_removed"] = (m["pct_removed"] * 100).round(1)
m["pct_added"] = (m["pct_added"] * 100).round(1)
added = m.nlargest(10, "pct_added")[["code", "factor", "n_post", "pct_added"]]
removed = m.nlargest(10, "pct_removed")[["code", "factor", "n_stats19", "pct_removed"]]

full = num.copy()
full.columns = ["代码", "因子", "STATS19 次数", "调查后次数", "STATS19 %", "调查后 %", "Δpp"]
full = full.drop(columns=["STATS19 次数", "调查后次数"])

up_n = int((num["pp_change"] > 0).sum())
down_n = int((num["pp_change"] < 0).sum())
same_n = int((num["pp_change"] == 0).sum())

full_md = full.to_markdown(index=False)
top20_md = (top20[["code", "factor", "stats19_pct", "post_pct", "pp_change", "方向"]]
            .rename(columns={"code": "代码", "factor": "因子", "stats19_pct": "STATS19 %",
                             "post_pct": "调查后 %", "pp_change": "Δpp"})
            .to_markdown(index=False))
added_md = (added.rename(columns={"factor": "因子", "n_post": "调查后次数", "pct_added": "新增占比 %"})
            .to_markdown(index=False))
removed_md = (removed.rename(columns={"factor": "因子", "n_stats19": "STATS19 次数", "pct_removed": "移除占比 %"})
              .to_markdown(index=False))

report = f"""# 实证报告：致命碰撞贡献因子——STATS19 初始记录 vs 深入调查后记录（英国 DfT，2021 数据）

## 1. 数据来源与方法

- 来源：英国交通部（DfT）官方统计《Contributory factors in fatal collisions: comparing STATS19 with post-investigation recording, 2021 data》（2023-11-29 发布）。
- 方法一句话：对 2021 年 1,053 起有警员到场且记录了至少 1 个贡献因子（CF）的致命碰撞，将 STATS19 系统中警方**现场初始判断**记录的贡献因子，与同批碰撞在**深入调查完成后**（事发至少 18 个月后，经 NPCC 向 38 个警队征集）重新记录的因子逐因子对比。
- 数据文件：GOV.UK 页面正文（HTML）+ 官方附带数据表 `ras_cfs_final.ods`（已存本目录）。正文 4 张表与 ODS 两个工作表（`Number_overall`、`Change_overall`）已交叉校验。
- 本任务为文献/官方数据提取，无微观数据，**通用清单中的建模步骤（类别先验、域分类器、严重度分类器、成本扫描）不适用**，原因见第 8 节。

## 2. 总体核心数字（官方口径，已与 ODS 核对）

| 指标 | 数值 |
|---|---|
| 覆盖碰撞数 | 1,053 起（2021 年范围内 1,271 起有 CF 记录的致命碰撞的多数） |
| 响应警队 | 38 支（未全覆盖，故与已发布全国 CF 统计略有出入） |
| 每起碰撞平均因子数 | STATS19 2.3 个 vs 调查后 2.6 个 |
| 因子总次数 | STATS19 {int(allrow['n_stats19'])} 次 → 调查后 {int(allrow['n_post'])} 次（净增 +9.1%） |
| 两来源同时记录 | {int(allrow['n_both'])} 次 |
| **初始记录中被移除/更改** | **880 次，占 {allrow['pct_removed']*100:.1f}%** |
| **调查后记录中为新增** | **1,105 次，占 {allrow['pct_added']*100:.1f}%** |
| 因子排名 Spearman 相关 | 0.965（官方称 >0.9，本站复算一致） |

> 解读：排名高度稳定（头部因子大同小异），但**记录层面 churn 极大**——超过 1/3 的初始因子在深入调查后被撤换，超过 2/5 的最终因子是新增的。聚合统计的"稳定"掩盖了微观层面的系统性记录误差。

## 3. 全部因子：STATS19 记录率 vs 调查后记录率（官方数据表，n=78 行）

官方 ODS 的 `Number_overall` 表共 78 行（正文称"77 个因子"；实际列出 78 个代码，其中 105/110/205 三个因子未出现在 `Change_overall` 表，见第 9 节）。Δpp = 调查后% − STATS19%。

{full_md}

**汇总：因子层面 42 个上升、22 个下降、14 个不变；平均绝对变化 0.61pp，中位数 +0.1pp。** 上升面更广，但头部变化集中。

## 4. 变化最大的前 10 个因子（按 |Δpp| 方向分列）

{top20_md}

官方正文仅给出各前 5；此处前 10 由完整数据表计算，与正文前 5 完全一致。

## 5. 记录 churn 最大的因子（逐碰撞层面）

**调查后新增占比最高**（该因子在调查后记录中的碰撞，原先 STATS19 未编码的比例——"初始记录漏记率"）：

{added_md}

**初始记录被移除占比最高**（该因子在 STATS19 记录中的碰撞，调查后被撤下的比例——"初始记录误记率"）：

{removed_md}

正文要点（官方表 4 与正文）：超速因子 199→257 起，仅 160 起两来源一致（39 起被移除、97 起新增）；`travelling too fast for conditions` 54.1%、`impaired by drugs` 60.4%、`mobile phone use` 70.8%（小样本）为新增占比最高；`aggressive driving` 54.9%、`careless, reckless or in a hurry` 48.4%、`Other` 60.9% 为移除占比最高。

## 6. "Fatal 4" 相关因子：官方汇总

| 因子组 | STATS19 | 调查后 |
|---|---|---|
| 超速（306+307） | 26% | 35%（成为最普遍单一因子） |
| 酒驾/毒驾（501+502） | 15% | 20% |
| 开车用手机（508） | 1% | 2% |
| 上述任一项 | 35% | 45% |

## 7. 各警队差异（官方描述性证据）

官方数据表未含分警队明细表，正文为定性描述，要点：

- **大都会警察（Metropolitan Police）**：调查后记录与 STATS19 几乎无差异——原因已查明：**Met 录入 STATS19 的数据本身已用事后信息更新过**（即其"初始记录"并非真正的现场初始记录）。
- **部分警队**：STATS19 中编码的因子有超过一半在调查后被移除/更改，churn 极大。
- 官方结论：各警队 CF 记录方式存在系统性不一致，值得进一步核查；并拟将此类工作扩展为 NPCC 牵头的长期 post-investigation 数据收集。

## 8. 与论文主张的定性对照

| 论文主张 | 本数据集证据 | 方向是否一致 |
|---|---|---|
| (a) informative recording：调查越深、记录越多越准 | **直接官方证据**。需法医/检验结果才能确认的因子（毒驾 +4.3pp、新增占比 60.4%；酒驾 +2.8pp；超速 +5.7pp；手机 +1.0pp、新增占比 70.8%）在初始记录中被系统性低估；现场即可判断的"驾驶失误"类因子（未认真观察 −4.9pp、误判路径 −1.5pp、分心 −1.0pp）在初始记录中被相对高估。78 个因子中 42 升 22 降。 | **一致** |
| (b) recording shift（制度/机构间差异） | **一致**。同一国家、同一年度、同一因子表，只因警队操作不同（Met 事后更新 vs 其他警队现场判断），记录机制即产生系统性差异。 | **一致** |
| (c) 需先审计记录机制再解读指标 | 官方报告本身的行文即遵循此逻辑：先报告记录制度差异，再给出因子排序（Spearman 0.965 的"稳定"）与记录 churn（35.5%/40.9%）并指出二者不能互相替代。 | **一致** |

对论文的补充价值：这是"记录机制混淆因子测量"的**机制层官方证据**——它发生在单一严重度（fatal）内部，说明 informative recording 不仅是"严重度→字段覆盖率"的梯度，更是"调查深度→因子检出率"的梯度；且警队层面存在 recording shift。

## 9. 局限性与数据质量说明

- **无微观数据**：只有因子层面的汇总计数（碰撞级联表不公开），通用清单中"类别先验+多数基线、域分类器、严重度分类器、成本扫描"均无法执行。本任务按任务书要求只做文献提取。
- **样本**：2021 年数据受疫情封锁影响；38/43 支警队响应，非全样本。
- **测量口径差异**：post-investigation 数据由 NPCC 中央表格按**碰撞级**收集（不关联当事人），STATS19 按当事人级收集，本身会带来差异；且请求在事发后 18+ 个月进行。
- **两工作表间的小不一致**（已逐行核对并保留原样）：`Number_overall` 含 78 行（官方口径 77 因子，多出 3 个 code 105/110/205 且 `Change_overall` 未含这三项，只覆盖 75 因子）；两表 12 个因子的计数有 ±1~6 的小差异（如 306 超速：197 vs 199），官方正文采用 `Change_overall` 口径（199）。百分比两表一致。
- **缺失/不适用编码**：本数据集为计数型官方汇总表，无个体级缺失编码，"缺失 vs 不适用"问题不适用。
- 警队差异仅为定性描述，无法量化各警队梯度。

## 附：文件清单

- `page.html` / `page_text.txt`：GOV.UK 报告正文（抓取于 2026-09-30）
- `parent.html`：统计发布主页（含附件链接）
- `ras_cfs_final.ods`：官方附带数据表原件
- `extract.py`：提取与校验脚本（python3 + pandas/odfpy）
- `cf_77_factors_rates.csv`：全因子两来源记录率（自 `Number_overall`）
- `cf_77_factors_churn.csv`：全因子 churn（自 `Change_overall`）
- `cf_merged_all.csv`：两表合并视图
- `cf_top10_increase.csv` / `cf_top10_decrease.csv`：变化最大因子
- `table_all_factors.md` / `table_top10.md` / `table_top10_added_share.md` / `table_top10_removed_share.md`：本报告所用中间表
"""

with open(f"{d}/report.md", "w", encoding="utf-8") as f:
    f.write(report)
print("report.md written,", len(report), "chars")
