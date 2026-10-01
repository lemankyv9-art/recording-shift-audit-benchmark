"""Audit-extension analyses computed from fixed stored predictions.

Adds four post-release diagnostics without refitting the primary analysis:
  1. a cost-ratio sweep (decision-curve style) for severe-case marking;
  2. a method-by-shift summary matrix from stored per-replicate metrics;
  3. a recording-coverage gradient summary by severity and scenario;
  4. severe-risk calibration of the pooled model and an oracle cost sweep after
     cross-fitted logistic recalibration on the test cohort.

Inputs are the released files under results/ and data/ only. Outputs are
written to results/audit_extension/ and manuscript/figures/decision_curve.pdf.
The c=10 composition-shift pooling row reproduces the published 10:1 decision
metrics (cost 0.767, recall 0.354, precision 0.137, marked 0.233) as a check.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "audit_extension"
OUT.mkdir(exist_ok=True)

SCENARIOS = ["stationary", "covariate_shift", "conditional_shift"]
SWEEP_METHODS = ["logistic_pooling", "tca", "count_only"]
# Evaluate every integer ratio so the reported window does not interpolate
# across unscanned values (the observed-data audits use the same grid).
COST_RATIOS = list(range(1, 51))
REPLICATES = [f"r{i:02d}" for i in range(30)]


def mc_summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    mean = v.mean()
    mcse = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    half = stats.t.ppf(0.975, n - 1) * mcse if n > 1 else np.nan
    return n, mean, mcse, mean - half, mean + half


def cost_sweep():
    rows = []
    for scenario in SCENARIOS:
        cache = {}
        for rep in REPLICATES:
            z = np.load(RESULTS / rep / scenario / "predictions.npz")
            y = z["true_external"] >= 2
            probs = {m: z[f"external__{m}"][:, 2] + z[f"external__{m}"][:, 3] for m in SWEEP_METHODS}
            cache[rep] = (y, probs)
        for method in SWEEP_METHODS:
            for c in COST_RATIOS:
                t = 1.0 / (c + 1.0)
                costs, nones, alls, recalls, precs, marked, nbs, nb_alls, wins = ([] for _ in range(9))
                for rep in REPLICATES:
                    y, probs = cache[rep]
                    p = probs[method]
                    n = len(y)
                    prev = y.mean()
                    mark = p >= t
                    tp = int((mark & y).sum())
                    fp = int((mark & ~y).sum())
                    fn = int((~mark & y).sum())
                    cost = (c * fn + fp) / n
                    cost_none = c * y.sum() / n
                    cost_all = (~y).sum() / n
                    costs.append(cost)
                    nones.append(cost_none)
                    alls.append(cost_all)
                    recalls.append(tp / y.sum() if y.sum() else np.nan)
                    precs.append(tp / (tp + fp) if tp + fp else np.nan)
                    marked.append(mark.mean())
                    nbs.append(tp / n - fp / n * c)
                    nb_alls.append(prev - (1 - prev) * c)
                    wins.append(cost < min(cost_none, cost_all))
                n, mean, mcse, lo, hi = mc_summary(costs)
                rows.append(dict(
                    scenario=scenario, method=method, cost_ratio=c, threshold=round(t, 5),
                    cost_mean=mean, cost_mcse=mcse, cost_low=lo, cost_high=hi,
                    cost_none_mean=np.mean(nones), cost_all_mean=np.mean(alls),
                    beats_both_fraction=np.mean(wins),
                    recall_mean=np.nanmean(recalls), precision_mean=np.nanmean(precs),
                    marked_mean=np.mean(marked),
                    net_benefit_mean=np.mean(nbs), net_benefit_all_mean=np.mean(nb_alls),
                    severe_prevalence_mean=np.mean([cache[r][0].mean() for r in REPLICATES]),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_sweep.csv", index=False)
    check = df.query("scenario=='covariate_shift' and method=='logistic_pooling' and cost_ratio==10").iloc[0]
    print("CHECK c=10 composition pooling: cost %.3f (paper 0.767), none %.3f (0.877), all %.3f (0.912),"
          " recall %.3f (0.354), precision %.3f (0.137), marked %.3f (0.233)" % (
              check.cost_mean, check.cost_none_mean, check.cost_all_mean,
              check.recall_mean, check.precision_mean, check.marked_mean))
    return df


def method_by_shift():
    src = pd.read_csv(RESULTS / "revision_diagnostics" / "threshold_independent_metrics.csv")
    keep = ["accuracy", "balanced_accuracy", "high_major_auroc", "high_major_ap",
            "top10_high_major_coverage", "recall_high", "recall_major"]
    rows = []
    for (scenario, method), g in src.groupby(["scenario", "method"]):
        row = {"scenario": scenario, "method": method, "n": len(g)}
        for m in keep:
            n_, mean, mcse, lo, hi = mc_summary(g[m].to_numpy())
            row[f"{m}_mean"] = mean
            row[f"{m}_low"] = lo
            row[f"{m}_high"] = hi
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "method_by_shift.csv", index=False)
    return df


def recording_gradient():
    src = pd.read_csv(RESULTS / "revision_diagnostics" / "recording_rate_by_severity.csv")
    rows = []
    for (scenario, severity), g in src.groupby(["scenario", "severity"]):
        n, mean, mcse, lo, hi = mc_summary(g["recording_rate"].to_numpy())
        rows.append(dict(scenario=scenario, severity=severity, n=n, rate_mean=mean, rate_low=lo, rate_high=hi))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recording_gradient.csv", index=False)
    return df


def _fit_logit(x, y, offset=None, slope=True, iters=100):
    """Unpenalised Newton-Raphson fit of y ~ a + b*x (+ offset)."""
    X = np.column_stack([np.ones_like(x), x]) if slope else np.ones((len(x), 1))
    off = np.zeros_like(x) if offset is None else offset
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta + off)))
        w = p * (1.0 - p)
        step = np.linalg.solve(X.T @ (X * w[:, None]) + 1e-9 * np.eye(X.shape[1]), X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def recalibration_diagnostic():
    """Severe-risk calibration of the stored pooled probabilities, and the cost
    sweep after a cross-fitted logistic recalibration on the test cohort.

    The recalibration uses independent-test labels (two random halves of each
    cohort, each half recalibrated with the other half), so it is an oracle
    diagnostic that separates ranking from calibration; it is not a deployable
    correction and is not used for any primary result.
    """
    cal_rows, sweep_rows = [], []
    for scenario in SCENARIOS:
        slopes, citls = [], []
        frozen = {c: [] for c in COST_RATIOS}
        recal = {c: [] for c in COST_RATIOS}
        cheaper = {c: [] for c in COST_RATIOS}
        for i, rep in enumerate(REPLICATES):
            z = np.load(RESULTS / rep / scenario / "predictions.npz")
            y = (z["true_external"] >= 2).astype(float)
            p = z["external__logistic_pooling"][:, 2] + z["external__logistic_pooling"][:, 3]
            lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
            slopes.append(_fit_logit(lp, y)[1])
            citls.append(_fit_logit(lp, y, offset=lp, slope=False)[0])
            idx = np.random.default_rng(910000 + i).permutation(len(y))
            half = len(y) // 2
            q = np.empty_like(p)
            for fit_idx, apply_idx in [(idx[:half], idx[half:]), (idx[half:], idx[:half])]:
                a, b = _fit_logit(lp[fit_idx], y[fit_idx])
                q[apply_idx] = 1.0 / (1.0 + np.exp(-(a + b * lp[apply_idx])))
            n = len(y)
            for c in COST_RATIOS:
                t = 1.0 / (c + 1.0)
                for probs, store in [(p, frozen), (q, recal)]:
                    mark = probs >= t
                    fn = int((~mark & (y == 1)).sum())
                    fp = int((mark & (y == 0)).sum())
                    store[c].append((c * fn + fp) / n)
                cheaper[c].append(min(c * y.sum() / n, (y == 0).sum() / n))
        for name, vals in [("calibration_slope", slopes), ("calibration_in_the_large", citls)]:
            n_, mean, mcse, lo, hi = mc_summary(vals)
            cal_rows.append(dict(scenario=scenario, method="logistic_pooling", metric=name,
                                 n=n_, mean=mean, mcse=mcse, low=lo, high=hi))
        for c in COST_RATIOS:
            for label, store in [("source_frozen", frozen), ("cross_fitted_recalibration", recal)]:
                excess = np.asarray(store[c]) - np.asarray(cheaper[c])
                n_, mean, mcse, lo, hi = mc_summary(excess)
                sweep_rows.append(dict(scenario=scenario, method="logistic_pooling", probabilities=label,
                                       cost_ratio=c, excess_cost_mean=mean, excess_cost_low=lo,
                                       excess_cost_high=hi, cheaper_fraction=np.mean(excess < 0)))
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration_by_scenario.csv", index=False)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "recalibration_sweep.csv", index=False)
    return cal, sweep


def decision_curve_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    style = {
        ("stationary", "logistic_pooling"): ("#7f7f7f", "-", "Pooling, stationary"),
        ("covariate_shift", "logistic_pooling"): ("#1f77b4", "-", "Pooling, composition shift"),
        ("conditional_shift", "logistic_pooling"): ("#d62728", "-", "Pooling, injury-mechanism shift"),
        ("covariate_shift", "tca"): ("#1f77b4", "--", "TCA, composition shift"),
        ("covariate_shift", "count_only"): ("#2ca02c", ":", "Factor count, composition shift"),
    }
    for (scenario, method), (color, ls, label) in style.items():
        sub = df.query("scenario==@scenario and method==@method").sort_values("cost_ratio")
        cheaper = np.minimum(sub.cost_none_mean, sub.cost_all_mean)
        ax.plot(sub.cost_ratio, sub.cost_mean - cheaper, color=color, ls=ls, lw=1.6, label=label)
    ax.axhline(0.0, color="black", lw=0.9)
    # Region labels sit on their own side of the zero line, left of the dense
    # 4--7:1 part of the sweep where all curves are still close to zero.
    ax.text(1.05, 0.025, "above zero: a constant rule is cheaper",
            fontsize=8, va="bottom", ha="left")
    ax.text(1.05, -0.05, "below zero: model is cheaper\nthan both constant rules",
            fontsize=8, va="top", ha="left", linespacing=1.2)
    ax.set_xscale("log")
    # The scan is integer-valued, but showing all 50 labels would make the
    # publication figure unreadable; retain representative log-scale ticks.
    display_ticks = [1, 5, 10, 15, 20, 30, 40, 50]
    ax.set_xticks(display_ticks)
    ax.set_xticklabels([str(c) for c in display_ticks])
    ax.set_xlabel("False-negative:false-positive cost ratio (log scale)")
    ax.set_ylabel("Mean cost minus cheaper constant rule")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(ROOT / "manuscript" / "figures" / "decision_curve.pdf")
    print("figure written: manuscript/figures/decision_curve.pdf")


if __name__ == "__main__":
    df = cost_sweep()
    method_by_shift()
    recording_gradient()
    recalibration_diagnostic()
    decision_curve_figure(df)
    print("outputs in results/audit_extension/")
