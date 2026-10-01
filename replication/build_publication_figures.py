"""Build the manuscript's publication figures from fixed result files.

The script reads released CSV/JSON summaries only. It does not generate
records, refit models, select parameters, or change any observations. All
outputs are PDF vector graphics so labels remain sharp at print scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision_diagnostics"
OUT = ROOT / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe--Ito inspired colours with redundant marker coding where possible.
BLUE = "#2166AC"
ORANGE = "#D55E00"
TEAL = "#009E73"
CORAL = "#CC6677"
PURPLE = "#6A3D9A"
INK = "#303030"
MID = "#666666"
GRID = "#D9D9D9"
PALE_BLUE = "#EAF2F8"
PALE_ORANGE = "#FDF2E9"
PALE_CORAL = "#FDEDEC"
PALE_TEAL = "#E8F8F5"
PALE_PURPLE = "#F4ECF7"

mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(
        OUT / name,
        format="pdf",
        facecolor="white",
        metadata={"Creator": "Matplotlib", "Title": name.removesuffix(".pdf")},
    )
    plt.close(fig)


def style_axes(ax, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55)
    ax.set_axisbelow(True)


def mean_ci(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, mean, mean
    # t_{29,.975}; all released summaries use 30 paired replicates.
    err = 2.045229642 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
    return mean, mean - err, mean + err


def workflow() -> None:
    """Draw a clean causal/recording schematic with separated paths."""
    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.65)
    ax.axis("off")

    def box(x, y, w, h, label, face, edge, *, fontsize=9.2):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.09",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
            linespacing=1.15,
        )
        return x, y, w, h

    def arrow(start, end, *, dashed=False, colour=MID, rad=0.0, lw=1.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=lw,
                linestyle="--" if dashed else "-",
                color=colour,
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=3,
                shrinkB=3,
            )
        )

    x = box(0.30, 4.65, 2.20, 0.98, "Latent/context\nstate $X$", PALE_BLUE, BLUE)
    h = box(3.05, 4.65, 2.20, 0.98, "Exposure and\nhazard mechanisms", PALE_ORANGE, ORANGE)
    y = box(5.80, 4.65, 2.42, 0.98, "Casualties and losses\nthen endpoint $Y$", PALE_CORAL, CORAL)
    m = box(3.05, 2.28, 2.20, 0.98, "Assessment mask\n$M$", PALE_TEAL, TEAL)
    r = box(6.00, 2.28, 2.20, 0.98, "Recorded factors\n$R=X\\odot M$", PALE_PURPLE, PURPLE)

    arrow((x[0] + x[2], x[1] + x[3] / 2), (h[0], h[1] + h[3] / 2))
    arrow((h[0] + h[2], h[1] + h[3] / 2), (y[0], y[1] + y[3] / 2))
    # Direct latent contribution to the endpoint, separated above the boxes.
    arrow((x[0] + x[2] * 0.82, x[1] + x[3]), (y[0] + y[2] * 0.18, y[1] + y[3]), rad=-0.36, lw=0.9)
    ax.text(4.22, 6.34, "latent contribution", ha="center", va="center", fontsize=7.2, color=MID)
    arrow((x[0] + x[2] / 2, x[1]), (m[0] + m[2] / 2, m[1] + m[3]), rad=0.04)
    arrow((h[0] + h[2] / 2, h[1]), (m[0] + m[2] / 2, m[1] + m[3]), rad=-0.04)
    arrow((m[0] + m[2], m[1] + m[3] / 2), (r[0], r[1] + r[3] / 2))
    # The elementwise definition in the recorded-factors box already shows
    # that R is formed from X and M.  A second diagonal X-to-R arrow crossed
    # the dashed intervention path at print size, so it is intentionally
    # omitted; the unambiguous M-to-R arrow and equation carry that relation.
    arrow((y[0] + y[2] * 0.78, y[1]), (m[0] + m[2] * 0.90, m[1] + m[3]), dashed=True, rad=0.16, lw=1.05)
    ax.text(
        8.62,
        4.12,
        "severity-linked recording\nintervention only",
        ha="center",
        va="center",
        fontsize=7.2,
        color=MID,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    ax.text(5.05, 0.78, "Primary predictors: $R$", ha="center", va="center", fontsize=8.7, color=PURPLE, weight="bold")
    ax.text(5.05, 0.38, "Outcome fields and diagnostic masks are excluded from primary model fitting.", ha="center", va="center", fontsize=7.8, color=MID)
    save(fig, "workflow.pdf")


def primary_results() -> None:
    """Plot primary threshold-independent metrics from the released matrices."""
    df = pd.read_csv(RESULTS / "threshold_independent_metrics.csv")
    wanted = ["majority", "logistic_pooling", "tca", "pca", "count_only", "context_augmented"]
    labels = {
        "majority": "Prior",
        "logistic_pooling": "Pooling",
        "pca": "PCA",
        "tca": "TCA",
        "count_only": "Factor count",
        "context_augmented": "Context diagnostic",
    }
    colors = {"majority": "#6B6B6B", "logistic_pooling": BLUE, "pca": TEAL, "tca": PURPLE, "count_only": ORANGE, "context_augmented": CORAL}
    markers = {"majority": "o", "logistic_pooling": "o", "pca": "s", "tca": "D", "count_only": "^", "context_augmented": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.55), sharey=True)
    panels = [
        ("high_major_auroc", "Severe-case ranking", "High + major AUROC", (0.46, 0.90), False),
        ("top10_high_major_coverage", "Risk concentration", "Top-decile coverage", (0.05, 0.52), True),
    ]
    y = np.arange(len(wanted))
    for panel_index, (metric, title, xlabel, limits, percent) in enumerate(panels):
        ax = axes[panel_index]
        sub = df[(df.scenario == "covariate_shift") & df.method.isin(wanted)]
        rows = []
        for method in wanted:
            vals = sub.loc[sub.method == method, metric].to_numpy(dtype=float)
            mean, low, high = mean_ci(vals)
            rows.append((method, mean, low, high))
        for yi, (method, mean, low, high) in zip(y, rows):
            ax.errorbar(
                mean, yi, xerr=[[mean - low], [high - mean]], fmt=markers[method],
                color=colors[method], markerfacecolor=colors[method], markeredgecolor="white",
                markeredgewidth=0.65, markersize=5.6, ecolor=colors[method],
                elinewidth=1.1, capsize=2.5, capthick=0.9, zorder=3,
            )
        ax.set_title(f"{chr(65 + panel_index)}. {title}", pad=9)
        ax.set_xlabel(xlabel)
        ax.set_xlim(*limits)
        ax.set_yticks(y)
        if panel_index == 0:
            ax.set_yticklabels([labels[m] for m in wanted])
        else:
            ax.tick_params(axis="y", labelleft=False)
        if panel_index == 0:
            ax.invert_yaxis()
        style_axes(ax, grid_axis="x")
        if percent:
            ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        else:
            ax.axvline(0.5, color="#8C8C8C", lw=0.8, ls=(0, (3, 2)), zorder=0)
    fig.text(0.5, 0.015, "Points are means; whiskers are 95% Monte Carlo intervals across 30 paired replicates.", ha="center", fontsize=7.4, color=MID)
    fig.subplots_adjust(wspace=0.10, left=0.17, right=0.985, bottom=0.19, top=0.88)
    save(fig, "primary_results.pdf")


def distributions() -> None:
    """Show generated class proportions and the reported-loss distribution."""
    frames = []
    for i in range(30):
        path = ROOT / "data" / "datasets" / f"r{i:02}" / "covariate_shift" / "records.csv"
        frames.append(pd.read_csv(path, usecols=["severity", "economic loss"]))
    records = pd.concat(frames, ignore_index=True)
    order = ["low", "medium", "high", "major"]
    props = records.severity.value_counts(normalize=True).reindex(order)
    losses = np.sort(np.log1p(records["economic loss"].to_numpy(dtype=float)))
    ecdf = np.arange(1, len(losses) + 1) / len(losses)
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), gridspec_kw={"width_ratios": [0.88, 1.12]})
    ax = axes[0]
    bars = ax.bar(order, props.to_numpy(), color=[PALE_BLUE, PALE_ORANGE, PALE_CORAL, PALE_PURPLE], edgecolor=[BLUE, ORANGE, CORAL, PURPLE], linewidth=0.9)
    for bar, value in zip(bars, props.to_numpy()):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.012, f"{value:.2f}", ha="center", va="bottom", fontsize=7.7)
    ax.set_ylim(0, 0.72); ax.set_ylabel("Proportion"); ax.set_title("A. Generated class proportions", pad=9)
    style_axes(ax, grid_axis="y")
    ax = axes[1]
    ax.step(losses, ecdf, where="post", color=BLUE, linewidth=1.45)
    ax.set_xlabel(r"Log(1 + loss / one million CNY)"); ax.set_ylabel("Empirical CDF"); ax.set_title("B. Reported-loss distribution", pad=9)
    ax.set_ylim(0, 1.02); style_axes(ax, grid_axis="y")
    fig.text(0.5, 0.015, "Thirty source-plus-target replicates are pooled for descriptive display; values are generator properties.", ha="center", fontsize=7.2, color=MID)
    fig.subplots_adjust(left=0.09, right=0.99, wspace=0.28, bottom=0.20, top=0.88)
    save(fig, "distributions.pdf")


def paired_differences() -> None:
    """Plot paired TCA-minus-pooling differences from released summaries."""
    df = pd.read_csv(ROOT / "results" / "paired_differences.csv")
    df = df[(df.method == "tca") & (df.reference == "logistic_pooling") & (df.cohort == "external") & df.metric.isin(["accuracy", "brier"])]
    scenarios = ["stationary", "covariate_shift", "conditional_shift"]
    names = {"stationary": "Stationary", "covariate_shift": "Composition", "conditional_shift": "Injury mechanism"}
    # Accuracy is shown in percentage points to match the text; Brier stays on its score scale.
    panels = [
        ("accuracy", "A. Accuracy difference", "TCA minus pooling (percentage points)", 100.0),
        ("brier", "B. Brier difference", "TCA minus pooling (Brier score)", 1.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.15), sharex=True)
    for ax, (metric, title, ylabel, scale) in zip(axes, panels):
        q = df[df.metric == metric].set_index("scenario").reindex(scenarios)
        x = np.arange(len(scenarios)); vals = scale * q.mean_difference.to_numpy(dtype=float)
        lows = scale * q.low.to_numpy(dtype=float); highs = scale * q.high.to_numpy(dtype=float)
        ax.errorbar(x, vals, yerr=[vals - lows, highs - vals], fmt="o", color=PURPLE, markerfacecolor=PURPLE,
                    markeredgecolor="white", markeredgewidth=0.7, markersize=6, ecolor=PURPLE,
                    elinewidth=1.15, capsize=2.8, capthick=0.9)
        ax.axhline(0, color=INK, lw=0.85)
        ax.set_title(title, pad=9); ax.set_xticks(x); ax.set_xticklabels([names[s] for s in scenarios], rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        style_axes(ax, grid_axis="y")
    fig.text(0.5, 0.015, "Points are paired means; whiskers are 95% Monte Carlo intervals across 30 replicates.", ha="center", fontsize=7.2, color=MID)
    fig.subplots_adjust(left=0.10, right=0.99, wspace=0.30, bottom=0.25, top=0.88)
    save(fig, "paired.pdf")

def factorial_effects() -> None:
    df = pd.read_csv(RESULTS / "factorial_effects.csv")
    order = [
        ("signal", "3.0-1.0", "Signal: 3 vs 1"),
        ("shift", "2.0-0.0", "Composition shift: 2 vs 0"),
        ("recording", "severity-complexity", "Severity-linked recording"),
        ("recording", "drift-complexity", "Period drift"),
        ("development_n", "121-36", "Development size: 121 vs 36"),
        ("fatality_intercept", "0.65-0.0", "Fatality intercept"),
    ][::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 4.45), sharey=True)
    metric_info = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("severe_auc", "Severe AUROC"),
        ("cost_recall", "Cost-sensitive recall"),
    ]
    for panel_index, (ax, (metric, title)) in enumerate(zip(axes, metric_info)):
        rows = []
        for factor, contrast, label in order:
            q = df[(df.method == "pooled_fixed") & (df.factor == factor) & (df.contrast == contrast) & (df.metric == metric)]
            if q.empty:
                continue
            row = q.iloc[0]
            rows.append((label, float(row["mean"]), float(row["low"]), float(row["high"])))
        y = np.arange(len(rows))
        for yi, (label, value, low, high) in zip(y, rows):
            colour = ORANGE if "Severity-linked" in label else BLUE
            ax.errorbar(value, yi, xerr=[[value - low], [high - value]], fmt="o", color=colour,
                        markeredgecolor="white", markeredgewidth=0.6, markersize=5.7,
                        ecolor=colour, elinewidth=1.15, capsize=2.4, capthick=0.9, zorder=3)
        ax.axvline(0.0, color=INK, lw=0.85, zorder=1)
        ax.set_title(f"{chr(65 + panel_index)}. {title}", pad=9)
        ax.set_xlabel("Absolute paired change")
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.3g"))
        ax.set_yticks(y)
        if panel_index == 0:
            ax.set_yticklabels([row[0] for row in rows], fontsize=7.4)
        else:
            ax.tick_params(axis="y", labelleft=False)
        if panel_index == 0:
            ax.invert_yaxis()
        style_axes(ax, grid_axis="x")
    fig.text(0.5, 0.015, "Whiskers show 95% Monte Carlo intervals; contrasts are averaged over the other crossed factors.", ha="center", fontsize=7.4, color=MID)
    fig.subplots_adjust(left=0.29, right=0.99, wspace=0.21, bottom=0.17, top=0.88)
    save(fig, "factorial_effects.pdf")


def recall_calibration() -> None:
    raw = pd.read_csv(RESULTS / "selection_metrics.csv")
    methods = ["pool_fixed", "tca_fixed"]
    names = {"pool_fixed": "Pooling", "tca_fixed": "TCA"}
    colours = {"pool_fixed": BLUE, "tca_fixed": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.50), gridspec_kw={"width_ratios": [1.12, 1.0]})

    categories = [("recall_low", "Low"), ("recall_medium", "Medium"), ("recall_high", "High"), ("recall_major", "Major")]
    x = np.arange(len(categories))
    for method in methods:
        means, lows, highs = [], [], []
        subset = raw[(raw.scenario == "covariate_shift") & (raw.method == method)]
        for metric, _ in categories:
            mean, low, high = mean_ci(subset[metric])
            means.append(mean); lows.append(low); highs.append(high)
        axes[0].errorbar(x + (-0.10 if method == "pool_fixed" else 0.10), means,
                         yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
                         fmt="o", ms=5.7, capsize=2.5, capthick=0.9, lw=1.05,
                         color=colours[method], markeredgecolor="white", markeredgewidth=0.6,
                         label=names[method])
    axes[0].set_title("A. Argmax class recall", pad=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([label for _, label in categories])
    axes[0].set_ylim(0, 1.03); axes[0].set_ylabel("Recall")
    axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].legend(frameon=False, loc="upper right")
    style_axes(axes[0], grid_axis="y")

    cal = raw[(raw.scenario == "covariate_shift") & raw.method.isin(methods)].groupby("method").agg(
        citl=("calibration_citl", "mean"), slope=("calibration_slope", "mean")
    ).reindex(methods)
    cal_err = raw[(raw.scenario == "covariate_shift") & raw.method.isin(methods)].groupby("method").agg(
        citl=("calibration_citl", lambda x: 2.045229642 * x.std(ddof=1) / np.sqrt(len(x))),
        slope=("calibration_slope", lambda x: 2.045229642 * x.std(ddof=1) / np.sqrt(len(x))),
    ).reindex(methods)
    x2 = np.arange(2)
    axes[1].errorbar(x2 - 0.11, cal.citl, yerr=cal_err.citl, fmt="o", color=BLUE, ms=5.7, capsize=2.5, capthick=0.9, label="CITL")
    axes[1].errorbar(x2 + 0.11, cal.slope, yerr=cal_err.slope, fmt="s", color=ORANGE, ms=5.7, capsize=2.5, capthick=0.9, label="Slope")
    axes[1].axhline(0, color=BLUE, lw=0.7, ls=(0, (3, 2)))
    axes[1].axhline(1, color=ORANGE, lw=0.7, ls=(0, (2, 2)))
    axes[1].set_title("B. Calibration diagnostics", pad=9)
    axes[1].set_xticks(x2); axes[1].set_xticklabels([names[m] for m in methods])
    axes[1].set_ylabel("Estimate"); axes[1].set_ylim(-0.10, 1.86)
    axes[1].legend(frameon=False, loc="upper left")
    style_axes(axes[1], grid_axis="y")
    fig.text(0.5, 0.015, "Points are means; whiskers are 95% Monte Carlo intervals across 30 replicates.", ha="center", fontsize=7.4, color=MID)
    fig.subplots_adjust(left=0.10, right=0.985, wspace=0.29, bottom=0.18, top=0.88)
    save(fig, "recall_calibration.pdf")


def learning_curves() -> None:
    df = pd.read_csv(RESULTS / "learning_metrics.csv")
    df = df[df.scenario == "covariate_shift"]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.50))
    colours = {"target": ORANGE, "pool": BLUE}
    names = {"target": "Target only", "pool": "Pooling"}
    for panel_index, (metric, title, ylabel, ylim) in enumerate([
        ("severe_auc", "A. Severe-case ranking", "AUROC", (0.45, 0.70)),
        ("log_loss", "B. Smoothed probability error", "Log loss", (0.85, 1.50)),
    ]):
        ax = axes[panel_index]
        for arm in ["target", "pool"]:
            means, lows, highs, ns = [], [], [], []
            for n, group in df[df.arm == arm].groupby("n", sort=True):
                mean, low, high = mean_ci(group[metric])
                ns.append(n); means.append(mean); lows.append(low); highs.append(high)
            ns = np.asarray(ns)
            ax.plot(ns, means, marker="o", ms=4.8, lw=1.35, color=colours[arm], label=names[arm])
            ax.fill_between(ns, lows, highs, color=colours[arm], alpha=0.14, linewidth=0)
        ax.set_title(title, pad=9); ax.set_xlabel("Target training rows"); ax.set_ylabel(ylabel)
        ax.set_xticks([12, 24, 36, 72, 121]); ax.set_ylim(*ylim)
        style_axes(ax, grid_axis="y")
        if panel_index == 0:
            ax.legend(frameon=False, loc="lower right")
    fig.text(0.5, 0.015, "Lines show means; shaded bands are 95% Monte Carlo intervals across 30 replicates.", ha="center", fontsize=7.4, color=MID)
    fig.subplots_adjust(left=0.10, right=0.985, wspace=0.27, bottom=0.18, top=0.88)
    save(fig, "learning_curves.pdf")


if __name__ == "__main__":
    workflow()
    primary_results()
    factorial_effects()
    recall_calibration()
    learning_curves()
    distributions()
    paired_differences()
    print("Wrote seven publication-quality vector figures to", OUT)
