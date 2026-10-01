"""Threshold-independent and recording-process diagnostics for v3.2.

This script reads only the released fixed matrices and stored predictions.  It
does not regenerate data or refit the primary protocol.  The outputs separate
ranking, calibration and hard-label decisions and quantify the recording-rate
and domain-shift diagnostics discussed in the manuscript revision.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "datasets"
RESULTS = ROOT / "results"
OUT = RESULTS / "revision_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
LABELS = ["low", "medium", "high", "major"]
LABEL_MAP = {x: i for i, x in enumerate(LABELS)}


def binary_code(frame):
    weights = (1 << np.arange(frame.shape[1], dtype=np.uint64))
    return (frame.to_numpy(np.uint8) @ weights).astype(np.uint64)


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def safe_ap(y, score):
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def score_row(y, p):
    pred = p.argmax(axis=1)
    cm = np.zeros((4, 4), dtype=int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    recall = [cm[k, k] / cm[k].sum() if cm[k].sum() else np.nan for k in range(4)]
    precision = [cm[k, k] / cm[:, k].sum() if cm[:, k].sum() else np.nan for k in range(4)]
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.nanmean(recall)),
        "recall_low": float(recall[0]),
        "recall_medium": float(recall[1]),
        "recall_high": float(recall[2]),
        "recall_major": float(recall[3]),
        "precision_high": float(precision[2]) if np.isfinite(precision[2]) else np.nan,
        "precision_major": float(precision[3]) if np.isfinite(precision[3]) else np.nan,
    }


def main():
    metric_rows = []
    recording_rows = []
    domain_rows = []
    selection_rows = []
    novelty_rows = []
    for rep in range(30):
        for scenario in ["stationary", "covariate_shift", "conditional_shift"]:
            d = DATA / f"r{rep:02}" / scenario
            source = pd.read_csv(d / "source.csv")
            external = pd.read_csv(d / "external_test.csv")
            mask = pd.read_csv(d / "external_assessment_mask.csv")
            y = external.severity.map(LABEL_MAP).to_numpy()
            severe = (y >= 2).astype(int)
            prior = np.bincount(y, minlength=4) / len(y)
            prior_p = np.tile(prior, (len(y), 1))
            stored = np.load(RESULTS / f"r{rep:02}" / scenario / "predictions.npz")
            result = json.loads((RESULTS / f"r{rep:02}" / scenario / "result.json").read_text())

            development_counts = result["class_counts"]["development"]
            selection_rows.append({
                "replicate": rep,
                "scenario": scenario,
                "development_n_major": development_counts[3],
                "development_n_high": development_counts[2],
                "C_pooling": result["selection"]["logistic_pooling"]["value"],
                "C_target": result["selection"]["logistic_target"]["value"],
                "source_ess": result["methods"]["logistic_weighted"].get("source_ess"),
            })

            record_rate = mask.mean(axis=1).to_numpy()
            for k, label in enumerate(LABELS):
                use = y == k
                recording_rows.append({
                    "replicate": rep,
                    "scenario": scenario,
                    "severity": label,
                    "n": int(use.sum()),
                    "recording_rate": float(record_rate[use].mean()),
                })

            domain_x = np.vstack([source.iloc[:, :20].to_numpy(), pd.read_csv(d / "external_recorded_features.csv").to_numpy()])
            domain_y = np.r_[np.zeros(len(source)), np.ones(len(external))]
            domain_p = cross_val_predict(
                make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=6000)),
                domain_x,
                domain_y,
                cv=StratifiedKFold(5, shuffle=True, random_state=120),
                method="predict_proba",
            )[:, 1]
            domain_rows.append({
                "replicate": rep,
                "scenario": scenario,
                "external_domain_auc": safe_auc(domain_y, domain_p),
            })

            for method in ["majority", "logistic_pooling", "tca", "pca", "count_only", "context_augmented"]:
                p = stored[f"external__{method}"]
                scores = p[:, 2:].sum(axis=1)
                hard = score_row(y, p)
                top_n = max(1, int(np.ceil(0.10 * len(y))))
                order = np.argsort(-scores)[:top_n]
                prior_log_loss = -np.mean(np.log(np.maximum(prior_p[np.arange(len(y)), y], 1e-12)))
                model_log_loss = -np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)))
                prior_brier = np.mean(np.sum((prior_p - np.eye(4)[y]) ** 2, axis=1))
                model_brier = np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1))
                metric_rows.append({
                    "replicate": rep,
                    "scenario": scenario,
                    "method": method,
                    **hard,
                    "high_major_auroc": safe_auc(severe, scores),
                    "high_major_ap": safe_ap(severe, scores),
                    "top10_high_major_coverage": float(severe[order].sum() / max(1, severe.sum())),
                    "log_loss_skill": float(1 - model_log_loss / prior_log_loss),
                    "brier_skill": float(1 - model_brier / prior_brier),
                })

            train_codes = binary_code(pd.read_csv(d / "recorded_features.csv").iloc[:379])
            test_codes = binary_code(pd.read_csv(d / "external_recorded_features.csv"))
            seen = np.isin(test_codes, np.unique(train_codes))
            novelty_rows.extend([
                {"replicate": rep, "scenario": scenario, "group": "seen", "n": int(seen.sum())},
                {"replicate": rep, "scenario": scenario, "group": "unseen", "n": int((~seen).sum())},
            ])

    metrics = pd.DataFrame(metric_rows)
    recording = pd.DataFrame(recording_rows)
    domains = pd.DataFrame(domain_rows)
    selections = pd.DataFrame(selection_rows)
    novelty = pd.DataFrame(novelty_rows)
    metrics.to_csv(OUT / "threshold_independent_metrics.csv", index=False)
    recording.to_csv(OUT / "recording_rate_by_severity.csv", index=False)
    domains.to_csv(OUT / "domain_auc.csv", index=False)
    selections.to_csv(OUT / "selection_and_ess.csv", index=False)
    novelty.to_csv(OUT / "novelty_counts.csv", index=False)

    summary_rows = []
    primary = metrics[(metrics.scenario == "covariate_shift") & metrics.method.isin([
        "majority", "logistic_pooling", "tca", "pca", "count_only", "context_augmented"
    ])]
    for method, group in primary.groupby("method"):
        for metric in ["accuracy", "balanced_accuracy", "high_major_auroc", "high_major_ap",
                       "top10_high_major_coverage", "log_loss_skill", "brier_skill",
                       "recall_high", "recall_major"]:
            values = group[metric].dropna().to_numpy()
            if len(values) == 0:
                continue
            q = t.ppf(0.975, len(values) - 1) if len(values) > 1 else np.nan
            se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else np.nan
            summary_rows.append({
                "method": method,
                "metric": metric,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1) if len(values) > 1 else 0.0,
                "mcse": se,
                "low": values.mean() - q * se if len(values) > 1 else np.nan,
                "high": values.mean() + q * se if len(values) > 1 else np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(OUT / "composition_shift_summary.csv", index=False)
    protocol_hash = hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    report = {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "scope": "fixed release matrices and stored predictions; no data generation or primary refitting",
        "replicates": 30,
        "scenarios": 3,
        "metrics": ["accuracy", "balanced_accuracy", "class recall", "class precision", "high_major AUROC", "high_major AP", "top-decile coverage", "log-loss skill", "Brier skill"],
    }
    (OUT / "diagnostic_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
