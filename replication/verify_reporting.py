"""Recompute reported decisions and comparisons without changing observations or fits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_diagnostics'
TEXT = ROOT / 'manuscript'


def source_text(name):
    """Return a manuscript source if present; the public repository omits the manuscript."""
    path = TEXT / name
    return path.read_text() if path.exists() else None


def check_text(fragment, text, label=None):
    """Assert that a reported value appears in a manuscript source when that source is present."""
    if text is not None:
        assert fragment in text, label if label is not None else fragment


def main():
    rows = []
    diagnostic = pd.read_csv(OUT / 'threshold_independent_metrics.csv')
    for rep in range(30):
        base = ROOT / f'data/datasets/r{rep:02}/covariate_shift'
        data = pd.read_csv(base / 'external_test.csv')
        y = data.severity.isin(['high', 'major']).to_numpy()
        stored = np.load(ROOT / f'results/r{rep:02}/covariate_shift/predictions.npz')
        for method in diagnostic.method.unique():
            scores = stored[f'external__{method}'][:, 2:].sum(axis=1)
            saved = diagnostic[(diagnostic.replicate == rep) & (diagnostic.scenario == 'covariate_shift') & (diagnostic.method == method)].iloc[0]
            assert np.isclose(roc_auc_score(y, scores), saved.high_major_auroc)
        p = stored['external__logistic_pooling']
        marked = p[:, 2:].sum(axis=1) >= 1 / 11
        tp = np.sum(marked & y)
        rows.append(dict(replicate=rep, severe_recall=tp/y.sum(),
                         severe_precision=tp/marked.sum(), marked_fraction=marked.mean(),
                         severe_prevalence=y.mean(), cost_loss=(10*np.sum(~marked & y)+np.sum(marked & ~y))/len(y)))
    primary = pd.DataFrame(rows)
    primary.to_csv(OUT / 'primary_cost_metrics.csv', index=False)
    # Compare the fitted 1/11 rule with constant decisions under the same labels.
    constant_costs = []
    for rep in range(30):
        base = ROOT / f'data/datasets/r{rep:02}/covariate_shift'
        y = pd.read_csv(base / 'external_test.csv').severity.isin(['high', 'major']).to_numpy()
        constant_costs.append(dict(replicate=rep, prevalence=y.mean(), mark_none=10*y.mean(), mark_all=(~y).mean()))
    constants = pd.DataFrame(constant_costs)
    assert np.isclose(constants.mark_all.mean(), 0.9123333333333333, atol=1e-12)
    assert np.isclose(constants.mark_none.mean(), 0.8766666666666667, atol=1e-12)
    selection = pd.read_csv(OUT / 'selection_metrics.csv')
    conditional = selection[selection.scenario == 'conditional_shift']
    base = conditional[conditional.method == 'source_fixed'].set_index('replicate').sort_index()
    adjusted = conditional[conditional.method == 'source_intercept'].set_index('replicate').sort_index()
    assert base.index.equals(adjusted.index) and len(base) == 30
    comparisons = []
    for metric in ['log_loss', 'calibration_citl', 'cost_recall', 'cost_precision', 'cost_positive_fraction', 'cost_loss']:
        diff = adjusted[metric] - base[metric]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        radius = t.ppf(.975, len(diff)-1)*se
        comparisons.append(dict(metric=metric, baseline='source_fixed', method='source_intercept',
                                baseline_mean=base[metric].mean(), adjusted_mean=adjusted[metric].mean(),
                                mean_difference=diff.mean(), mcse=se, low=diff.mean()-radius, high=diff.mean()+radius))
    pd.DataFrame(comparisons).to_csv(OUT / 'source_adjustment_comparison.csv', index=False)
    tex = source_text('supplementary.tex')
    calibration = selection[selection.scenario == 'covariate_shift']
    for method, label in [('pool_fixed','Pooled LR, fixed'),('tca_fixed','TCA, fixed'),('ordered_fixed','Ordered LR, fixed'),('count','Factor-count LR')]:
        q = calibration[calibration.method == method]
        fields = []
        for metric, decimals in [('calibration_citl',3),('calibration_slope',2),('severe_auc',3),('severe_ap',3)]:
            sddec = 2 if metric.startswith('calibration') else 3
            fields.append(f'{q[metric].mean():.{decimals}f} ({q[metric].std(ddof=1):.{sddec}f})')
        check_text(label+' & '+' & '.join(fields), tex, label)
    main_text = source_text('manuscript.tex')
    for value in [primary.severe_recall.mean(), primary.severe_precision.mean(), primary.marked_fraction.mean()]:
        check_text(f'{100*value:.1f}'+r'\%', main_text)
    effects = pd.read_csv(OUT / 'factorial_effects.csv')
    q = effects[(effects.method=='pooled_fixed') & (effects.factor=='shift') & (effects.contrast=='2.0-0.0')]
    assert q[q.metric=='balanced_accuracy']['mean'].item() > 0
    assert q[q.metric=='severe_auc']['mean'].item() < 0
    # The original learning-curve table is generated from results/supplementary/learning_curve.csv.
    learning = pd.read_csv(ROOT / 'results/supplementary/learning_curve.csv')
    expected = {
        ('stationary','target'):[57.4,59.1,59.3,59.9,60.8],
        ('stationary','pooling'):[61.6,62.0,61.9,62.1,62.2],
        ('covariate_shift','target'):[55.4,55.8,56.1,57.3,57.7],
        ('covariate_shift','pooling'):[58.5,58.7,58.9,59.0,59.1],
        ('conditional_shift','target'):[46.3,46.6,47.3,48.9,49.3],
        ('conditional_shift','pooling'):[50.9,51.0,51.0,51.2,51.3],
    }
    for (scenario, method), values in expected.items():
        q = learning[(learning.scenario == scenario) & (learning.method == method)].groupby('target_n').accuracy.mean().reindex([12,24,36,72,121]).to_numpy() * 100
        assert np.allclose(q, values, atol=0.05)
    # Reconstruct both constant decisions and every cost row in Supplementary Table S18.
    conditional_prevalence = []
    for rep in range(30):
        frame = pd.read_csv(ROOT / f'data/datasets/r{rep:02}/conditional_shift/external_test.csv')
        conditional_prevalence.append(frame.severity.isin(['high', 'major']).mean())
    conditional_prevalence = pd.Series(conditional_prevalence, index=base.index)
    assert np.allclose(base.severe_prevalence, conditional_prevalence)
    assert np.allclose(adjusted.severe_prevalence, conditional_prevalence)
    decision_rows = []
    settings = [
        ('Primary composition shift', constants.prevalence,
         [('Pooled LR, $1/11$', primary.marked_fraction, primary.cost_loss)]),
        ('Extension injury-mechanism shift', conditional_prevalence,
         [('Source-fixed LR, $1/11$', base.cost_positive_fraction, base.cost_loss),
          ('Source-intercept LR, $1/11$', adjusted.cost_positive_fraction, adjusted.cost_loss)])
    ]
    for setting, prevalence, methods in settings:
        rules = [('Mark none severe', np.zeros(30), 10*prevalence),
                 ('Mark all severe', np.ones(30), 1-prevalence)] + methods
        for label, marked, costs in rules:
            expected_row = f'{label} & {100*np.mean(marked):.1f} & {np.mean(costs):.3f}'
            check_text(expected_row, tex, (setting, expected_row))
            for rep, (fraction, cost) in enumerate(zip(marked, costs)):
                decision_rows.append(dict(setting=setting, method=label, replicate=rep,
                                          marked_fraction=fraction, cost=cost))
    pd.DataFrame(decision_rows).to_csv(OUT / 'decision_baseline_metrics.csv', index=False)
    cost_diff = primary.cost_loss.to_numpy() - np.minimum(constants.mark_none, constants.mark_all)
    assert np.all(cost_diff < 0)
    radius = t.ppf(.975, 29) * cost_diff.std(ddof=1) / np.sqrt(30)
    check_text(f'{cost_diff.mean():.3f}', main_text)
    for bound in [cost_diff.mean()-radius, cost_diff.mean()+radius]:
        check_text(f'{bound:.3f}', main_text)

    # Factorial intervals have 30 independent seed units, not 72*30 independent cells.
    factorial = pd.read_csv(OUT / 'factorial_results.csv')
    keys = ['shift', 'signal', 'development_n', 'recording', 'fatality_intercept']
    levels = [('signal', 1., 3.), ('shift', 0., .8), ('shift', 0., 2.),
              ('development_n', 36, 121), ('recording', 'complexity', 'severity'),
              ('recording', 'complexity', 'drift'), ('fatality_intercept', 0., .65)]
    effect_checks = 0
    metrics = effects.metric.unique().tolist()
    for method, group in factorial.groupby('method'):
        for factor, low, high in levels:
            index = ['replicate'] + [key for key in keys if key != factor]
            left = group[group[factor] == low].set_index(index)[metrics].sort_index()
            right = group[group[factor] == high].set_index(index)[metrics].sort_index()
            assert left.index.equals(right.index)
            delta = (right-left).groupby(level='replicate').mean()
            assert len(delta) == 30
            saved = effects[(effects.method == method) & (effects.factor == factor)
                            & (effects.contrast == f'{high}-{low}')].set_index('metric')
            assert set(saved.index) == set(metrics)
            for metric in metrics:
                values = delta[metric].replace([np.inf, -np.inf], np.nan).dropna()
                n = len(values)
                row = saved.loc[metric]
                assert row['n'] == n
                if n == 0:
                    assert row[['mean','mcse','low','high']].isna().all()
                elif n == 1:
                    assert np.isclose(row['mean'], values.mean())
                    assert row[['mcse','low','high']].isna().all()
                else:
                    mean = values.mean()
                    mcse = values.std(ddof=1)/np.sqrt(n)
                    assert np.allclose([row['mean'], row['mcse'], row['low'], row['high']],
                                       [mean, mcse, mean-t.ppf(.975,n-1)*mcse, mean+t.ppf(.975,n-1)*mcse],
                                       atol=1e-12, rtol=1e-10), (method, factor, metric)
                effect_checks += 1
    for method in ['pool_fixed','pool_cv','pool_oracle','tca_fixed','tca_cv','tca_oracle']:
        check_text(f'{calibration[calibration.method == method].severe_auc.mean():.3f}', main_text)
    report = dict(status='PASS', primary_cost_means=primary.drop(columns='replicate').mean().to_dict(),
                  constant_cost_means=constants.drop(columns='replicate').mean().to_dict(),
                  primary_auroc_checks=180, calibration_table_rows_checked=4 if tex is not None else 0,
                  paired_source_comparison_replicates=30, learning_table_rows_checked=6,
                  decision_cost_table_rows_checked=7 if tex is not None else 0, factorial_contrasts_checked=effect_checks,
                  manuscript_text_checks='run' if tex is not None and main_text is not None else 'skipped: manuscript sources not present',
                  scope='Stored primary predictions, calibration and decision-cost tables, selection means, and seed-level factorial intervals; no refitting or data generation.')
    (OUT / 'reporting_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
