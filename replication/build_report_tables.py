"""Export report tables from saved results without modifying manuscript sources."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_diagnostics/report_tables'
SCENARIOS = {'stationary':'Stationary', 'covariate_shift':'Composition shift',
             'conditional_shift':'Injury-mechanism shift'}
LABELS = ['low','medium','high','major']

def summary(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if not n: return dict(n=0,mean=np.nan,mcse=np.nan,low=np.nan,high=np.nan)
    se=x.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
    err=t.ppf(.975,n-1)*se if n>1 else np.nan
    return dict(n=n,mean=x.mean(),mcse=se,low=x.mean()-err,high=x.mean()+err)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(ROOT/'results/summary.csv')
    diag=pd.read_csv(ROOT/'results/revision_diagnostics/threshold_independent_metrics.csv')
    rows=[]
    body=r'''\begin{table}[htbp]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{2.3cm}lYYYYY}\toprule
Scenario & Model & Accuracy & Balanced accuracy & Macro-F1 & Log loss & Brier \\\midrule
'''
    for sc,caption in SCENARIOS.items():
        for method,name in [('majority','Prior'),('logistic_pooling','Pooling'),('tca','TCA')]:
            g=primary[(primary.scenario==sc)&(primary.cohort=='external')&(primary.method==method)].set_index('metric')
            b=summary(diag[(diag.scenario==sc)&(diag.method==method)].balanced_accuracy)
            a=g.loc['accuracy']
            vals=[caption,name,f"{100*a['mean']:.1f} ({100*a['low']:.1f}--{100*a['high']:.1f})",f"{b['mean']:.3f}"]
            vals += [f"{g.loc[k,'mean']:.3f}" for k in ['macro_f1','log_loss','brier']]
            body+=' & '.join(vals)+r' \\'+'\n'
            rows.append(dict(scenario=sc,method=method,accuracy=a['mean'],accuracy_low=a['low'],accuracy_high=a['high'],balanced_accuracy=b['mean']))
        body+=r'\addlinespace'+'\n'
    body+=r'''\bottomrule\end{tabularx}
\caption{Original-protocol external-cohort results across 30 replicates. Accuracy is percent with a 95\% Monte Carlo interval; other columns are mean scores. Log loss retains the original probability floor; smoothed comparisons are reported separately. Full method results, intervals and MCSEs are supplied with the replication files.}\label{tab:accuracy}
\end{table}'''
    pd.DataFrame(rows).to_csv(OUT/'primary_table_values.csv',index=False)
    (OUT/'primary_table.tex').write_text(body)

    # Every method, scenario and class: denominators, recall and precision.
    records=[]
    for sc in SCENARIOS:
        for rep in range(30):
            with np.load(ROOT/f'results/r{rep:02}/{sc}/predictions.npz') as z:
                y=z['true_external']
                for key in z.files:
                    if not key.startswith('external__'): continue
                    method=key[len('external__'):]; pred=z[key].argmax(1)
                    cm=np.bincount(4*y+pred,minlength=16).reshape(4,4)
                    for k,label in enumerate(LABELS):
                        n=cm[k].sum(); predicted=cm[:,k].sum()
                        records.append(dict(replicate=rep,scenario=sc,method=method,severity=label,
                            actual_n=n,predicted_n=predicted,recall=cm[k,k]/n if n else np.nan,
                            precision=cm[k,k]/predicted if predicted else np.nan))
    values=pd.DataFrame(records); values.to_csv(OUT/'class_metrics.csv',index=False)
    summaries=[]
    for keys,g in values.groupby(['scenario','method','severity']):
        for metric in ['actual_n','predicted_n','recall','precision']:
            summaries.append(dict(zip(['scenario','method','severity'],keys),metric=metric,**summary(g[metric])))
    pd.DataFrame(summaries).to_csv(OUT/'class_summary.csv',index=False)
    body=r'''\begin{table}[htbp]
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{2.3cm}llrrrr}\toprule
Scenario & Model & Class & True $n$ & Predicted $n$ & Recall (\%) & Precision (\%) \\\midrule
'''
    for sc,caption in SCENARIOS.items():
        for method,name in [('logistic_pooling','Pooling'),('tca','TCA')]:
            for label in LABELS:
                g=values[(values.scenario==sc)&(values.method==method)&(values.severity==label)]
                nvalid=g.precision.notna().sum()
                precision=f'{100*g.precision.mean():.2f} ({nvalid})' if nvalid else '-- (0)'
                body+=' & '.join([caption if label=='low' else '',name if label=='low' else '',label.title(),f'{g.actual_n.mean():.1f}',f'{g.predicted_n.mean():.1f}',f'{100*g.recall.mean():.2f}',precision])+r' \\'+'\n'
            body+=r'\addlinespace'+'\n'
    body+=r'''\bottomrule\end{tabularx}
\caption{External-cohort class counts and decisions, averaged across 30 replicates. Precision averages only defined values; parentheses give the number of replicates with at least one prediction for that class. Recall averages all 30 replicates. Full results and MCSEs for all 15 methods are in \texttt{report\_tables/class\_summary.csv}.}\label{tab:scenario_recall}
\end{table}'''
    (OUT/'class_table.tex').write_text(body)
    print('Exported table fragments and values; manuscript sources are unchanged.')

if __name__=='__main__': main()
