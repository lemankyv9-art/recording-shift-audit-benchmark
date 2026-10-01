"""Export legacy figure fragments from released rows/predictions only.

No event generator, model fitting, or model selection is invoked. TeX snippets
are also written separately so their plotted coordinates can be inspected.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_diagnostics/report_figures'
LABELS = ['low', 'medium', 'high', 'major']

def summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    mean = float(np.mean(x))
    err = float(t.ppf(.975, len(x)-1)*np.std(x, ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0
    return mean, err

def plot(coords, colour='blue!70!black', style='', errors=False):
    options='mark=*,mark size=1.8pt,thick,'+colour
    if style: options += ','+style
    if errors: options += ',error bars/.cd,y dir=both,y explicit'
    return '\\addplot+['+options+'] coordinates {'+' '.join(coords)+'};\n'

def panel(options, body):
    return '\\nextgroupplot['+options+']\n'+body

def figure(body, caption, label, height='5.7cm'):
    return r'''\begin{figure}[htbp]
\centering
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.7cm},
 width=0.46\textwidth,height='''+height+r''',grid=major,grid style={gray!15},
 label style={font=\small},tick label style={font=\small},
 legend style={font=\footnotesize,draw=none,fill=white}]
'''+body+r'''\end{groupplot}
\end{tikzpicture}
\caption{'''+caption+'}\n\\label{'+label+'}\n\\end{figure}\n'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=pd.read_csv(ROOT/'results/revision_diagnostics/threshold_independent_metrics.csv')
    all_records=pd.concat([pd.read_csv(ROOT/f'data/datasets/r{i:02}/covariate_shift/records.csv') for i in range(30)],ignore_index=True)
    proportions=all_records.severity.value_counts(normalize=True).reindex(LABELS)
    cls=panel(r'xlabel={Study category},ylabel={Proportion},xtick={0,1,2,3},xticklabels={Low,Medium,High,Major},ymin=0,ymax=.7',
              plot([f'({k},{proportions.loc[n]:.7f})' for k,n in enumerate(LABELS)],style='ybar,bar width=12pt'))
    # Loss ECDF includes every generated row; logarithmic x axis uses log(1+loss).
    loss=np.sort(np.log1p(all_records['economic loss'].to_numpy()))
    ix=np.unique(np.linspace(0,len(loss)-1,180).astype(int))
    cls+=panel(r'xlabel={Log(1 + loss in million CNY)},ylabel={Empirical CDF},ymin=0,ymax=1',
               plot([f'({loss[i]:.7f},{(i+1)/len(loss):.7f})' for i in ix],style='no marks'))
    distribution=figure(cls,'Generated source-plus-target distributions in composition shift (13,920 rows). Categories and losses describe the declared mechanism; these are not population estimates.','fig:distributions')
    # Class recall (all classes, paired replicate intervals).
    recall=panel(r'xlabel={True category},ylabel={Recall},xtick={0,1,2,3},xticklabels={Low,Medium,High,Major},ymin=0,ymax=1.02,legend pos=north east','')
    for m,lab,col in [('logistic_pooling','Pooling','blue!70!black'),('tca','TCA','red!70!black')]:
        g=primary[(primary.scenario=='covariate_shift')&(primary.method==m)]
        points=[]
        for i,c in enumerate(LABELS):
            mean,err=summary(g['recall_'+c]);points.append(f'({i},{mean:.7f}) +- (0,{err:.7f})')
        recall+=plot(points,col,errors=True)+r'\addlegendentry{'+lab+'}\n'
    # Reliability diagram: deciles within each independently fitted replicate.
    bins=[]
    for rep in range(30):
        with np.load(ROOT/f'results/r{rep:02}/covariate_shift/predictions.npz') as z:
            y=z['true_external']>=2
            for method in ['logistic_pooling','tca']:
                p=z['external__'+method][:,2:].sum(1)
                order=np.argsort(p,kind='stable')
                for k,chunk in enumerate(np.array_split(order,10)):
                    bins.append(dict(replicate=rep,method=method,bin=k,predicted=p[chunk].mean(),observed=y[chunk].mean(),n=len(chunk)))
    bins=pd.DataFrame(bins);bins.to_csv(OUT/'calibration_bins.csv',index=False)
    recall+=panel(r'xlabel={Mean predicted severe risk},ylabel={Observed severe fraction},xmin=0,xmax=.3,ymin=0,ymax=.3,legend pos=north west',
                  r'\addplot[gray,dashed,no marks] coordinates {(0,0) (.3,.3)};'+ '\n')
    for m,lab,col in [('logistic_pooling','Pooling','blue!70!black'),('tca','TCA','red!70!black')]:
        points=[]
        for k,g in bins[bins.method==m].groupby('bin'):
            mean,err=summary(g.observed);points.append(f'({g.predicted.mean():.7f},{mean:.7f}) +- (0,{err:.7f})')
        recall+=plot(points,col,errors=True)+r'\addlegendentry{'+lab+'}\n'
    calibration=figure(recall,r'Original-protocol composition-shift recall and severe-risk reliability. Points and 95\% Monte Carlo intervals summarise 30 replicates. Reliability bins contain equal numbers of rows within each replicate; the dashed line is ideal calibration. These curves diagnose saved probabilities and do not recalibrate models.','fig:recallcalibration')
    # Paired differences: connect all primary scenarios, use zero reference.
    paired=pd.read_csv(ROOT/'results/paired_differences.csv')
    body=''
    for metric,label in [('accuracy','TCA minus pooling accuracy'),('brier','TCA minus pooling Brier')]:
        x=paired[(paired.method=='tca')&(paired.cohort=='external')&(paired.metric==metric)]
        coords=[]
        for i,sc in enumerate(['stationary','covariate_shift','conditional_shift']):
            r=x[x.scenario==sc].iloc[0];coords.append(f'({i},{r.mean_difference:.7f}) +- (0,{(r.high-r.low)/2:.7f})')
        body+=panel(r'xlabel={Scenario},ylabel={'+label+r'},xtick={0,1,2},xticklabels={Stationary,Composition,Injury},xticklabel style={rotate=15,anchor=east}',
                    r'\addplot[gray,dashed,no marks] coordinates {(-.3,0) (2.3,0)};'+'\n'+plot(coords,style='only marks',errors=True))
    differences=figure(body,r'Paired original-protocol differences with 95\% Monte Carlo intervals. Positive accuracy differences and negative Brier differences favour TCA. Both metrics are on their original proportion/score scales; no multiplicity adjustment is applied.','fig:paired')
    # Revised learning curves, with epsilon=0.01 and expanded C grid.
    curves=pd.read_csv(ROOT/'results/revision_diagnostics/learning_metrics.csv')
    curves=curves[curves.scenario=='covariate_shift'];body=''
    for metric,lab in [('severe_auc','Severe-case AUROC'),('log_loss','Smoothed log loss')]:
        body+=panel(r'xlabel={Target training rows},ylabel={'+lab+r'},xtick={12,36,72,121},legend pos=north east','')
        for arm,label,col in [('target','Target only','red!70!black'),('pool','Pooling','blue!70!black')]:
            points=[]
            for n,g in curves[curves.arm==arm].groupby('n'):
                mean,err=summary(g[metric]);points.append(f'({n},{mean:.7f}) +- (0,{err:.7f})')
            body+=plot(points,col,errors=True)+r'\addlegendentry{'+label+'}\n'
    learning=figure(body,r'Sensitivity-extension learning curves on the independent composition-shift cohort. Each prefix is refitted using the expanded grid and training-prior smoothing with $\epsilon=0.01$. Bars are 95\% Monte Carlo intervals across 30 replicates. The 85-row primary holdout is not reused for scoring.','fig:learning')
    diagram=r'''\begin{figure}[htbp]
\centering
\begin{tikzpicture}[every node/.style={draw,rounded corners,align=center,font=\small,text width=3.7cm,minimum height=1cm},>=stealth]
\node (event) at (0,3.6) {Context, latent factors\\and shared event state};
\node (physical) at (0,1.8) {Vehicles, exposure\\and hazard mechanisms};
\node (outcome) at (-3.1,0) {Casualties and losses\\then study label $Y$};
\node (mask) at (3.1,0) {Complexity and report quality\\then assessment mask $M$};
\node (features) at (3.1,-1.8) {Recorded predictors\\$R=X\odot M$};
\draw[->] (event)--(physical);
\draw[->] (physical)--(outcome);
\draw[->] (physical)--(mask);
\draw[->] (event.east) to[bend left=45] (mask.north);
\draw[->] (mask)--(features);
\draw[->,dashed] (outcome)-- node[above,draw=none,text width=2cm,font=\scriptsize] {Severity-linked\\extension only} (mask);
\end{tikzpicture}
\caption{Schematic of the specified generating mechanism. The primary mask receives shared event complexity but no casualty or loss field. The dashed connection is added only in the severity-linked recording intervention. Models receive $R$; outcome fields and diagnostic masks are excluded from primary predictors.}
\label{fig:process}
\end{figure}'''
    figs={'process':diagram,'distributions':distribution,'paired':differences,'recallcalibration':calibration,'learning':learning}
    for name,fig in figs.items(): (OUT/(name+'.tex')).write_text(fig)
    print('Exported five legacy figure fragments; manuscript sources are unchanged.')

if __name__=='__main__': main()
