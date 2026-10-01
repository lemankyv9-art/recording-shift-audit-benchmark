"""Secondary learning curves, rare-class, coverage and projection diagnostics.
The primary data and primary analysis outputs are not modified.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import t, spearmanr
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import analysis_pipeline as ap
HERE=Path(__file__).resolve().parent
OUT=ap.OUT/'supplementary'; OUT.mkdir(parents=True,exist_ok=True)
SIZES=[12,24,36,72,121]
SPEC={'design':'Secondary exploratory analyses added after the primary run; no changes to primary data, hyperparameters or comparisons.', 'learning_sizes':SIZES,'learning_evaluation':'2,000-row independent cohort only; target prefixes 72/121 reuse primary holdout rows for training, so no primary holdout scores are reported for those fits.','selection':'Same log-loss target CV as primary for target and pooling logistic; all prefixes fixed; no choice by independent-cohort results.', 'repeats':30,'scenarios':list(ap.SPEC['scenarios'])}
(HERE/'supplementary_protocol.json').write_text(json.dumps(SPEC,indent=2)+'\n')
curves=[]; recalls=[]; descriptive=[]; diagnostics=[]; probabilities={};cvrows=[]
for rep in range(30):
    for scenario in ap.SPEC['scenarios']:
        p=ap.DATA/f'r{rep:02}'/scenario
        src,tgt,ext=[pd.read_csv(p/f'{n}.csv') for n in ['source','target','external_test']]
        xs,ys=ap.xy(src);xe,ye=ap.xy(ext)
        primary=json.loads((ap.OUT/f'r{rep:02}'/scenario/'result.json').read_text())
        for method,entries in primary['methods'].items():
            for cohort in ['holdout','external']:
                for j,label in enumerate(ap.L):
                    recalls.append({'replicate':rep,'scenario':scenario,'method':method,'cohort':cohort,'class':label,'recall':entries[cohort]['recall'][j],'n_class':primary['class_counts'][cohort][j]})
        for k,v in primary['source_cv'].items():
            cvrows.append({'replicate':rep,'scenario':scenario,'split':k,**{m:v[m] for m in ['accuracy','macro_f1','log_loss','brier']}})
        for cohort,d,ctx_name in [('source',src,'event_context'),('target',tgt,'event_context'),('external',ext,'external_context')]:
            ctx=pd.read_csv(p/(ctx_name+'.csv'))
            ctx=ctx if cohort=='external' else ctx.iloc[:343] if cohort=='source' else ctx.iloc[343:].reset_index(drop=True)
            row={'replicate':rep,'scenario':scenario,'cohort':cohort,'n':len(d),'fatal_mean':d.fatalities.mean(),'fatal_max':d.fatalities.max(),'loss_median':d['economic loss'].median(),'loss_p95':d['economic loss'].quantile(.95),'loss_max':d['economic loss'].max(),'rho_count_severity':spearmanr(d[ap.F].sum(axis=1),d.severity.map(ap.LM)).statistic,'full_vector_duplicates':int(d.duplicated().sum())}
            for label in ap.L:row['n_'+label]=int((d.severity==label).sum())
            descriptive.append(row)
        xt,yt=ap.xy(tgt.iloc[:36]);train=np.vstack([xs,xt]);cfg=primary['selection']['tca']['value']
        for name in ['pca','kpca']:
            proj=(PCA(n_components=cfg['d']) if name=='pca' else ap.KPCA(cfg['sigma'],cfg['d'])).fit(train)
            z=proj.transform(train);ze=proj.transform(xe)
            diagnostics.append({'replicate':rep,'scenario':scenario,'method':name,'dimension':cfg['d'],'sigma':cfg['sigma'] if name=='kpca' else None,'reconstruction_r2':r2_score(xe,LinearRegression().fit(z,train).predict(ze))})
        diagnostics.append({'replicate':rep,'scenario':scenario,'method':'tca','dimension':cfg['d'],'sigma':cfg['sigma'],'reconstruction_r2':primary['projection_diagnostics']['r2_z_to_x_external']})
        for size in SIZES:
            tx,ty=ap.xy(tgt.iloc[:size])
            for arm in ['target','pooling']:
                a,grid=ap.select(arm,'logistic',xs,ys,tx,ty)
                x,y,w=ap.train(arm,xs,ys,tx,ty)
                ap.STAGE=f'learning:{rep}:{scenario}:{arm}:{size}:final';model=ap.fit('logistic',x,y,a,w=w)
                pr=ap.probability(model,xe);v=ap.score(ye,pr)
                key=f'r{rep:02}__{scenario}__{arm}__{size}'
                probabilities[key]=pr
                curves.append({'replicate':rep,'scenario':scenario,'method':arm,'target_n':size,'selected_C':a,**{m:v[m] for m in ['accuracy','macro_f1','log_loss','brier']}})
    print(f'supplementary r{rep:02} complete',flush=True)
pd.DataFrame(curves).to_csv(OUT/'learning_curve.csv',index=False)
pd.DataFrame(recalls).to_csv(OUT/'class_recall.csv',index=False)
pd.DataFrame(descriptive).to_csv(OUT/'data_descriptives.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'projection_diagnostics.csv',index=False)
pd.DataFrame(cvrows).to_csv(OUT/'source_cv.csv',index=False)
np.savez_compressed(OUT/'learning_predictions.npz',**probabilities)
(OUT/'fits.json').write_text(json.dumps(ap.FITLOG,indent=2)+'\n')
report={'completed':True,'learning_fits':len(curves),'fit_attempts':len(ap.FITLOG),'nonconverged_attempts':sum(not x['converged'] for x in ap.FITLOG),'protocol_sha256':hashlib.sha256((HERE/'supplementary_protocol.json').read_bytes()).hexdigest()}
(OUT/'run_summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
