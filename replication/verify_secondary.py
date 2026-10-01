"""Verify secondary learning curves and diagnostics without refitting."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,log_loss
HERE=Path(__file__).resolve().parent
DATA=Path(__import__('os').getenv('BENCHMARK_DATA',str(HERE.parent/'data/datasets')))
OUT=Path(__import__('os').getenv('BENCHMARK_RESULTS',str(HERE.parent/'results')))
LM={'low':0,'medium':1,'high':2,'major':3}
primary_methods=['logistic_target','logistic_pooling']
learn=pd.read_csv(OUT/'supplementary/learning_curve.csv'); prob=np.load(OUT/'supplementary/learning_predictions.npz')
assert len(learn)==900 and len(prob.files)==900
for row in learn.to_dict('records'):
 p=DATA/f'r{int(row["replicate"]):02}'/row['scenario']; d=pd.read_csv(p/'external_test.csv'); y=d.severity.map(LM).to_numpy(); key=f'r{int(row["replicate"]):02}__{row["scenario"]}__{row["method"]}__{int(row["target_n"])}'; z=prob[key];pred=z.argmax(1)
 vals=[np.mean(pred==y),f1_score(y,pred,labels=range(4),average='macro',zero_division=0),log_loss(y,z,labels=range(4)),np.mean(np.sum((z-np.eye(4)[y])**2,axis=1))]
 assert z.shape==(2000,4) and np.isfinite(z).all() and np.allclose(z.sum(1),1,atol=1e-12)
 assert np.allclose(vals,[row[x] for x in ['accuracy','macro_f1','log_loss','brier']],atol=1e-12)
 if int(row['target_n'])==36:
  primary=json.loads((OUT/f'r{int(row["replicate"]):02}'/row['scenario']/'result.json').read_text())['methods'][row['method'].replace('target','logistic_target').replace('pooling','logistic_pooling')]['external']
  assert np.allclose(vals,[primary[x] for x in ['accuracy','macro_f1','log_loss','brier']],atol=1e-12)
fit=json.loads((OUT/'supplementary/fits.json').read_text()); rs=json.loads((OUT/'supplementary/run_summary.json').read_text())
assert rs['completed'] and rs['learning_fits']==900 and rs['fit_attempts']==8346 and rs['nonconverged_attempts']==0 and len(fit)==8346
recall=pd.read_csv(OUT/'supplementary/class_recall.csv'); desc=pd.read_csv(OUT/'supplementary/data_descriptives.csv'); proj=pd.read_csv(OUT/'supplementary/projection_diagnostics.csv'); cv=pd.read_csv(OUT/'supplementary/source_cv.csv')
assert len(recall)==30*3*15*2*4 and len(desc)==30*3*3 and len(proj)==30*3*3 and len(cv)==30*3*2
report={'status':'PASS','learning_fits_recomputed':900,'prediction_arrays':900,'fit_attempts':8346,'nonconverged_attempts':0,'prefix_36_matches_primary':True,'class_recall_rows':len(recall),'descriptive_rows':len(desc),'projection_rows':len(proj),'source_cv_rows':len(cv)}
(OUT/'supplementary/verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
