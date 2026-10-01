"""Strict fit/selection boundary and complete paired simulation evaluation."""
from pathlib import Path
import json,os,sys,argparse,warnings,hashlib,platform
import numpy as np
import pandas as pd
import scipy,sklearn
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.stats import t as student_t,binomtest,spearmanr
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold,cross_val_predict
from sklearn.metrics import log_loss,f1_score,confusion_matrix,roc_auc_score,r2_score
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning

HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'protocol.json').read_text());MP=SPEC['model_protocol']
DATA=Path(os.getenv('BENCHMARK_DATA',str(HERE.parent/'data/datasets')))
OUT=Path(os.getenv('BENCHMARK_RESULTS',str(HERE.parent/'results')))
F=['H01','H02','H05','H08','H10','H11','V01','V02','V03','V04','V05','V06','E01','E02','E04','E05','E06','E07','E08','E11']
L=['low','medium','high','major'];LM={v:i for i,v in enumerate(L)}
FITLOG=[];STAGE=''

class FailedFit(Exception):pass
def xy(d):return d[F].to_numpy(float),d.severity.map(LM).to_numpy(int)
def folds(y):
    counts=np.bincount(y,minlength=4);k=min(3,int(counts[counts>0].min()))
    if k>=2:return list(StratifiedKFold(k,shuffle=True,random_state=120).split(y,y))
    return list(KFold(3,shuffle=True,random_state=120).split(y))
def fit(kind,x,y,a,seed=0,w=None):
    if len(np.unique(y))==1:return DummyClassifier(strategy='prior').fit(x,y)
    for limit in [MP['initial_max_iter'],MP['retry_max_iter']]:
        if kind=='mlp':est=MLPClassifier(hidden_layer_sizes=tuple(MP['mlp_hidden']),activation=MP['activation'],solver='lbfgs',alpha=a,random_state=seed,max_iter=limit,max_fun=300000,tol=MP['tolerance'])
        else:est=LogisticRegression(C=a,solver='lbfgs',max_iter=limit,tol=MP['tolerance'])
        m=make_pipeline(StandardScaler(),est)
        name='mlpclassifier' if kind=='mlp' else 'logisticregression'
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always',ConvergenceWarning)
            m.fit(x,y,**{name+'__sample_weight':w})
        msgs=[str(v.message) for v in ws if issubclass(v.category,ConvergenceWarning)]
        FITLOG.append({'stage':STAGE,'kind':kind,'parameter':a,'seed':seed,'n':len(y),'p':x.shape[1],
          'budget':limit,'iterations':np.atleast_1d(est.n_iter_).astype(int).tolist(),'converged':not msgs,'warnings':msgs})
        if not msgs:return m
    raise FailedFit(f'Unresolved convergence: {STAGE}, {kind}, {a}, {seed}')
def probability(m,x):
    p=np.full((len(x),4),1e-12);p[:,np.array(m.classes_,int)]=m.predict_proba(x)
    return p/p.sum(axis=1,keepdims=True)
def score(y,p):
    pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=range(4))
    return {'accuracy':float(np.mean(pred==y)), 'macro_f1':float(f1_score(y,pred,labels=range(4),average='macro',zero_division=0)),
      'log_loss':float(log_loss(y,p,labels=range(4))),'brier':float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1))),
      'confusion':cm.tolist(),'recall':[float(cm[j,j]/cm[j].sum()) if cm[j].sum() else None for j in range(4)]}
def weight(xs,xt):
    m=LogisticRegression(C=.3,max_iter=6000,tol=1e-6).fit(np.vstack([xs,xt]),np.r_[np.zeros(len(xs)),np.ones(len(xt))])
    p=np.clip(m.predict_proba(xs)[:,1],1e-6,1-1e-6)
    w=np.clip(p/(1-p)*len(xs)/len(xt),*MP['weight_clip']);return w/w.mean()
def train(arm,xs,ys,xt,yt):
    if arm=='source':return xs,ys,None
    if arm=='target':return xt,yt,None
    return np.vstack([xs,xt]),np.r_[ys,yt],(np.r_[weight(xs,xt),np.ones(len(xt))] if arm=='weighted' else None)
def select(arm,kind,xs,ys,xt,yt):
    global STAGE
    rows=[]
    for a in MP['mlp_alpha' if kind=='mlp' else 'logistic_C']:
        ss=[];valid=True
        for j,(tr,va) in enumerate(folds(yt)):
            x,y,w=train(arm,xs,ys,xt[tr],yt[tr]);STAGE=f'{kind}:{arm}:CV:{j}'
            try:ss.append(score(yt[va],probability(fit(kind,x,y,a,0,w),xt[va])))
            except FailedFit:valid=False;break
        rows.append({'parameter':a,'valid':valid,'log_loss':float(np.mean([s['log_loss'] for s in ss])) if valid else None,
          'macro_f1':float(np.mean([s['macro_f1'] for s in ss])) if valid else None})
    valid=[r for r in rows if r['valid']]
    if not valid:raise FailedFit('No valid selection candidates')
    return min(valid,key=lambda r:(r['log_loss'],-r['macro_f1'],r['parameter']))['parameter'],rows

def kernel(a,b,sigma):return np.exp(-cdist(a,b,'sqeuclidean')/(2*sigma*sigma))
class TCA:
    def __init__(self,sigma,mu,dim=20):self.sigma=sigma;self.mu=mu;self.dim=dim
    def fit(self,xs,xt):
        self.x=np.vstack([xs,xt]);n=len(self.x);k=kernel(self.x,self.x,self.sigma)
        e=np.r_[np.ones(len(xs))/len(xs),-np.ones(len(xt))/len(xt)];v=k@e
        a=np.outer(v,v)+self.mu*np.eye(n);kc=k-k.mean(1,keepdims=True);b=kc@kc.T+MP['tca_ridge']*np.eye(n)
        _,self.w=eigh(a,b,subset_by_index=[0,self.dim-1],driver='gvx');return self
    def transform(self,x):return kernel(x,self.x,self.sigma)@self.w
class KPCA:
    def __init__(self,sigma,dim):self.sigma=sigma;self.dim=dim
    def fit(self,x):
        self.x=x;k=kernel(x,x,self.sigma);self.mean=k.mean(0);self.grand=k.mean()
        kc=k-self.mean[None,:]-self.mean[:,None]+self.grand
        vals,vec=eigh(kc,subset_by_index=[len(x)-self.dim,len(x)-1]);ix=np.argsort(vals)[::-1]
        self.w=vec[:,ix]/np.sqrt(np.maximum(vals[ix],1e-12));return self
    def transform(self,x):
        k=kernel(x,self.x,self.sigma);kc=k-k.mean(1,keepdims=True)-self.mean[None,:]+self.grand;return kc@self.w

def projection_select(xs,ys,xt,yt):
    global STAGE
    ff=folds(yt);rows=[]
    # Cache maximum-dimensional transforms; smaller d uses the same eigenvectors.
    for sig in MP['tca_sigma']:
      for mu in MP['tca_mu']:
        cache=[]
        for tr,va in ff:
            tca=TCA(sig,mu).fit(xs,xt[tr]);cache.append((tca.transform(np.vstack([xs,xt[tr]])),np.r_[ys,yt[tr]],tca.transform(xt[va]),yt[va]))
        for d in MP['tca_dims']:
          for C in MP['logistic_C']:
            scores=[];valid=True
            for j,(z,y,zv,yv) in enumerate(cache):
                STAGE=f'TCA:{d}:{sig}:{mu}:CV:{j}'
                try:scores.append(score(yv,probability(fit('logistic',z[:,:d],y,C),zv[:,:d])))
                except FailedFit:valid=False;break
            rows.append({'d':d,'sigma':sig,'mu':mu,'C':C,'valid':valid,
              'log_loss':float(np.mean([z['log_loss'] for z in scores])) if valid else None,
              'macro_f1':float(np.mean([z['macro_f1'] for z in scores])) if valid else None})
    valid=[z for z in rows if z['valid']]
    if not valid:raise FailedFit('No valid TCA candidates')
    return min(valid,key=lambda z:(z['log_loss'],-z['macro_f1'],z['d'],z['sigma'],z['mu'],z['C'])),rows

def novelty(trainx,testx,y,p):
    distance=cdist(testx,trainx,'hamming').min(1)*trainx.shape[1];distance=np.rint(distance).astype(int);counts=testx.sum(1)
    masks={'seen':distance==0,'unseen':distance>0,'hamming_1':distance==1,'hamming_ge2':distance>=2,
      'count_0_2':counts<=2,'count_3_5':(counts>=3)&(counts<=5),'count_ge6':counts>=6}
    out={}
    for name,m in masks.items():
        n=int(m.sum());correct=int(np.sum(p[m].argmax(1)==y[m])) if n else 0
        ci=binomtest(correct,n).proportion_ci() if n else None
        out[name]={'n':n,'correct':correct,'accuracy':correct/n if n else None,'ci95':[float(ci.low),float(ci.high)] if ci else None,
          'high_major':int(np.sum(y[m]>=2)),'mean_factor_count':float(counts[m].mean()) if n else None}
    return out

def run(rep,scenario):
    global FITLOG,STAGE
    FITLOG=[];d=DATA/f'r{rep:02}'/scenario
    src,tgt,ext=[pd.read_csv(d/f'{n}.csv') for n in ['source','target','external_test']]
    xs,ys=xy(src);xt,yt=xy(tgt.iloc[:36]);xq,yq=xy(tgt.iloc[36:]);xe,ye=xy(ext)
    trains=np.vstack([xs,xt]);trainy=np.r_[ys,yt];result={'replicate':rep,'scenario':scenario,'methods':{},'selection':{}}
    psmall={};plarge={}
    def save(name,ph,pe,extra=None):
        psmall[name]=ph;plarge[name]=pe
        result['methods'][name]={'holdout':score(yq,ph),'external':score(ye,pe)}
        if extra:result['methods'][name].update(extra)
    prior=DummyClassifier(strategy='prior').fit(trains,trainy);save('majority',probability(prior,xq),probability(prior,xe))
    for kind in ['logistic','mlp']:
      for arm in ['source','target','pooling','weighted']:
        a,grid=select(arm,kind,xs,ys,xt,yt);name=kind+'_'+arm;result['selection'][name]={'value':a,'grid':grid}
        x,y,w=train(arm,xs,ys,xt,yt);ph=[];pe=[]
        for seed in (MP['mlp_seeds'] if kind=='mlp' else [0]):
            STAGE=name+':final';m=fit(kind,x,y,a,seed,w);ph.append(probability(m,xq));pe.append(probability(m,xe))
        save(name,np.mean(ph,axis=0),np.mean(pe,axis=0),{'source_ess':float(w[:343].sum()**2/(w[:343]**2).sum())} if w is not None else None)
    cfg,grid=projection_select(xs,ys,xt,yt);result['selection']['tca']={'value':cfg,'grid':grid}
    tca=TCA(cfg['sigma'],cfg['mu'],cfg['d']).fit(xs,xt);z=tca.transform(trains);zq=tca.transform(xq);ze=tca.transform(xe)
    STAGE='tca:final';m=fit('logistic',z,trainy,cfg['C']);save('tca',probability(m,zq),probability(m,ze))
    result['projection_diagnostics']={'r2_z_to_x_external':float(r2_score(xe,LinearRegression().fit(z,trains).predict(ze))),
      'r2_x_to_z_external':float(r2_score(ze,LinearRegression().fit(trains,z).predict(xe))),
      'dimension':cfg['d'],'raw_dimension':20}
    for name in ['pca','kpca']:
        grid=[]
        for C in MP['logistic_C']:
            ss=[]
            for j,(tr,va) in enumerate(folds(yt)):
                xx=np.vstack([xs,xt[tr]]);yy=np.r_[ys,yt[tr]]
                proj=(PCA(n_components=cfg['d']) if name=='pca' else KPCA(cfg['sigma'],cfg['d'])).fit(xx)
                STAGE=name+f':CV:{j}';ss.append(score(yt[va],probability(fit('logistic',proj.transform(xx),yy,C),proj.transform(xt[va]))))
            grid.append({'C':C,'log_loss':float(np.mean([x['log_loss'] for x in ss])),'macro_f1':float(np.mean([x['macro_f1'] for x in ss]))})
        C=min(grid,key=lambda v:(v['log_loss'],-v['macro_f1'],v['C']))['C']
        proj=(PCA(n_components=cfg['d']) if name=='pca' else KPCA(cfg['sigma'],cfg['d'])).fit(trains)
        STAGE=name+':final';m=fit('logistic',proj.transform(trains),trainy,C)
        save(name,probability(m,proj.transform(xq)),probability(m,proj.transform(xe)))
        result['selection'][name]={'C':C,'d':cfg['d'],'sigma':cfg['sigma'] if name=='kpca' else None,'grid':grid}
    # Secondary mechanisms use fixed regularization, avoiding additional selection objectives.
    STAGE='count_only:final';m=fit('logistic',trains.sum(1)[:,None],trainy,.3);save('count_only',probability(m,xq.sum(1)[:,None]),probability(m,xe.sum(1)[:,None]))
    true=pd.read_csv(d/'model_factors.csv').to_numpy(float);true_ext=pd.read_csv(d/'external_model_factors.csv').to_numpy(float)
    STAGE='complete_factors:final';m=fit('logistic',true[:379],trainy,.3);save('complete_factors',probability(m,true[379:]),probability(m,true_ext))
    ctx=pd.read_csv(d/'event_context.csv');ec=pd.read_csv(d/'external_context.csv')
    def augment(x,c):
        return np.column_stack([x,np.log1p(c[['people_exposed','vehicles_involved']]),c[['bus_involved','hazardous_cargo','fire','rollover','built_environment']]])
    xc=augment(np.vstack([xs,xt,xq]),ctx);xce=augment(xe,ec)
    STAGE='context_augmented:final';m=fit('logistic',xc[:379],trainy,.3);save('context_augmented',probability(m,xc[379:]),probability(m,xce))
    dd=cdist(xq,trains,'hamming');lookup=np.zeros((len(xq),4))
    for i in range(len(xq)):lookup[i]=np.bincount(trainy[dd[i]==dd[i].min()],minlength=4)+1e-3
    lookup/=lookup.sum(1,keepdims=True)
    result['nearest_pattern_holdout']=score(yq,lookup)
    result['novelty']=novelty(trains,xe,ye,plarge['logistic_pooling'])
    _,groups=np.unique(xs,axis=0,return_inverse=True);cv={}
    for typ,ff in [('row',folds(ys)),('pattern_group',list(GroupKFold(3).split(xs,ys,groups)))]:
        pr=np.zeros((len(ys),4))
        for j,(tr,va) in enumerate(ff):
            STAGE=typ+f':CV:{j}';pr[va]=probability(fit('logistic',xs[tr],ys[tr],.3),xs[va])
        cv[typ]=score(ys,pr)
    result['source_cv']=cv
    domx=np.vstack([xs,xt]);domy=np.r_[np.zeros(len(xs)),np.ones(len(xt))]
    result['development_domain_auc']=float(roc_auc_score(domy,cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=6000)),domx,domy,
      cv=StratifiedKFold(3,shuffle=True,random_state=120),method='predict_proba')[:,1]))
    b=int(sum((psmall['logistic_pooling'].argmax(1)==yq)&(psmall['logistic_source'].argmax(1)!=yq)))
    c=int(sum((psmall['logistic_pooling'].argmax(1)!=yq)&(psmall['logistic_source'].argmax(1)==yq)))
    result['mcnemar_pool_vs_source']={'b':b,'c':c,'p':float(binomtest(b,b+c,.5).pvalue) if b+c else 1.0}
    result['class_counts']={k:np.bincount(y,minlength=4).tolist() for k,y in [('source',ys),('development',yt),('holdout',yq),('external',ye)]}
    result['inputs_sha256']={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['source.csv','target.csv','external_test.csv','model_factors.csv','event_context.csv']}
    result['protocol_sha256']=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    result['attempts']=len(FITLOG);result['retry_attempts']=sum(not z['converged'] for z in FITLOG);result['unresolved_final_fits']=0
    output=OUT/f'r{rep:02}'/scenario;output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions.npz',true_holdout=yq,true_external=ye,**{'holdout__'+k:v for k,v in psmall.items()},**{'external__'+k:v for k,v in plarge.items()})
    (output/'fits.json').write_text(json.dumps(FITLOG,indent=2)+'\n')
    result['completed']=True;(output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'finished r{rep:02} {scenario}; fits={len(FITLOG)}, retries={result["retry_attempts"]}',flush=True)

def aggregate():
    rows=[];rr=[]
    for rep in range(len(SPEC['replicate_seeds'])):
      for scenario in SPEC['scenarios']:
        r=json.loads((OUT/f'r{rep:02}'/scenario/'result.json').read_text());assert r['completed'];rr.append(r)
        for method,z in r['methods'].items():
          for cohort in ['holdout','external']:
            rows.append({'replicate':rep,'scenario':scenario,'method':method,'cohort':cohort,**{k:z[cohort][k] for k in ['accuracy','macro_f1','log_loss','brier']}})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'metrics.csv',index=False);summary=[];paired=[]
    for (sc,method,cohort),g in frame.groupby(['scenario','method','cohort']):
      for metric in ['accuracy','macro_f1','log_loss','brier']:
        x=g[metric].to_numpy();se=x.std(ddof=1)/np.sqrt(len(x));q=student_t.ppf(.975,len(x)-1)
        summary.append({'scenario':sc,'method':method,'cohort':cohort,'metric':metric,'n':len(x),'mean':x.mean(),'sd':x.std(ddof=1),'mcse':se,'low':x.mean()-q*se,'high':x.mean()+q*se})
        ref=frame[(frame.scenario==sc)&(frame.method=='logistic_pooling')&(frame.cohort==cohort)].set_index('replicate')[metric]
        dif=g.set_index('replicate')[metric].sort_index()-ref.sort_index();ds=dif.std(ddof=1)/np.sqrt(len(dif))
        paired.append({'scenario':sc,'method':method,'reference':'logistic_pooling','cohort':cohort,'metric':metric,'mean_difference':dif.mean(),'mcse':ds,'low':dif.mean()-q*ds,'high':dif.mean()+q*ds})
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    (OUT/'run_summary.json').write_text(json.dumps({'completed':True,'repeats':len(SPEC['replicate_seeds']),'scenarios':list(SPEC['scenarios']),
      'methods':sorted(frame.method.unique().tolist()),'attempts':sum(r['attempts'] for r in rr),'retries':sum(r['retry_attempts'] for r in rr),'unresolved_final_fits':0,
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__}},indent=2)+'\n')
    print('Aggregation complete',len(rr),'scenario runs',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--end',type=int,default=30);ap.add_argument('--aggregate',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.aggregate:aggregate()
    else:
      for rep in range(args.start,min(args.end,len(SPEC['replicate_seeds']))):
       for scenario in SPEC['scenarios']:
        p=OUT/f'r{rep:02}'/scenario/'result.json'
        if p.exists():
            z=json.loads(p.read_text())
            if (z.get('completed') and z['protocol_sha256']==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
                and all(hashlib.sha256((DATA/f'r{rep:02}'/scenario/n).read_bytes()).hexdigest()==h for n,h in z['inputs_sha256'].items())):continue
        run(rep,scenario)
