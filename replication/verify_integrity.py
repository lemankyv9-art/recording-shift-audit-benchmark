"""Independent dataset, prediction, metric and summary verification.
Works on released fixed data without importing the generator or analysis code.
"""
from pathlib import Path
import argparse, hashlib, json, os
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import confusion_matrix, f1_score, log_loss
HERE=Path(__file__).resolve().parent
DATA=Path(os.getenv('BENCHMARK_DATA', str(HERE.parent/'data/datasets')))
OUT=Path(os.getenv('BENCHMARK_RESULTS', str(HERE.parent/'results')))
SPEC=json.loads((HERE/'protocol.json').read_text())
F=['H01','H02','H05','H08','H10','H11','V01','V02','V03','V04','V05','V06','E01','E02','E04','E05','E06','E07','E08','E11']
LABELS=['low','medium','high','major']; LM={v:i for i,v in enumerate(LABELS)}
COUNT=0

def ck(value,message):
    global COUNT
    COUNT+=1
    if not bool(value): raise AssertionError(message)

def close(a,b,message):
    ck(np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True),message)

def verify_data():
    ns,nt,ne,nd=[SPEC[k] for k in ['source_n','target_n','external_test_n','development_n']]
    for rep in range(len(SPEC['replicate_seeds'])):
        root=DATA/f'r{rep:02}'
        for sc in SPEC['scenarios']:
            p=root/sc
            src,tgt,ext=[pd.read_csv(p/f'{n}.csv') for n in ['source','target','external_test']]
            full=pd.concat([src,tgt],ignore_index=True)
            for d,n in [(src,ns),(tgt,nt),(ext,ne)]:
                ck(d.shape==(n,23),f'{p}: dimensions')
                ck(d.columns.tolist()==F+['fatalities','economic loss','severity'],'schema')
                ck(not d.isna().any().any(),'no missing values in generated outputs')
                ck(d[F].isin([0,1]).all().all(),'binary factors')
                ck(((d.fatalities>=0)&(d.fatalities%1==0)).all(),'nonnegative integer fatality')
                ck((d['economic loss']>=0).all(),'nonnegative loss')
                expected=np.select([(d.fatalities>=10)|(d['economic loss']>=50),(d.fatalities>=3)|(d['economic loss']>=10),(d.fatalities>=1)|(d['economic loss']>=1)],['major','high','medium'],default='low')
                ck(np.array_equal(expected,d.severity),'labels from declared threshold rule')
            for prefix,d,cfile in [('',full,'event_context.csv'),('external_',ext,'external_context.csv')]:
                c=pd.read_csv(p/cfile)
                latent,recorded,mask=[pd.read_csv(p/(prefix+n+'.csv')) for n in ['model_factors','recorded_features','assessment_mask']]
                for x in [latent,recorded,mask]:
                    ck(x.shape==(len(d),20) and x.isin([0,1]).all().all(),'factor/mask shape and binary')
                ck(recorded.equals(d[F]),'primary data are recorded factors')
                ck(np.array_equal(recorded,latent.to_numpy()*mask.to_numpy()),'latent times mask')
                ck((latent.E04<=latent.E11).all() and (recorded.E04<=recorded.E11).all(),'curved ramp subset')
                ck(len(c)==len(d) and not c.isna().any().any(),'context coverage')
                counts=['trucks','cars','bus_involved','bus_seats','bus_occupants','vehicle_occupants','car_occupants','vulnerable_road_users','community_present','external_exposed','people_exposed','vehicles_involved','serious_injuries','minor_injuries','unhurt','occupant_fatalities','vru_fatalities','external_fatalities']
                ck(((c[counts]>=0)&(c[counts]%1==0)).all().all(),'nonnegative counts')
                ck((c.vehicles_involved==c.trucks+c.cars+c.bus_involved).all(),'vehicle count')
                ck((c.car_occupants>=c.cars).all(),'at least car driver')
                ck((c.bus_occupants>=c.bus_involved).all() and (c.bus_occupants<=c.bus_seats).all(),'bus driver and capacity')
                ck(((c.vehicle_occupants-c.car_occupants-c.bus_occupants)>=c.trucks).all(),'truck drivers')
                ck((c.people_exposed==c.vehicle_occupants+c.vulnerable_road_users+c.external_exposed).all(),'exposure count')
                ck((c.people_exposed==d.fatalities+c.serious_injuries+c.minor_injuries+c.unhurt).all(),'casualty sum')
                ck((d.fatalities==c.occupant_fatalities+c.vru_fatalities+c.external_fatalities).all(),'fatality groups')
                ck((c.occupant_fatalities<=c.vehicle_occupants).all() and (c.vru_fatalities<=c.vulnerable_road_users).all() and (c.external_fatalities<=c.external_exposed).all(),'role fatality bounds')
                ck((c.external_exposed<=c.community_present).all(),'community exposure bound')
                path=(c.fire+c.hazardous_release+c.crowd_collision)>0
                ck(((c.external_exposed==0)|path).all(),'community exposure requires mechanism')
                ck((c.property_loss_million_cny<=c.asset_value_million_cny+1e-8).all(),'property loss bound')
                comps=c[['property_loss_million_cny','casualty_cost_million_cny','response_cost_million_cny']]
                ck((comps>=0).all().all(),'loss components nonnegative')
                ck(np.max(abs(comps.sum(axis=1)-c.loss_raw_million_cny))<=0.000151,'loss components rounding tolerance')
                ck(c.loss_precision.isin([.001,.01,.1]).all(),'loss precision support')
                rounded=np.round(np.round(c.loss_raw_million_cny/c.loss_precision)*c.loss_precision,3)
                ck(np.allclose(rounded,d['economic loss'],atol=1e-7,rtol=0),'rounded reported loss')
            meta=pd.read_csv(p/'metadata.csv'); rec=pd.read_csv(p/'records.csv')
            ck(meta.record_id.is_unique,'unique generated keys')
            ck(meta.record_order.tolist()==list(range(1,ns+nt+1)),'row order')
            ck(meta.split.tolist()==['source_train']*ns+['target_development']*nd+['target_holdout']*(nt-nd),'split membership')
            ck(rec.equals(pd.concat([meta,full],axis=1)),'records view matches')
            ck(src.equals(pd.read_csv(root/'stationary/source.csv')),'paired source identical across scenarios')
        a,b=root/'covariate_shift',root/'conditional_shift'
        for name in ['model_factors.csv','recorded_features.csv','assessment_mask.csv','external_model_factors.csv','external_recorded_features.csv','external_assessment_mask.csv']:
            ck(pd.read_csv(a/name).equals(pd.read_csv(b/name)),'conditional shift keeps factor and mask draws fixed')
        outcome_cols=['serious_injuries','minor_injuries','unhurt','casualty_cost_million_cny','loss_raw_million_cny','occupant_fatalities','vru_fatalities','external_fatalities']
        for name in ['event_context.csv','external_context.csv']:
            ck(pd.read_csv(a/name).drop(columns=outcome_cols).equals(pd.read_csv(b/name).drop(columns=outcome_cols)),'conditional shift context pairing')
    manifest=DATA.parent/'DATA_MANIFEST.json'
    saved=json.loads(manifest.read_text())
    actual={str(p.relative_to(DATA.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.rglob('*.csv'))}
    ck(actual==saved,'data manifest exact file set and hashes')

def verify_results():
    run_summary=json.loads((OUT/'run_summary.json').read_text()); frame=[]; attempts=0; retries=0
    ck(run_summary['completed'],'completed flag')
    nd=SPEC['development_n']
    protocol_hash=hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    for rep in range(len(SPEC['replicate_seeds'])):
        for sc in SPEC['scenarios']:
            d=DATA/f'r{rep:02}'/sc; p=OUT/f'r{rep:02}'/sc
            r=json.loads((p/'result.json').read_text()); fits=json.loads((p/'fits.json').read_text())
            ck(r['completed'] and r['protocol_sha256']==protocol_hash,'run protocol')
            for name,h in r['inputs_sha256'].items():ck(hashlib.sha256((d/name).read_bytes()).hexdigest()==h,'analysis input hash')
            ck(len(fits)==r['attempts'],'fit count')
            ck(all(0<x['n']<=SPEC['source_n']+nd for x in fits),'fitting row bound')
            ck(all(x['converged'] or ':CV:' in x['stage'] for x in fits),'no unresolved final fit')
            attempts+=len(fits);retries+=r['retry_attempts']
            z=np.load(p/'predictions.npz'); source=pd.read_csv(d/'source.csv'); target=pd.read_csv(d/'target.csv'); external=pd.read_csv(d/'external_test.csv')
            for cohort,dd in [('holdout',target.iloc[nd:]),('external',external)]:
                y=dd.severity.map(LM).to_numpy();ck(np.array_equal(z['true_'+cohort],y),'test labels match data')
                ck(r['class_counts'][cohort]==np.bincount(y,minlength=4).tolist(),'class counts')
                for method,entry in r['methods'].items():
                    pr=z[cohort+'__'+method];ck(pr.shape==(len(y),4),'probability dimensions')
                    ck(np.isfinite(pr).all() and (pr>=0).all() and (pr<=1).all(),'valid probabilities');close(pr.sum(axis=1),1,'probability sum')
                    pred=pr.argmax(axis=1);cm=confusion_matrix(y,pred,labels=range(4))
                    vals={'accuracy':np.mean(pred==y),'macro_f1':f1_score(y,pred,average='macro',labels=range(4),zero_division=0),'log_loss':log_loss(y,pr,labels=range(4)),'brier':np.mean(np.sum((pr-np.eye(4)[y])**2,axis=1))}
                    for metric,val in vals.items():close(val,entry[cohort][metric],'recomputed '+metric)
                    ck(cm.tolist()==entry[cohort]['confusion'],'confusion matrix')
                    recall=[cm[j,j]/cm[j].sum() if cm[j].sum() else np.nan for j in range(4)]
                    close(recall,[np.nan if x is None else x for x in entry[cohort]['recall']],'class recall')
                    frame.append({'replicate':rep,'scenario':sc,'method':method,'cohort':cohort,**vals})
    expected=pd.DataFrame(frame); saved=pd.read_csv(OUT/'metrics.csv'); keys=['replicate','scenario','method','cohort']
    ck(len(saved)==len(expected) and not saved.duplicated(keys).any(),'metrics coverage')
    a,b=[x.sort_values(keys).reset_index(drop=True) for x in [expected,saved]]
    ck(a[keys].equals(b[keys]),'metrics keys');close(a.iloc[:,4:],b.iloc[:,4:],'metrics values')
    summary=pd.read_csv(OUT/'summary.csv'); paired=pd.read_csv(OUT/'paired_differences.csv')
    ck(len(summary)==len(paired)==3*len(run_summary['methods'])*2*4,'summary coverage')
    for (sc,method,cohort),g in expected.groupby(['scenario','method','cohort']):
        ref=expected[(expected.scenario==sc)&(expected.method=='logistic_pooling')&(expected.cohort==cohort)].set_index('replicate')
        for metric in ['accuracy','macro_f1','log_loss','brier']:
            x=g[metric].to_numpy();n=len(x);sd=x.std(ddof=1);se=sd/np.sqrt(n);q=t.ppf(.975,n-1)
            row=summary[(summary.scenario==sc)&(summary.method==method)&(summary.cohort==cohort)&(summary.metric==metric)]
            ck(len(row)==1,'one summary per key'); row=row.iloc[0]
            close([row['n'],row['mean'],row['sd'],row['mcse'],row['low'],row['high']],[n,x.mean(),sd,se,x.mean()-q*se,x.mean()+q*se],'summary recomputation')
            dif=g.set_index('replicate')[metric].sort_index()-ref[metric].sort_index();ds=dif.std(ddof=1)/np.sqrt(n)
            row=paired[(paired.scenario==sc)&(paired.method==method)&(paired.cohort==cohort)&(paired.metric==metric)]
            ck(len(row)==1,'one paired summary per key');row=row.iloc[0]
            close([row['mean_difference'],row['mcse'],row['low'],row['high']],[dif.mean(),ds,dif.mean()-q*ds,dif.mean()+q*ds],'paired summary recomputation')
    ck(attempts==run_summary['attempts'] and retries==run_summary['retries'],'total fitting attempts')
    ck(run_summary['unresolved_final_fits']==0,'no unresolved fits')
    return attempts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-only',action='store_true');args=ap.parse_args()
    verify_data();attempts=None if args.data_only else verify_results()
    report={'status':'PASS','executed_assertions':COUNT,'datasets':len(SPEC['replicate_seeds'])*len(SPEC['scenarios']),'fit_attempts':attempts,'scope':'all datasets including external rows, role and loss accounting, masks, paired scenarios, input hashes, stored predictions and all four metrics, class recalls, metrics.csv, summary.csv, paired_differences.csv','external_validity':'not established by internal checks'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/('data_verification.json' if args.data_only else 'verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
