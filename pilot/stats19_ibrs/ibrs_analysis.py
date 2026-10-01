# STATS19 2024 IBRS quasi-experiment: same year, same database, two recording regimes.
# Steps: (1) severity priors by IBRS flag; (2) domain classifier IBRS vs non-IBRS (5-fold AUROC);
# (3) -1 missing-rate table by group; (4) cross-regime transfer fatal-vs-rest + intercept fix;
# (5) adjusted-severity priors vs raw priors.
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
EPS = 0.01            # additive smoothing for log loss
N_SEEDS = 30          # seeds for the intercept-fix experiment
DEV_SIZES = [36, 121]  # dev-set sizes drawn from the target group

DATA = '../stats19_2024/data'
OUT = {}
rng = np.random.default_rng(SEED)

# ---------- load ----------
col = pd.read_csv(f'{DATA}/dft-road-casualty-statistics-collision-2024.csv', low_memory=False)
veh = pd.read_csv(f'{DATA}/dft-road-casualty-statistics-vehicle-2024.csv', usecols=['collision_index', 'vehicle_type'])
col['ibrs'] = col['collision_injury_based'].astype(int)
sev_name = {1: 'fatal', 2: 'serious', 3: 'slight'}
col['sev'] = col['collision_severity'].map(sev_name)
N = len(col)
OUT['n_collisions'] = int(N)
OUT['ibrs_counts'] = col['ibrs'].value_counts().to_dict()
OUT['ibrs_share_pct'] = round(float(col['ibrs'].mean() * 100), 4)

# ---------- step 1: raw severity priors by group ----------
pri = {}
for g, name in [(1, 'IBRS'), (0, 'non_IBRS')]:
    s = col[col['ibrs'] == g]
    c = s['sev'].value_counts()
    pri[name] = {'n': int(len(s)),
                 'fatal_pct': round(c.get('fatal', 0) / len(s) * 100, 4),
                 'serious_pct': round(c.get('serious', 0) / len(s) * 100, 4),
                 'slight_pct': round(c.get('slight', 0) / len(s) * 100, 4)}
OUT['step1_raw_priors_pct'] = pri
OUT['step1_serious_gap_pp'] = round(pri['IBRS']['serious_pct'] - pri['non_IBRS']['serious_pct'], 4)

# ---------- shared feature engineering (scene-available, regime-comparable content) ----------
X = col[['collision_index', 'collision_severity', 'ibrs', 'number_of_vehicles', 'time',
         'road_type', 'junction_control', 'light_conditions', 'weather_conditions', 'road_surface_conditions',
         'urban_or_rural_area', 'day_of_week', 'junction_detail', 'pedestrian_crossing', 'carriageway_hazards',
         'special_conditions_at_site']].copy()
X['hour'] = pd.to_datetime(X['time'], format='%H:%M', errors='coerce').dt.hour

vt = veh[['collision_index', 'vehicle_type']].copy()
cats = {'has_hgv': [20, 21, 22, 23, 97, 98], 'has_motorcycle': [2, 3, 4, 5, 23, 97],
        'has_pedal_cycle': [1], 'has_bus': [10, 11], 'has_car': [8, 9]}
agg = pd.DataFrame({'collision_index': X['collision_index']})
for nm, codes in cats.items():
    m = vt[vt['vehicle_type'].isin(codes)].groupby('collision_index').size()
    agg[nm] = agg['collision_index'].map(m).fillna(0).clip(upper=1).astype(int)
agg['veh_count_tbl'] = agg['collision_index'].map(vt.groupby('collision_index').size())
X = X.merge(agg, on='collision_index', how='left')
for nm in cats:
    X[nm] = X[nm].fillna(0)

# content-only fields: comparable across regimes (no schema-split / administrative fields)
catA = ['road_type', 'junction_control', 'light_conditions', 'weather_conditions',
        'road_surface_conditions', 'urban_or_rural_area', 'day_of_week']
numA = ['number_of_vehicles', 'speed_limit', 'hour',
        'has_hgv', 'has_motorcycle', 'has_pedal_cycle', 'has_bus', 'has_car', 'veh_count_tbl']
# fields that are structurally -1 on one regime's form (schema split) -> as-stored discriminators
catB_extra = ['junction_detail', 'pedestrian_crossing', 'carriageway_hazards', 'special_conditions_at_site']

base = col[['speed_limit']].copy()
X['speed_limit'] = base['speed_limit']


def prep(d, cat_feats, num_feats):
    d = d.copy()
    for f in cat_feats:
        d[f] = d[f].apply(lambda v: 'miss' if (pd.isna(v) or v == -1) else str(int(v)))
    for f in ['speed_limit', 'hour']:
        d[f + '_miss'] = (d[f].isna() | (d[f] == -1)).astype(int)
        d[f] = d[f].replace(-1, np.nan)
    nf = num_feats + ['speed_limit_miss', 'hour_miss']
    d = d.dropna(subset=['speed_limit', 'hour'])
    return d, nf


def make_pipe(cat_feats, nf):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=50), cat_feats),
                             ('num', StandardScaler(), nf)])
    return Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=2000, C=1.0))])


skf = StratifiedKFold(5, shuffle=True, random_state=SEED)

# ---------- step 2: domain classifier IBRS(1) vs non-IBRS(0) ----------
y_dom = X['ibrs'].values
dom = {}
for tag, cat_feats in [('content_only', catA), ('as_stored_incl_schema_split', catA + catB_extra)]:
    d, nf = prep(X, cat_feats, numA)
    pipe = make_pipe(cat_feats, nf)
    proba = cross_val_predict(pipe, d[cat_feats + nf], d['ibrs'].values, cv=skf, method='predict_proba')[:, 1]
    yv = d['ibrs'].values
    dom[tag] = {'n': int(len(d)), 'AUROC': round(float(roc_auc_score(yv, proba)), 4),
                'AP': round(float(average_precision_score(yv, proba)), 4)}
OUT['step2_domain_classifier'] = dom

# ---------- step 3: -1 missing rates by group ----------
drop = {'collision_index', 'date', 'time', 'longitude', 'latitude', 'sev', 'ibrs',
        'location_easting_osgr', 'location_northing_osgr'}
rows = []
for f in col.columns:
    if f in drop:
        continue
    s = col[f]
    if s.dtype == object:
        miss = s.isna() | s.astype(str).str.strip().isin(['-1', '', 'nan'])
    else:
        miss = s.isna() | (s == -1)
    m1 = miss[col['ibrs'] == 1].mean() * 100
    m0 = miss[col['ibrs'] == 0].mean() * 100
    if m1 == 0 and m0 == 0:
        continue
    rows.append({'field': f, 'IBRS_missing_pct': round(float(m1), 4),
                 'nonIBRS_missing_pct': round(float(m0), 4),
                 'delta_pp': round(float(m1 - m0), 4)})
miss_tbl = pd.DataFrame(rows).sort_values('IBRS_missing_pct', ascending=False)
miss_tbl.to_csv('missing_by_group.csv', index=False)
OUT['step3_missing_top'] = miss_tbl.head(15).to_dict('records')

# ---------- metric helpers ----------


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logloss(y, p, pi):
    ps = (1 - EPS) * p + EPS * pi
    return float(-np.mean(y * np.log(ps) + (1 - y) * np.log(1 - ps)))


def top_decile(y, p):
    k = max(1, int(np.ceil(len(y) * 0.10)))
    top = y[np.argsort(-p)[:k]]
    return float(top.sum() / max(1, y.sum()))


def cal_slope(y, p):
    lr = LogisticRegression(C=1e8, max_iter=2000)
    lr.fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def eval_target(y, p, pi):
    return {'n': int(len(y)), 'pos_rate_pct': round(float(y.mean() * 100), 4),
            'AUROC': round(float(roc_auc_score(y, p)), 4),
            'AP': round(float(average_precision_score(y, p)), 4),
            'top_decile_coverage': round(top_decile(y, p), 4),
            'cal_in_large_pp': round(float((p.mean() - y.mean()) * 100), 4),
            'cal_slope': round(cal_slope(y, p), 4),
            'log_loss_eps001': round(logloss(y, p, pi), 4)}


# ---------- step 4: cross-regime transfer, fatal vs rest ----------
dA, nfA = prep(X, catA, numA)
y_fatal = (dA['collision_severity'] == 1).astype(int).values
groups = dA['ibrs'].values
transfer, fix_rows, ref = {}, [], {}

# in-domain reference: 5-fold CV within each regime
for g, name in [(0, 'nonIBRS'), (1, 'IBRS')]:
    m = groups == g
    pipe = make_pipe(catA, nfA)
    proba = cross_val_predict(pipe, dA.loc[m, catA + nfA], y_fatal[m], cv=skf, method='predict_proba')[:, 1]
    ref[name] = eval_target(y_fatal[m], proba, y_fatal[m].mean())
OUT['step4_indomain_reference'] = ref

for g_tr, name in [(0, 'nonIBRS_to_IBRS'), (1, 'IBRS_to_nonIBRS')]:
    g_te = 1 - g_tr
    mtr, mte = groups == g_tr, groups == g_te
    Xtr, ytr = dA.loc[mtr, catA + nfA], y_fatal[mtr]
    Xte, yte = dA.loc[mte, catA + nfA], y_fatal[mte]
    pi_tr = float(ytr.mean())
    pipe = make_pipe(catA, nfA)
    pipe.fit(Xtr, ytr)
    p_te = pipe.predict_proba(Xte)[:, 1]
    transfer[name] = eval_target(yte, p_te, pi_tr)
    transfer[name]['train_n'] = int(len(ytr))
    transfer[name]['train_pos_rate_pct'] = round(pi_tr * 100, 4)

    # intercept fix on small dev sets from the target group
    idx_te = np.where(mte)[0]
    for ds in DEV_SIZES:
        d_fix, cil_before, cil_after, ll_before, ll_after, dev_pos = [], [], [], [], [], []
        for s in range(N_SEEDS):
            seed = int(rng.integers(0, 2**31 - 1))
            r = np.random.default_rng(seed)
            dev = r.choice(idx_te, size=min(ds, len(idx_te)), replace=False)
            mask = np.zeros(len(yte), dtype=bool)
            mask[np.isin(idx_te, dev)] = True
            ev = ~mask
            dev_pos.append(int(yte[mask].sum()))
            z = logit(p_te)

            def dev_loss(delta):
                return logloss(yte[mask], 1 / (1 + np.exp(-(z[mask] + delta))), pi_tr)

            opt = minimize_scalar(dev_loss, bounds=(-8, 8), method='bounded')
            delta = float(opt.x)
            p_fix = 1 / (1 + np.exp(-(z + delta)))
            d_fix.append(delta)
            cil_before.append((p_te[ev].mean() - yte[ev].mean()) * 100)
            cil_after.append((p_fix[ev].mean() - yte[ev].mean()) * 100)
            ll_before.append(logloss(yte[ev], p_te[ev], pi_tr))
            ll_after.append(logloss(yte[ev], p_fix[ev], pi_tr))
            fix_rows.append({'direction': name, 'dev_size': ds, 'seed': seed,
                             'dev_fatal_count': int(yte[mask].sum()),
                             'delta': round(delta, 4),
                             'cil_before_pp': round(cil_before[-1], 4), 'cil_after_pp': round(cil_after[-1], 4),
                             'll_before': round(ll_before[-1], 6), 'll_after': round(ll_after[-1], 6)})

        def ms(v):
            return {'mean': round(float(np.mean(v)), 4), 'sd': round(float(np.std(v)), 4)}

        transfer[name][f'fix_dev{ds}'] = {
            'delta': ms(d_fix), 'cil_before_pp': ms(cil_before), 'cil_after_pp': ms(cil_after),
            'll_before': ms(ll_before), 'll_after': ms(ll_after),
            'dev_fatal_count_mean': round(float(np.mean(dev_pos)), 3)}

OUT['step4_transfer'] = transfer
pd.DataFrame(fix_rows).to_csv('intercept_fix.csv', index=False)

# ---------- step 5: adjusted-severity priors ----------
adj = {}
for g, name in [(1, 'IBRS'), (0, 'non_IBRS')]:
    s = col[col['ibrs'] == g]
    raw_sev = (s['collision_severity'] == 2).mean() * 100
    raw_sli = (s['collision_severity'] == 3).mean() * 100
    adj_sev = s['collision_adjusted_severity_serious'].mean() * 100  # sum(col)/n, fatal rows = 0
    adj_sli = s['collision_adjusted_severity_slight'].mean() * 100
    adj[name] = {'n': int(len(s)),
                 'raw_serious_pct': round(raw_sev, 4), 'adj_serious_pct': round(adj_sev, 4),
                 'adj_minus_raw_serious_pp': round(adj_sev - raw_sev, 4),
                 'raw_slight_pct': round(raw_sli, 4), 'adj_slight_pct': round(adj_sli, 4),
                 'adj_minus_raw_slight_pp': round(adj_sli - raw_sli, 4)}
OUT['step5_adjusted_priors_pct'] = adj
OUT['step5_serious_gap_pp'] = {'raw': round(adj['IBRS']['raw_serious_pct'] - adj['non_IBRS']['raw_serious_pct'], 4),
                               'adjusted': round(adj['IBRS']['adj_serious_pct'] - adj['non_IBRS']['adj_serious_pct'], 4)}

with open('results.json', 'w') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)

print(json.dumps(OUT, ensure_ascii=False, indent=2))
