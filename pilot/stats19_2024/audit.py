# STATS19 2024 audit: severity priors, missingness gradient, severity classifier, cost scan
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = 'data'
OUT = {}

# ---------- load ----------
col = pd.read_csv(f'{DATA}/dft-road-casualty-statistics-collision-2024.csv', low_memory=False)
veh = pd.read_csv(f'{DATA}/dft-road-casualty-statistics-vehicle-2024.csv', low_memory=False)
sev_name = {1: 'fatal', 2: 'serious', 3: 'slight'}
col['sev'] = col['collision_severity'].map(sev_name)

# ---------- step 1: priors + majority baseline ----------
counts = col['sev'].value_counts()
priors = (counts / len(col) * 100).round(3)
maj = counts.max() / len(col)
OUT['priors_pct'] = priors.to_dict()
OUT['majority_baseline_acc'] = round(float(maj), 4)
OUT['n_collisions'] = int(len(col))

# ---------- step 2: missingness (-1) by severity, all collision fields ----------
drop = {'collision_index', 'date', 'time', 'longitude', 'latitude', 'sev',
        'location_easting_osgr', 'location_northing_osgr'}
fields = [c for c in col.columns if c not in drop and col[c].dtype != object]
# convert -1 to NaN-like mask; strings/objects handled separately
sev_order = {'fatal': 1, 'serious': 2, 'slight': 3}
rows = []
for f in fields:
    sub = col[[f, 'sev']].copy()
    if sub[f].dtype == object:
        miss = sub[f].isna() | sub[f].astype(str).str.strip().isin(['-1', '', 'nan'])
    else:
        miss = sub[f].isna() | (sub[f] == -1)
    g = miss.groupby(col['sev']).mean() * 100
    if g.sum() == 0:
        continue
    rates = [round(float(g.get(s, np.nan)), 3) for s in ['fatal', 'serious', 'slight']]
    # Spearman: severity rank (fatal=1 most severe) vs missing rate
    rho, p = stats.spearmanr([1, 2, 3], rates)
    rows.append({'field': f, 'fatal_%': rates[0], 'serious_%': rates[1], 'slight_%': rates[2],
                 'spearman_rho(sev,missing)': round(float(rho), 3),
                 'p_value': round(float(p), 4)})
miss_tbl = pd.DataFrame(rows).sort_values('fatal_%', ascending=False)
miss_tbl.to_csv('missing_by_severity.csv', index=False)

# ---------- features for steps 4/5 ----------
feat_col = ['road_type', 'speed_limit', 'junction_detail', 'light_conditions',
            'weather_conditions', 'road_surface_conditions', 'urban_or_rural_area',
            'day_of_week']
X = col[feat_col + ['collision_index', 'collision_severity', 'number_of_vehicles', 'time']].copy()
X['hour'] = pd.to_datetime(X['time'], format='%H:%M', errors='coerce').dt.hour

# vehicle aggregation
vt = veh[['collision_index', 'vehicle_type']].copy()
cats = {
    'has_hgv': [20, 21, 22, 23, 97, 98],          # HGV variants
    'has_motorcycle': [2, 3, 4, 5, 23, 97],        # motorcycle/moped incl scooter
    'has_pedal_cycle': [1],
    'has_bus': [10, 11],
    'has_car': [8, 9],
}
agg = pd.DataFrame({'collision_index': X['collision_index']})
for name, codes in cats.items():
    m = vt[vt['vehicle_type'].isin(codes)].groupby('collision_index').size()
    agg[name] = agg['collision_index'].map(m).fillna(0).clip(upper=1).astype(int)
nveh = vt.groupby('collision_index').size()
agg['veh_count_tbl'] = agg['collision_index'].map(nveh)
X = X.merge(agg, on='collision_index', how='left')
X['has_hgv'] = X['has_hgv'].fillna(0); X['has_motorcycle'] = X['has_motorcycle'].fillna(0)
X['has_pedal_cycle'] = X['has_pedal_cycle'].fillna(0)
X['has_bus'] = X['has_bus'].fillna(0); X['has_car'] = X['has_car'].fillna(0)

cat_feats = feat_col
num_feats = ['number_of_vehicles', 'hour', 'has_hgv', 'has_motorcycle',
             'has_pedal_cycle', 'has_bus', 'has_car']
# -1 handling: make -1 its own category string for cat feats; missing indicator for numeric
for f in cat_feats:
    X[f] = X[f].apply(lambda v: 'miss' if (pd.isna(v) or v == -1) else str(int(v)))
for f in ['speed_limit', 'hour']:
    X[f + '_miss'] = (X[f].isna() | (X[f] == -1)).astype(int)
    X[f] = X[f].replace(-1, np.nan)
num_feats += ['speed_limit_miss', 'hour_miss']
X = X.dropna(subset=['hour'])
X['hour'] = X['hour'].astype(int)

pre = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
    ('num', StandardScaler(), num_feats),
])
skf = StratifiedKFold(5, shuffle=True, random_state=42)


def run_task(mask_pos, label):
    d = X[mask_pos | (X['collision_severity'] == 3)].copy() if label == 'fs_vs_slight' else X.copy()
    y = (d['collision_severity'].isin(mask_pos_val)).astype(int) if False else None
    return d


results = {}
for task, (pos_mask, pos_name) in {
    'fatal_vs_rest': (X['collision_severity'] == 1, 'fatal'),
    'fats_vs_slight': (X['collision_severity'].isin([1, 2]), 'fatal+serious'),
}.items():
    if task == 'fatal_vs_rest':
        d = X.copy(); y = (d['collision_severity'] == 1).astype(int).values
    else:
        d = X[X['collision_severity'].isin([1, 2, 3])].copy()
        y = (d['collision_severity'].isin([1, 2])).astype(int).values
    Xm = d[cat_feats + num_feats]
    pipe = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=2000, C=1.0))])
    proba = cross_val_predict(pipe, Xm, y, cv=skf, method='predict_proba')[:, 1]
    auroc = roc_auc_score(y, proba)
    ap = average_precision_score(y, proba)
    n = len(y); k = max(1, int(np.ceil(n * 0.10)))
    top = np.argsort(-proba)[:k]
    tdc = y[top].sum() / y.sum() if y.sum() else np.nan
    results[task] = {'n': int(n), 'pos_rate': round(float(y.mean()), 4),
                     'AUROC': round(float(auroc), 4), 'AP': round(float(ap), 4),
                     'top_decile_coverage': round(float(tdc), 4)}

# ---------- step 5: cost scan on fatal_vs_rest out-of-fold scores ----------
d = X.copy(); y = (d['collision_severity'] == 1).astype(int).values
Xm = d[cat_feats + num_feats]
pipe = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=2000, C=1.0))])
proba = cross_val_predict(pipe, Xm, y, cv=skf, method='predict_proba')[:, 1]
n = len(y); P = y.sum()
cost_rows = []
for c in range(1, 51):
    t = 1.0 / (c + 1)
    pred = proba >= t
    FP = int((pred & (y == 0)).sum()); FN = int((~pred & (y == 1)).sum())
    model_cost = (c * FN + FP) / n
    allpos = (c * 0 + (n - P)) / n            # mark all severe: no FN, all negatives FP
    allneg = (c * P + 0) / n                  # mark none: all positives FN
    cost_rows.append({'c': c, 'thr': round(t, 4), 'model': round(model_cost, 5),
                      'all_severe': round(allpos, 5), 'none': round(allneg, 5),
                      'beats_all_severe': model_cost < allpos, 'beats_none': model_cost < allneg})
ct = pd.DataFrame(cost_rows)
ct.to_csv('cost_scan_fatal.csv', index=False)
wins = ct[ct['beats_all_severe'] & ct['beats_none']]
OUT['cost_window'] = (int(wins['c'].min()), int(wins['c'].max())) if len(wins) else None
# break-even points
b1 = ct[(~ct['beats_none'])]['c'].max()          # last c where model loses to none
b2 = ct[(ct['beats_all_severe'])]['c'].min() if ct['beats_all_severe'].any() else None
OUT['break_points'] = {'last_c_beats_none': int(b1) if pd.notna(b1) else None,
                       'first_c_beats_allsevere': int(b2) if b2 else None}
OUT['tasks'] = results
OUT['missing_gradient'] = miss_tbl.head(20).to_dict('records')

with open('results.json', 'w') as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print(json.dumps(OUT['tasks'], indent=2))
print('priors:', OUT['priors_pct'], 'maj acc:', OUT['majority_baseline_acc'])
print('cost window:', OUT['cost_window'], OUT['break_points'])
print(miss_tbl.head(15).to_string(index=False))
