# -*- coding: utf-8 -*-
"""
FARS vs CRSS 2022 记录机制审计
(a) 共有字段 unknown 率对比（仅限两系统均为 fatal 的案例）
(b) 逻辑回归域分类器：FARS-fatal vs CRSS-fatal（共有字段，5 折 CV AUROC）
附：CRSS 内部严重度梯度（informative recording 对照）
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42
FARS_ACC = 'fars2022/FARS2022NationalCSV/accident.csv'
CRSS_ACC = 'crss2022/CRSS2022CSV/accident.csv'
FARS_VEH = 'fars2022/FARS2022NationalCSV/vehicle.csv'
CRSS_VEH = 'crss2022/CRSS2022CSV/vehicle.csv'

# ---------------- 编码约定（NHTSA FARS/CRSS 2022 Analytical Reference Guide，两系统编码一致） ----------------
# 98 = Not Reported, 99 = Reported as Unknown（ weather 等两位数字段）; 8 = Not Reported, 9 = Reported as Unknown（个位数字段）
# 注意：LGT_COND=6 (Dark - Unknown Lighting)、REL_ROAD=6 (Off Roadway-Location Unknown) 是实质性类别，不算缺失
UNKNOWN_CODES = {
    'LGT_COND': {8, 9}, 'WEATHER': {98, 99}, 'MAN_COLL': {98, 99},
    'REL_ROAD': {98, 99}, 'TYP_INT': {98, 99}, 'HOUR': {99}, 'DAY_WEEK': {9},
    'SCH_BUS': {9}, 'HARM_EV': {98, 99}, 'BODY_TYP': {98, 99},
}
FIELDS = ['LGT_COND', 'WEATHER', 'MAN_COLL', 'REL_ROAD', 'TYP_INT', 'HOUR']

fa = pd.read_csv(FARS_ACC)
ca = pd.read_csv(CRSS_ACC)
cf = ca[ca['MAX_SEV'] == 4].copy()          # CRSS fatal（MAX_SEV=4, Fatal Injury K）

results = {}

# ---------------- 0. 严重度先验（CRSS 2022 样本） ----------------
sev_prior = ca['MAX_SEVNAME'].value_counts()
maj = sev_prior.max() / sev_prior.sum()
results['severity_prior'] = {'n': int(len(ca)),
                             'distribution': {str(k): int(v) for k, v in sev_prior.items()},
                             'majority_class': str(sev_prior.idxmax()),
                             'majority_baseline_acc': round(float(maj), 4)}
print('== CRSS 严重度先验 ==')
print(sev_prior)
print(f"多数类基线 = {maj:.4f}")

# ---------------- 1a. 共有字段 unknown 率：FARS-fatal vs CRSS-fatal ----------------
def unk_rate(df, col, weight=None):
    codes = UNKNOWN_CODES[col]
    mask = df[col].isin(codes)
    if weight is None:
        return mask.mean() * 100, int(mask.sum()), int(len(df))
    w = df[weight]
    return (w[mask].sum() / w.sum() * 100), float(w[mask].sum()), float(w.sum())

rows = []
for col in FIELDS + ['SCH_BUS']:
    fr, fn_, fd = unk_rate(fa, col)
    cr, cn_, cd = unk_rate(cf, col)
    crw, cnw, cdw = unk_rate(cf, col, weight='WEIGHT')
    rows.append({'field': col, 'FARS_fatal_pct': round(fr, 2), 'FARS_fatal_n': f"{fn_}/{fd}",
                 'CRSS_fatal_pct': round(cr, 2), 'CRSS_fatal_n': f"{cn_}/{cd}",
                 'CRSS_fatal_pct_weighted': round(crw, 2)})
# 城乡：FARS RUR_URB(6/8/9=不可用/未报/未知) vs CRSS URBANICITY（由 NHTSA 派生，无 unknown）
f_mask = fa['RUR_URB'].isin({6, 8, 9})
c_mask = cf['URBANICITY'].isin({3, 8, 9})  # 仅作形式检查
rows.append({'field': 'RUR_URB/URBANICITY', 'FARS_fatal_pct': round(f_mask.mean() * 100, 2),
             'FARS_fatal_n': f"{int(f_mask.sum())}/{len(fa)}",
             'CRSS_fatal_pct': round(c_mask.mean() * 100, 2),
             'CRSS_fatal_n': f"{int(c_mask.sum())}/{len(cf)}",
             'CRSS_fatal_pct_weighted': round(cf.loc[c_mask, 'WEIGHT'].sum() / cf['WEIGHT'].sum() * 100, 2)})
# 施工区：空白=不适用（非施工区），未知=类型未知(4)。空白不计入缺失。
fa_wz = fa['WRK_ZONE'].notna().mean() * 100
cf_wz = cf['WRK_ZONE'].notna().mean() * 100
fa_wz4 = (fa['WRK_ZONE'] == 4).sum() / fa['WRK_ZONE'].notna().sum() * 100
cf_wz4 = (cf['WRK_ZONE'] == 4).sum() / cf['WRK_ZONE'].notna().sum() * 100
rows.append({'field': 'WRK_ZONE(present, %; blank=不适用)', 'FARS_fatal_pct': round(fa_wz, 2),
             'FARS_fatal_n': f"{fa['WRK_ZONE'].notna().sum()}/{len(fa)}",
             'CRSS_fatal_pct': round(cf_wz, 2),
             'CRSS_fatal_n': f"{cf['WRK_ZONE'].notna().sum()}/{len(cf)}",
             'CRSS_fatal_pct_weighted': np.nan})
results['workzone_type_unknown_pct_among_WZ'] = {'FARS': round(float(fa_wz4), 2), 'CRSS': round(float(cf_wz4), 2)}

# 车型：fatal 事故涉及车辆的 BODY_TYP unknown(98/99) 率
fv = pd.read_csv(FARS_VEH, usecols=['ST_CASE', 'BODY_TYP', 'UNITTYPE'])
cvv = pd.read_csv(CRSS_VEH, usecols=['CASENUM', 'BODY_TYP', 'UNITTYPE'])
fv_f = fv[fv['ST_CASE'].isin(fa['ST_CASE'])]
cv_f = cvv[cvv['CASENUM'].isin(cf['CASENUM'])]
fb = fv_f['BODY_TYP'].isin({98, 99}).mean() * 100
cb = cv_f['BODY_TYP'].isin({98, 99}).mean() * 100
rows.append({'field': 'BODY_TYP(车辆级,98/99)', 'FARS_fatal_pct': round(float(fb), 2),
             'FARS_fatal_n': f"{int(fv_f['BODY_TYP'].isin({98,99}).sum())}/{len(fv_f)}",
             'CRSS_fatal_pct': round(float(cb), 2),
             'CRSS_fatal_n': f"{int(cv_f['BODY_TYP'].isin({98,99}).sum())}/{len(cv_f)}",
             'CRSS_fatal_pct_weighted': np.nan})

unk_tbl = pd.DataFrame(rows)
unk_tbl.to_csv('table1_unknown_rates_fatal.csv', index=False)
print('\n== 表1 unknown 率（fatal） ==')
print(unk_tbl.to_string(index=False))

# ---------------- 1b. CRSS 内部严重度梯度（informative recording 对照） ----------------
order = ['No Apparent Injury (O)', 'Possible Injury (C)', 'Suspected Minor Injury (B)',
         'Suspected Serious Injury (A)', 'Fatal Injury (K)']
grad_rows = []
for col in ['LGT_COND', 'WEATHER', 'MAN_COLL', 'HOUR']:
    codes = UNKNOWN_CODES[col]
    r = {'field': col}
    for s in order:
        sub = ca[ca['MAX_SEVNAME'] == s]
        r[s.split('(')[0].strip()] = round(sub[col].isin(codes).mean() * 100, 2)
    sub = ca[ca['MAX_SEVNAME'] == 'Unknown/Not Reported']
    r['Unknown sev'] = round(sub[col].isin(codes).mean() * 100, 2)
    grad_rows.append(r)
grad_tbl = pd.DataFrame(grad_rows)
grad_tbl.to_csv('table2_crss_severity_gradient.csv', index=False)
print('\n== 表2 CRSS 内部严重度梯度：unknown% ==')
print(grad_tbl.to_string(index=False))

# ---------------- 2. 域分类器：FARS-fatal vs CRSS-fatal ----------------
def harm_group(x):
    if x in (98, 99) or pd.isna(x): return 'unk'
    if x == 1: return 'rollover'
    if x == 8: return 'ped'
    if x == 9: return 'bike'
    if x == 12: return 'mv_transport'
    if x == 14: return 'parked'
    if x == 24: return 'attenuator'
    if 30 <= x <= 34: return 'pole_support'
    if 35 <= x <= 45: return 'fixed_object'
    if x in (46, 47): return 'working_tow'
    if 60 <= x <= 69: return 'animal_wild'
    if 70 <= x <= 79: return 'animal_farm'
    return 'other'

def unit_flags(veh, key):
    bt = veh['BODY_TYP']
    flags = pd.DataFrame({
        key: veh[key],
        'any_moto': bt.between(80, 89).groupby(veh[key]).any(),
        'any_truck': bt.between(60, 79).groupby(veh[key]).any(),
        'any_bus': bt.between(50, 59).groupby(veh[key]).any(),
    }).reset_index()
    return flags

def crash_frame(df, is_fars):
    d = pd.DataFrame({
        'LGT_COND': df['LGT_COND'], 'WEATHER': df['WEATHER'], 'MAN_COLL': df['MAN_COLL'],
        'REL_ROAD': df['REL_ROAD'], 'TYP_INT': df['TYP_INT'], 'DAY_WEEK': df['DAY_WEEK'],
        'SCH_BUS': df['SCH_BUS'], 'HOUR': df['HOUR'],
        'HARM_EV': df['HARM_EV'].map(harm_group),
        'VE_TOTAL': pd.cut(df['VE_TOTAL'], [0, 1, 2, 99], labels=['1', '2', '3+']),
        'WRKZONE': np.where(df['WRK_ZONE'].isna(), 'no', np.where(df['WRK_ZONE'] == 4, 'wz_typeunk', 'yes')),
        'key': df['ST_CASE'] if is_fars else df['CASENUM'],
    })
    if is_fars:
        d['RURURB'] = np.select([df['RUR_URB'] == 1, df['RUR_URB'] == 2], ['rural', 'urban'], default='unk')
        veh = fv[['ST_CASE', 'BODY_TYP']].rename(columns={'ST_CASE': 'key'})
    else:
        d['RURURB'] = np.select([df['URBANICITY'] == 1, df['URBANICITY'] == 2], ['rural', 'urban'], default='unk')
        veh = cvv[['CASENUM', 'BODY_TYP']].rename(columns={'CASENUM': 'key'})
    bt = veh['BODY_TYP']
    g = pd.DataFrame({'key': veh['key'],
                      'any_moto': bt.between(80, 89), 'any_truck': bt.between(60, 79),
                      'any_bus': bt.between(50, 59)}).groupby('key').any().reset_index()
    d = d.merge(g, on='key', how='left')
    d['any_moto'] = d['any_moto'].fillna(False).astype(str)
    d['any_truck'] = d['any_truck'].fillna(False).astype(str)
    d['any_bus'] = d['any_bus'].fillna(False).astype(str)
    return d.drop(columns=['key'])

XF = crash_frame(fa, True)
XC = crash_frame(cf, False)
CAT = ['LGT_COND', 'WEATHER', 'MAN_COLL', 'REL_ROAD', 'TYP_INT', 'DAY_WEEK', 'SCH_BUS',
       'HOUR', 'HARM_EV', 'VE_TOTAL', 'WRKZONE', 'RURURB', 'any_moto', 'any_truck', 'any_bus']

def cv_auroc(X, y, seed=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, prs = [], []
    for tr, te in skf.split(X, y):
        enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        Xt = enc.fit_transform(X.iloc[tr])
        Xe = enc.transform(X.iloc[te])
        clf = LogisticRegression(max_iter=5000)
        clf.fit(Xt, y[tr])
        p = clf.predict_proba(Xe)[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
    return aucs, prs

def run_variant(Xf, Xc, tag):
    n = len(Xc)
    rng = np.random.RandomState(RANDOM_STATE)
    sub = rng.choice(len(Xf), size=n, replace=False)
    X = pd.concat([Xf.iloc[sub], Xc], ignore_index=True)
    y = np.array([1] * n + [0] * n)   # 1=FARS, 0=CRSS
    aucs, prs = cv_auroc(X, y)
    results[f'domain_{tag}'] = {
        'n_per_class': n,
        'AUROC_mean': round(float(np.mean(aucs)), 4), 'AUROC_sd': round(float(np.std(aucs)), 4),
        'folds': [round(float(a), 4) for a in aucs],
        'AP_mean': round(float(np.mean(prs)), 4)}
    print(f"\n== 域分类器 [{tag}] n={n}/类 ==")
    print(f"AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}  folds={[round(a,3) for a in aucs]}  AP={np.mean(prs):.3f}")
    return X, y

# 变体 A：unknown 保留为独立类别（默认）
XA, yA = run_variant(XF, XC, 'A_unknown_as_level')
# 变体 C：complete-case（剔除 6 个核心字段任一为 unknown 的事故）
core = ['LGT_COND', 'WEATHER', 'MAN_COLL', 'REL_ROAD', 'TYP_INT', 'HOUR']
f_core_unk = XF[core].isin([8, 9, 98, 99]).any(axis=1)
c_core_unk = XC[core].isin([8, 9, 98, 99]).any(axis=1)
run_variant(XF[~f_core_unk], XC[~c_core_unk], 'C_complete_case')

# 全量 FARS（不抽样）对照
Xall = pd.concat([XF, XC], ignore_index=True)
yall = np.array([1] * len(XF) + [0] * len(XC))
aucs, prs = cv_auroc(Xall, yall)
results['domain_full_FARS'] = {'AUROC_mean': round(float(np.mean(aucs)), 4),
                               'AUROC_sd': round(float(np.std(aucs)), 4),
                               'AP_mean': round(float(np.mean(prs)), 4)}
print(f"\n== 域分类器 [全量 FARS {len(XF)} vs CRSS-fatal {len(XC)}] AUROC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

# 系数（变体 A，全数据拟合，用于解释方向）
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
Z = enc.fit_transform(XA)
clf = LogisticRegression(max_iter=5000).fit(Z, yA)
names = enc.get_feature_names_out(CAT)
coef = pd.Series(clf.coef_[0], index=names).sort_values()
top = pd.concat([coef.head(12), coef.tail(12)])
top.to_csv('table3_domain_classifier_coef.csv')
print('\n== 域分类器 top 系数（负=偏向 CRSS，正=偏向 FARS） ==')
print(top.to_string())

# ---------------- 3. fatal 案例构成对照（解释用） ----------------
comp = pd.DataFrame({
    'FARS_fatal_%': [round(fa['VE_TOTAL'].eq(1).mean() * 100, 1),
                     round(XF['any_moto'].eq('True').mean() * 100, 1),
                     round(XF['any_truck'].eq('True').mean() * 100, 1),
                     round(XF['any_bus'].eq('True').mean() * 100, 1),
                     round((XF['RURURB'] == 'rural').mean() * 100, 1)],
    'CRSS_fatal_%': [round(cf['VE_TOTAL'].eq(1).mean() * 100, 1),
                     round(XC['any_moto'].eq('True').mean() * 100, 1),
                     round(XC['any_truck'].eq('True').mean() * 100, 1),
                     round(XC['any_bus'].eq('True').mean() * 100, 1),
                     round((XC['RURURB'] == 'rural').mean() * 100, 1)],
}, index=['单车事故', '涉及摩托车', '涉及中重型货车', '涉及巴士', '乡村道路'])
comp.to_csv('table4_fatal_composition.csv')
print('\n== 表4 fatal 案例构成 ==')
print(comp.to_string())

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('\nsaved: table1-4 csv, results.json')
