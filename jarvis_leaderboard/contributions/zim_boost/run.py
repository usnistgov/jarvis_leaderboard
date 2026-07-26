"""zim_boost -- contribution to JARVIS-Leaderboard benchmark AI/SinglePropertyPrediction/supercon_chem_Tc

Self-contained end-to-end reproduction. Running this file regenerates
`AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv.zip` and prints the test MAE.

    python run.py            # ~10 min on a laptop CPU

--------------------------------------------------------------------------------------
METHOD -- zero-inflated median gating
--------------------------------------------------------------------------------------
24.4% of SuperCon labels are exactly Tc = 0 (measured, not superconducting). MAE is
minimised by the conditional MEDIAN, not the conditional mean that a squared-error forest
produces. For a mixture with an atom of mass p0 at zero and a continuous positive part,

    F(t) = p0 + (1-p0) F+(t)   =>   median = 0  whenever  p0 >= 1/2 ,

so the Bayes act under MAE is to emit an exact zero on that region. Concretely the current
leaderboard #1 (matminer_rf) predicts a mean of 6.95 K on the 760 test rows whose true Tc
is 0, which alone is a third of its total error. Gating an ensemble on P(Tc=0|x) removes
most of it; the gate threshold (0.35) is tuned by cross-validation around the theoretical
1/2, absorbing classifier miscalibration.

    pred(x) = 0                   if P(Tc=0|x) >= 0.35
            = mean_k R_k(x)       otherwise

FEATURES: matminer composition featurizers (Magpie ElementProperty, Stoichiometry,
ValenceOrbital, IonProperty, ElementFraction, TMetalFraction, BandCenter) plus
superconductivity-physics descriptors used by no existing entry on this benchmark:
  * cuprate hole doping p per CuO2 plane, recovered from composition alone by charge
    balance against fixed oxidation states, with the Presland-Tallon dome 1-82.6(p-0.16)^2
    (hits 27.7% of rows -- exactly the high-Tc regime that dominates the error budget);
  * Matthias valence-electron-count rules (e/a peaks near 4.7 and 6.5);
  * isotope-effect phonon proxies <M^-1/2>; block fractions; family indicators.

PROTOCOL: every choice (feature set, model family, ensemble members, threshold) was made by
5-fold CV (seed 42) on the official TRAIN split only. The official TEST split was evaluated
exactly once. CV MAE 4.5917 vs 5.2769 for a byte-faithful reproduction of matminer_rf under
the identical CV; test MAE 4.2619 vs 4.8511 published.

DISCLOSURE: 3.32% of test reduced formulas also occur in train. That is a property of the
official split shared by every entry; nothing here exploits it beyond what any model sees.
"""
import os, re, json, zipfile, warnings
import numpy as np
warnings.filterwarnings('ignore')

from jarvis.db.figshare import data as jdata
from pymatgen.core import Composition, Element
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (ElementProperty, Stoichiometry, ValenceOrbital,
                                              IonProperty, ElementFraction, TMetalFraction,
                                              BandCenter)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

SEED, THRESH = 42, 0.35
HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = 'supercon_chem_Tc.json.zip'
CSV = 'AI-SinglePropertyPrediction-Tc-supercon_chem-test-mae.csv'

FIXED_OX = {'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1, 'Ag': 1, 'Be': 2, 'Mg': 2,
            'Ca': 2, 'Sr': 2, 'Ba': 2, 'Zn': 2, 'Cd': 2, 'Hg': 2, 'Sc': 3, 'Y': 3, 'La': 3,
            'Pr': 3, 'Nd': 3, 'Pm': 3, 'Sm': 3, 'Gd': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3,
            'Lu': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Eu': 2, 'Yb': 3, 'Tb': 3, 'Ce': 4, 'Th': 4,
            'Zr': 4, 'Hf': 4, 'Ti': 4, 'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 2, 'Bi': 3, 'Tl': 3,
            'Sb': 3, 'As': -3, 'O': -2, 'F': -1, 'Cl': -1, 'Br': -1, 'I': -1, 'S': -2,
            'Se': -2, 'Te': -2, 'N': -3, 'C': -4, 'P': -3}
PHYS = ['phys_cu_valence', 'phys_hole_doping_p', 'phys_dome', 'phys_dome_abs_dev',
        'phys_is_cuprate', 'phys_cu_per_fu', 'phys_o_per_cu', 'phys_o_excess',
        'phys_is_pnictide', 'phys_is_chalcogenide', 'phys_is_hydride', 'phys_is_carbide',
        'phys_ea_matthias', 'phys_matthias_peak', 'phys_inv_sqrt_mass', 'phys_debye_proxy',
        'phys_n_elements', 'phys_max_frac', 'phys_frac_entropy', 'phys_is_oxide',
        'phys_heavy_fermion', 'phys_frac_dblock', 'phys_frac_fblock']
ELEMENTS = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br',
            'Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te',
            'I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm',
            'Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra','Ac','Th','Pa','U','Np','Pu']


def lgbp(**kw):
    p = dict(learning_rate=0.03, num_leaves=63, min_child_samples=10, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0, random_state=SEED,
             n_jobs=-1, verbose=-1)
    p.update(kw); return p


def clean(f):
    """SuperCon formula strings carry non-stoichiometric junk: trailing formal charges,
    '=z' placeholders, stray '!1.5', negative amounts."""
    g = re.sub(r'[=!][a-zA-Z0-9.]*$', '', f)
    g = re.sub(r'[+\-]\d+(\.\d+)?$', '', g)
    return re.sub(r'([A-Z][a-z]?)-\d+(\.\d+)?', r'\1', g)


def valence_per_atom(comp):
    tot = n = 0.0
    for el, amt in comp.get_el_amt_dict().items():
        try:
            v = Element(el).full_electronic_structure
            nmax = max(t[0] for t in v)
            tot += sum(t[2] for t in v if t[0] == nmax or t[1] == 'd') * amt
            n += amt
        except Exception:
            pass
    return tot / n if n else np.nan


def physics_row(comp):
    d = comp.get_el_amt_dict(); tot = sum(d.values())
    fr = {k: v / tot for k, v in d.items()}
    o = {k: np.nan for k in PHYS}
    o['phys_n_elements'] = len(d); o['phys_max_frac'] = max(fr.values())
    q = np.array(list(fr.values())); o['phys_frac_entropy'] = float(-(q * np.log(q + 1e-12)).sum())
    o['phys_is_oxide'] = 1.0 if 'O' in d else 0.0
    o['phys_is_hydride'] = fr.get('H', 0.0); o['phys_is_carbide'] = fr.get('C', 0.0)
    o['phys_is_pnictide'] = float(any(p in d for p in ('As', 'P', 'Sb')) and
                                  any(t in d for t in ('Fe', 'Ni', 'Co', 'Mn', 'Cr')))
    o['phys_is_chalcogenide'] = float(any(c in d for c in ('S', 'Se', 'Te')))
    o['phys_heavy_fermion'] = sum(fr.get(e, 0.0) for e in ('Ce', 'U', 'Yb', 'Np', 'Pu'))
    fd = ff = 0.0
    for el, f in fr.items():
        try:
            b = Element(el).block
            fd += f if b == 'd' else 0.0; ff += f if b == 'f' else 0.0
        except Exception:
            pass
    o['phys_frac_dblock'], o['phys_frac_fblock'] = fd, ff
    m, a = [], []
    for el, amt in d.items():
        try:
            m.append(Element(el).atomic_mass); a.append(amt)
        except Exception:
            pass
    if m:
        m = np.array(m, float); a = np.array(a, float)
        o['phys_inv_sqrt_mass'] = float((a * m ** -0.5).sum() / a.sum())
        o['phys_debye_proxy'] = float(np.sqrt(1.0 / (a * m).sum() * a.sum()))
    ea = valence_per_atom(comp); o['phys_ea_matthias'] = ea
    if np.isfinite(ea):
        o['phys_matthias_peak'] = float(min(abs(ea - 4.7), abs(ea - 6.5)))
    if 'Cu' in d and 'O' in d:
        n_cu = d['Cu']; known = 0.0; unknown = False
        for el, amt in d.items():
            if el == 'Cu':
                continue
            if el in FIXED_OX:
                known += FIXED_OX[el] * amt
            else:
                unknown = True
        if not unknown and n_cu > 0:
            cu_val = -known / n_cu; p = cu_val - 2.0
            o['phys_cu_valence'] = cu_val; o['phys_hole_doping_p'] = p
            o['phys_dome'] = float(1.0 - 82.6 * (p - 0.16) ** 2)
            o['phys_dome_abs_dev'] = float(abs(p - 0.16))
        o['phys_is_cuprate'] = 1.0; o['phys_cu_per_fu'] = float(n_cu)
        o['phys_o_per_cu'] = float(d['O'] / n_cu) if n_cu else np.nan
        o['phys_o_excess'] = float(d['O'] - round(d['O']))
    else:
        o['phys_is_cuprate'] = 0.0
    return [o[k] for k in PHYS]


def main():
    bench = os.path.join(HERE, BENCH)
    if not os.path.exists(bench):
        import urllib.request
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/atomgptlab/jarvis_leaderboard/main/'
            'jarvis_leaderboard/benchmarks/AI/SinglePropertyPrediction/' + BENCH, bench)
    bm = json.loads(zipfile.ZipFile(bench).read('supercon_chem_Tc.json'))
    byid = {str(r['id']): r for r in jdata('supercon_chem')}
    ids = list(bm['train']) + list(bm['test'])
    y = np.array([byid[i]['Tc'] for i in ids], float)
    is_test = np.array([False] * len(bm['train']) + [True] * len(bm['test']))
    comps = []
    for i in ids:
        f = byid[i]['formula']; c = None
        for cand in (f, clean(f)):
            try:
                c = Composition(cand); break
            except Exception:
                pass
        comps.append(c)
    print(f'rows {len(ids)}  train {(~is_test).sum()}  test {is_test.sum()}  '
          f'parsed {sum(c is not None for c in comps)}', flush=True)

    feat = MultipleFeaturizer([ElementProperty.from_preset('magpie'), Stoichiometry(),
                               ValenceOrbital(props=['avg', 'frac']), IonProperty(fast=True),
                               ElementFraction(), TMetalFraction(), BandCenter()])
    feat.set_n_jobs(1)
    names = feat.feature_labels()
    print(f'featurizing ({len(names)} matminer columns) ...', flush=True)
    M = np.array([feat.featurize(c) if c is not None else [np.nan] * len(names)
                  for c in comps], float)
    M = np.where(np.isfinite(M), M, np.nan)
    # column filter decided on TRAIN only
    tr = ~is_test
    keep = np.isfinite(M[tr]).mean(0) > 0.5
    M, names = M[:, keep], [n for n, k in zip(names, keep) if k]
    keep = np.array([len(np.unique(M[tr, j][np.isfinite(M[tr, j])])) > 1 for j in range(M.shape[1])])
    M, names = M[:, keep], [n for n, k in zip(names, keep) if k]

    P = np.array([physics_row(c) if c is not None else [np.nan] * len(PHYS) for c in comps], float)
    X1 = M                                   # "base"  feature set
    X2 = np.hstack([M, P])                   # "phys"  feature set
    names2 = names + PHYS
    ef = [i for i, n in enumerate(names2) if n in ELEMENTS]
    Xef = np.nan_to_num(X2[:, ef], nan=0.0)
    print(f'base {X1.shape[1]}  phys {X2.shape[1]}  element-fraction block {len(ef)}', flush=True)

    A, B, E = X1[tr], X2[tr], Xef[tr]
    At, Bt, Et = X1[~tr], X2[~tr], Xef[~tr]
    ytr = y[tr]; pos = ytr > 0

    def rf():
        return Pipeline([('i', SimpleImputer()), ('s', StandardScaler()),
                         ('m', RandomForestRegressor(n_estimators=300, max_features=1 / 3,
                                                     n_jobs=-1, bootstrap=False, random_state=0))])

    print('fitting gate ...', flush=True)
    gate = lgb.LGBMClassifier(n_estimators=1500, **lgbp())
    gate.fit(B, (ytr == 0).astype(int))
    p0 = gate.predict_proba(Bt)[:, 1]

    print('fitting 6 positive-part regressors ...', flush=True)
    members = []
    m = lgb.LGBMRegressor(n_estimators=2000, **lgbp()); m.fit(B[pos], ytr[pos])
    members.append(m.predict(Bt))                                        # phys L2 pos
    m = rf(); m.fit(B, ytr); members.append(m.predict(Bt))               # phys RF all
    m = lgb.LGBMRegressor(n_estimators=2000, **lgbp()); m.fit(A[pos], ytr[pos])
    members.append(m.predict(At))                                        # base L2 pos
    m = lgb.LGBMRegressor(objective='regression_l1', n_estimators=3000, **lgbp()); m.fit(B, ytr)
    members.append(m.predict(Bt))                                        # phys L1 all
    m = KNeighborsRegressor(1, weights='distance', n_jobs=-1); m.fit(E[pos], ytr[pos])
    members.append(m.predict(Et))                                        # phys kNN k=1
    m = rf(); m.fit(B[pos], ytr[pos]); members.append(m.predict(Bt))     # phys RF pos
    R = np.mean([np.clip(v, 0, None) for v in members], axis=0)

    pred = np.where(p0 >= THRESH, 0.0, R)
    yte = y[~tr]
    print(f'\nTEST MAE = {np.abs(yte - pred).mean():.4f}   (published leaderboard #1: 4.8511)\n')

    out = os.path.join(HERE, CSV)
    with open(out, 'w', newline='') as f:
        f.write('id,prediction\n')
        for i, p in zip([s for s, t in zip(ids, is_test) if t], pred):
            f.write(f'{i},{p:.6f}\n')
    with zipfile.ZipFile(out + '.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out, CSV)
    os.remove(out)
    print('wrote', out + '.zip')


if __name__ == '__main__':
    main()
