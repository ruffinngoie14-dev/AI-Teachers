"""Analyses du papier « enseignants » (A2MRA 2027).

Usage : python analyse_enseignants.py enseignants_codes.csv sortie/
Clé du test (confirmée) : textes d'IA = T1, T4, T5 ; copies d'étudiants = T2, T3, T6.
V1 affichait T1..T6, V2 affichait T6..T1. Graine : 20261025.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon, spearmanr, chi2
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmcm import mmcm_profile, mmcm_score, nu

SEED = 20261025
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
o = pd.read_csv(src)
R = {}
n = len(o)
R['n'] = n
R['version'] = o['version'].value_counts().to_dict()
R['etab'] = o['etab'].value_counts().to_dict()
R['grade'] = o['grade'].value_counts().to_dict()
R['femmes'] = int(o['femme'].sum())
R['usagers'] = int(o['usager'].sum())
R['anciennete'] = o['anciennete'].value_counts().to_dict()
R['effectif'] = o['effectif'].value_counts().to_dict()
R['version_ia'] = o['version_ia'].value_counts().to_dict()

B3 = ['Préparer un cours', 'Rédiger des questions', 'QCM', 'Barème', 'Corriger/noter', 'Commentaires',
      'Évaluer TFC/mémoire', 'Vérifier IA', 'Article/projet', 'Traduire']
R['B3_au_moins_parfois'] = {lab: round(100 * float((o[f'B3_{i}'] >= 3).mean()), 1) for i, lab in enumerate(B3, 1)}
R['B4'] = o['B4'].value_counts().to_dict()
R['B5'] = o['B5'].value_counts().to_dict()
R['C1'] = o['C1_part'].value_counts().to_dict()
R['C3'] = o['C3_reaction'].value_counts().to_dict()


def multi(col):
    c = {}
    for v in o[col].dropna():
        for p in str(v).split(', '):
            c[p] = c.get(p, 0) + 1
    return dict(sorted(c.items(), key=lambda x: -x[1]))


R['C2'] = multi('C2_indices')
R['C4'] = multi('C4_adapt')
R['G4'] = multi('G4')
R['G1'] = o['G1'].value_counts().to_dict()
R['G2'] = o['G2'].value_counts().to_dict()
R['G3_aucune'] = int((o['G3'] == 'Non').sum())
R['C5_moy'] = round(float(o['C5_fiab'].mean()), 2)


def alpha(X):
    X = X.dropna()
    k = X.shape[1]
    return float(k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1)))


R['likert'] = {c: round(float(o[c].mean()), 2) for c in ['DT1', 'DT2', 'DT3', 'AC1', 'AC2', 'AC3', 'MI1', 'MI2', 'MI3',
                                                         'EV1', 'EV2', 'EV3', 'FO1', 'FO2']}
R['likert_accord'] = {c: round(100 * float((o[c] >= 4).mean()), 1) for c in R['likert']}
R['alpha_DT12'] = round(alpha(o[['DT1', 'DT2']]), 2)

# ------------------------------------------------------------------ test de détection
AI, HU = ['T1', 'T4', 'T5'], ['T2', 'T3', 'T6']
T = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
R['textes'] = {}
for t in T:
    vc = o[t + '_jug'].value_counts()
    R['textes'][t] = dict(source='IA' if t in AI else 'etudiant', juge_IA=int(vc.get('IA', 0)),
                          juge_H=int(vc.get('H', 0)), nsp=int(vc.get('nsp', 0)),
                          note=round(float(o[t + '_note'].mean()), 2), note_et=round(float(o[t + '_note'].std()), 2),
                          cert=round(float(o[t + '_cert'].mean()), 2))
o['h'] = sum((o[t + '_jug'] == 'IA').astype(int) for t in AI)
o['fa'] = sum((o[t + '_jug'] == 'IA').astype(int) for t in HU)
o['allnsp'] = sum((o[t + '_jug'] == 'nsp').astype(int) for t in T) == 6
R['tout_nsp'] = int(o['allnsp'].sum())
hr = (o['h'] + 0.5) / 4
fr = (o['fa'] + 0.5) / 4                  # correction log-linéaire de Hautus
o['dprime'] = norm.ppf(hr) - norm.ppf(fr)
o['crit'] = -0.5 * (norm.ppf(hr) + norm.ppf(fr))
R['taux_reconnaissance_IA'] = round(100 * o['h'].sum() / (3 * n), 1)
R['taux_fausse_alarme'] = round(100 * o['fa'].sum() / (3 * n), 1)
R['taux_reconnaissance_H'] = round(100 * sum((o[t + '_jug'] == 'H').sum() for t in HU) / (3 * n), 1)
R['DET_moy'] = round(float(o['DET'].mean()), 2)
R['DET_dist'] = o['DET'].value_counts().sort_index().to_dict()
R['dprime'] = dict(moy=round(float(o['dprime'].mean()), 2), med=round(float(o['dprime'].median()), 2),
                   positifs=int((o['dprime'] > 0).sum()), negatifs=int((o['dprime'] < 0).sum()),
                   p_wilcoxon=float(wilcoxon(o['dprime']).pvalue))
R['critere_c_moy'] = round(float(o['crit'].mean()), 2)
act = ~o['allnsp']
R['dprime_actifs'] = round(float(o.loc[act, 'dprime'].mean()), 2)
o['DTs'] = o[['DT1', 'DT2']].mean(axis=1)
rho = spearmanr(o['DTs'], o['dprime'])
R['rho_DT_dprime'] = [round(float(rho[0]), 3), float(rho[1])]
R['DT_accord'] = round(100 * float((o['DTs'] >= 4).mean()), 1)

# ------------------------------------------------------------------ notes
rows = []
for i, r in o.iterrows():
    for t in T:
        rows.append(dict(id=i, t=t, ia=int(t in AI), jug=r[t + '_jug'], susp=int(r[t + '_jug'] == 'IA'),
                         note=r[t + '_note'], pos=r[t + '_pos'], cert=r[t + '_cert']))
L = pd.DataFrame(rows).dropna(subset=['note'])
L.to_csv(os.path.join(out, 'notes_format_long.csv'), index=False)
g = L.groupby(['ia', 'jug'])['note'].agg(['mean', 'std', 'count'])
R['notes_source_jugement'] = {f"{'IA' if a else 'H'}|{b}": dict(m=round(float(v['mean']), 2), n=int(v['count']))
                              for (a, b), v in g.iterrows()}
o['noteIA'] = o[[t + '_note' for t in AI]].mean(axis=1)
o['noteH'] = o[[t + '_note' for t in HU]].mean(axis=1)
d = (o['noteIA'] - o['noteH']).dropna()
R['prime_IA'] = dict(moy=round(float(d.mean()), 2), med=round(float(d.median()), 2), plus=int((d > 0).sum()),
                     moins=int((d < 0).sum()), egal=int((d == 0).sum()), p=float(wilcoxon(d).pvalue),
                     noteIA=round(float(o['noteIA'].mean()), 2), noteH=round(float(o['noteH'].mean()), 2))
m1 = smf.mixedlm('note ~ ia + susp + pos', L, groups=L['id']).fit(reml=True)
m2 = smf.mixedlm('note ~ ia * susp + pos', L, groups=L['id']).fit(reml=True)
ci = m1.conf_int()
R['mixte'] = {k: dict(b=round(float(m1.params[k]), 2), lo=round(float(ci.loc[k, 0]), 2),
                      hi=round(float(ci.loc[k, 1]), 2), p=float(m1.pvalues[k])) for k in ['ia', 'susp', 'pos']}
R['mixte_interaction'] = dict(b=round(float(m2.params['ia:susp']), 2), p=round(float(m2.pvalues['ia:susp']), 3))
R['mixte_n'] = dict(obs=int(len(L)), groupes=int(L['id'].nunique()))
# exactitude selon la position et la version
R['exact_position'] = {int(k): round(100 * float(v), 1) for k, v in
                       L.assign(j=(L['jug'] == np.where(L['ia'] == 1, 'IA', 'H')).astype(int)).groupby('pos')['j'].mean().items()}

# ------------------------------------------------------------------ agrégation des classements
F = ['QCM', 'Ecrit_salle', 'Domicile', 'Oral', 'Projet', 'Portfolio']
FL = {'QCM': 'MCQ (invigilated)', 'Ecrit_salle': 'Open questions (invigilated)', 'Domicile': 'Take-home essay',
      'Oral': 'Oral exam', 'Projet': 'Project + defence', 'Portfolio': 'Staged portfolio'}
CR = ['F1', 'F2', 'F3', 'F4']
CRL = {'F1': 'AI-fraud resistance', 'F2': 'Large cohorts', 'F3': 'Real understanding', 'F4': 'Teacher workload'}
G = {c: 7 - o[[f'{c}_{f}' for f in F]].values for c in CR}           # rang 1 -> note 6
# poids ROC individuels à partir du classement des critères (F5)
roc = {r: sum(1 / i for i in range(r, 5)) / 4 for r in range(1, 5)}
Wr = o[['W_fraude', 'W_effectif', 'W_comprehension', 'W_charge']].values
W = np.vectorize(roc.get)(Wr.astype(int))
Gg = sum(W[:, [j]] * G[c] for j, c in enumerate(CR))               # note globale individuelle pondérée
R['poids_moyens'] = {CRL[c]: round(float(W[:, j].mean()), 3) for j, c in enumerate(CR)}
R['critere_rang1'] = {CRL[c]: int((Wr[:, j] == 1).sum()) for j, c in enumerate(CR)}


def kendall_w(Rk):
    m, k = Rk.shape
    s = Rk.sum(axis=0)
    w = 12 * ((s - s.mean()) ** 2).sum() / (m ** 2 * (k ** 3 - k))
    return float(w), float(chi2.sf(m * (k - 1) * w, k - 1))


def rank_methods(Gm):
    """Retourne le classement des formats selon chaque méthode (1 = meilleur)."""
    res = {}
    prof = {f: mmcm_profile(Gm[:, j]) for j, f in enumerate(F)}
    order = sorted(F, key=lambda f: prof[f], reverse=True)
    res['MMCM'] = {f: order.index(f) + 1 for f in F}
    res['MMCM_k2'] = {f: round(prof[f][0], 3) for f in F}
    mean = Gm.mean(axis=0)
    res['Range'] = {f: int(1 + (mean > mean[j]).sum()) for j, f in enumerate(F)}
    med = np.median(Gm, axis=0)
    res['Median'] = {f: int(1 + (med > med[j]).sum()) for j, f in enumerate(F)}
    P = np.array([[(Gm[:, i] > Gm[:, j]).sum() for j in range(6)] for i in range(6)])
    cop = np.array([(P[i] > P[:, i]).sum() - (P[i] < P[:, i]).sum() for i in range(6)])
    res['Copeland'] = {f: int(1 + (cop > cop[j]).sum()) for j, f in enumerate(F)}
    cw = [F[i] for i in range(6) if all(P[i, j] > P[j, i] for j in range(6) if j != i)]
    res['Condorcet_winner'] = cw[0] if cw else None
    return res


R['agregation'] = {}
for c in CR:
    rk = rank_methods(G[c])
    rk['W'] = kendall_w(7 - G[c])
    rk['moyenne_rang'] = {f: round(float((7 - G[c])[:, j].mean()), 2) for j, f in enumerate(F)}
    R['agregation'][c] = rk
rk = rank_methods(Gg)
rk['moyenne_note'] = {f: round(float(Gg[:, j].mean()), 3) for j, f in enumerate(F)}
R['agregation']['global'] = rk
R['F6'] = o['F6_choix'].value_counts().to_dict()
R['nu_41'] = nu(n)

# stabilité : rééchantillonnage bootstrap des enseignants
rng = np.random.default_rng(SEED)
B = 2000
first = {m: {f: 0 for f in F} for m in ['MMCM', 'Range', 'Median', 'Copeland']}
for _ in range(B):
    idx = rng.integers(0, n, n)
    rk = rank_methods(Gg[idx])
    for m in first:
        winners = [f for f in F if rk[m][f] == 1]
        for f in winners:
            first[m][f] += 1 / len(winners)
R['bootstrap_premier'] = {m: {f: round(100 * v / B, 1) for f, v in d.items()} for m, d in first.items()}

with open(os.path.join(out, 'resultats_enseignants.json'), 'w', encoding='utf8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)

# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'],
                     'font.size': 8, 'axes.edgecolor': '#8a8984', 'axes.linewidth': 0.6,
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#2b2a27'})
C_AI, C_H, C_N = '#ef7d4a', '#4a8ad8', '#cfcdc6'
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.15), gridspec_kw={'width_ratios': [1.2, 1]})
ax = axs[0]
order = ['T1', 'T4', 'T5', 'T2', 'T3', 'T6']
for i, t in enumerate(order):
    d_ = R['textes'][t]
    vals = [d_['juge_IA'], d_['nsp'], d_['juge_H']]
    left = 0
    y = len(order) - 1 - i
    for v, col in zip(vals, (C_AI, C_N, C_H)):
        w = 100 * v / n
        ax.barh(y, w - 0.4, left=left + 0.2, height=0.62, color=col)
        if w >= 9:
            ax.text(left + w / 2, y, f'{w:.0f}%', ha='center', va='center', fontsize=6.8,
                    color='#ffffff' if col != C_N else '#2b2a27')
        left += w
ax.set_yticks(range(6))
ax.set_yticklabels([f"{t} ({'AI' if t in AI else 'student'})" for t in order[::-1]])
ax.axhline(2.5, color='#8a8984', lw=0.6)
ax.set_xlim(0, 100)
ax.set_xlabel(f'Judgements (%), n = {n} teachers')
ax.set_title('(a) Source attributed to each text', fontsize=8)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_AI, label='judged AI'), Patch(color=C_N, label="don't know"),
                   Patch(color=C_H, label='judged student')], ncol=3, frameon=False, fontsize=6.5,
          loc='upper center', bbox_to_anchor=(0.45, -0.28))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
ax = axs[1]
pos = {('H', 'H'): 0, ('H', 'IA'): 1, ('IA', 'H'): 2.6, ('IA', 'IA'): 3.6}
rng = np.random.default_rng(SEED)
for (srcl, jl), x in pos.items():
    v = L[(L['ia'] == (srcl == 'IA')) & (L['jug'] == jl)]['note'].values
    bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    col = C_AI if srcl == 'IA' else C_H
    ax.plot([x, x], [lo, hi], color=col, lw=2, solid_capstyle='round')
    ax.plot(x, v.mean(), 'o', ms=6, color=col, mec='#fcfcfb', mew=1.5)
    ax.text(x + 0.12, v.mean(), f'{v.mean():.1f}\n(n={len(v)})', fontsize=6.3, va='center', color='#52514e')
ax.set_xticks(list(pos.values()))
ax.set_xticklabels(['judged\nstudent', 'judged\nAI', 'judged\nstudent', 'judged\nAI'], fontsize=6.8)
ax.text(0.5, 9.3, 'student copies', ha='center', fontsize=7, color=C_H)
ax.text(3.1, 9.3, 'AI texts', ha='center', fontsize=7, color=C_AI)
ax.set_ylim(9, 17.5)
ax.set_xlim(-0.5, 4.4)
ax.set_ylabel('Mark awarded (/20)')
ax.set_title('(b) Marks by true source and judgement', fontsize=8)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout(w_pad=1.8)
fig.savefig(os.path.join(out, 'fig_detection.pdf'))

# figure 2 : probabilité bootstrap d'être premier, par méthode (formats ayant au moins une victoire)
meths = ['MMCM', 'Range', 'Median', 'Copeland']
mlab = {'MMCM': 'MMCM', 'Range': 'Range voting (mean)', 'Median': 'Median grade (ties shared)',
        'Copeland': 'Copeland (pairwise)'}
cols = ['#2a78d6', '#9cc3ec', '#1baf7a', '#e8a33a']
fs = [f for f in sorted(F, key=lambda f: -R['bootstrap_premier']['MMCM'][f])
      if max(R['bootstrap_premier'][m][f] for m in meths) > 0]
fig, ax = plt.subplots(figsize=(4.6, 1.9))
h = 0.19
for j, m in enumerate(meths):
    ys = np.arange(len(fs)) + (j - 1.5) * h
    vals = [R['bootstrap_premier'][m][f] for f in fs]
    ax.barh(ys, vals, height=h - 0.03, color=cols[j], label=mlab[m])
    for y_, v in zip(ys, vals):
        ax.text(v + 1, y_, f'{v:.0f}', va='center', fontsize=6, color='#52514e')
ax.set_yticks(np.arange(len(fs)))
ax.set_yticklabels([FL[f] for f in fs], fontsize=7)
ax.invert_yaxis()
ax.set_xlim(0, 120)
ax.set_xlabel('Ranked first (% of 2,000 bootstrap resamples of teachers)', fontsize=7)
ax.legend(ncol=2, frameon=False, fontsize=6.3, loc='lower right', bbox_to_anchor=(1.0, 0.02))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out, 'fig_bootstrap.pdf'))
print(json.dumps({k: R[k] for k in ['n', 'taux_reconnaissance_IA', 'taux_fausse_alarme', 'dprime', 'prime_IA', 'mixte',
                                     'bootstrap_premier']}, indent=1, default=str))
for c in CR + ['global']:
    a = R['agregation'][c]
    print(c, 'MMCM', a['MMCM'], '\n   Range', a['Range'], '\n   Median', a['Median'], '\n   Copeland', a['Copeland'], a['Condorcet_winner'], a.get('W'))
