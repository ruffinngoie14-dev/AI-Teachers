"""Statistiques inférentielles complémentaires du papier « enseignants ».

Usage : python inferences_enseignants.py results/enseignants_codes.csv results/
Lit aussi results/notes_format_long.csv. Écrit results/inferences_enseignants.json. Graine 20261025.
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import f as fdist, norm, spearmanr, mannwhitneyu, wilcoxon
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mmcm import mmcm_profile

SEED = 20261025
src, out = sys.argv[1], sys.argv[2]
o = pd.read_csv(src)
L = pd.read_csv(os.path.join(out, 'notes_format_long.csv'))
n = len(o)
AI, HU = ['T1', 'T4', 'T5'], ['T2', 'T3', 'T6']
R = {}
rng = np.random.default_rng(SEED)
B = 5000


def cronbach_ci(X, level=0.95):
    X = X.dropna()
    m, k = X.shape
    a = k / (k - 1) * (1 - X.var(ddof=1).sum() / X.sum(axis=1).var(ddof=1))
    d1, d2 = m - 1, (m - 1) * (k - 1)
    g = (1 - level) / 2
    return dict(alpha=round(float(a), 2), lo=round(float(1 - (1 - a) * fdist.ppf(1 - g, d1, d2)), 2),
                hi=round(float(1 - (1 - a) * fdist.ppf(g, d1, d2)), 2), n=int(m), k=int(k))


def boot_ci(stat, idx_n, B=B):
    vals = [stat(rng.integers(0, idx_n, idx_n)) for _ in range(B)]
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


# taux de reconnaissance et de fausses alarmes : IC par bootstrap en grappes (enseignants)
h = sum((o[t + '_jug'] == 'IA').astype(int) for t in AI).values
fa = sum((o[t + '_jug'] == 'IA').astype(int) for t in HU).values
cr = sum((o[t + '_jug'] == 'H').astype(int) for t in HU).values
R['hit'] = dict(p=round(100 * h.sum() / (3 * n), 1), ci=[100 * x for x in boot_ci(lambda i: h[i].sum() / (3 * len(i)), n)])
R['fa'] = dict(p=round(100 * fa.sum() / (3 * n), 1), ci=[100 * x for x in boot_ci(lambda i: fa[i].sum() / (3 * len(i)), n)])
R['cr'] = dict(p=round(100 * cr.sum() / (3 * n), 1), ci=[100 * x for x in boot_ci(lambda i: cr[i].sum() / (3 * len(i)), n)])
for k in ('hit', 'fa', 'cr'):
    R[k]['ci'] = [round(x, 1) for x in R[k]['ci']]
dp = norm.ppf((h + 0.5) / 4) - norm.ppf((fa + 0.5) / 4)
R['dprime'] = dict(moy=round(float(dp.mean()), 2), ci=boot_ci(lambda i: dp[i].mean(), n))
crit = -0.5 * (norm.ppf((h + 0.5) / 4) + norm.ppf((fa + 0.5) / 4))
R['critere'] = dict(moy=round(float(crit.mean()), 2), ci=boot_ci(lambda i: crit[i].mean(), n),
                    p_wilcoxon=float(wilcoxon(crit[crit != 0]).pvalue))
# effet d'ordre : sensibilité selon la version, exactitude selon la position (GEE logistique)
R['dprime_version'] = dict(V1=round(float(dp[o['version'] == 'V1'].mean()), 2),
                           V2=round(float(dp[o['version'] == 'V2'].mean()), 2),
                           p=float(mannwhitneyu(dp[o['version'] == 'V1'], dp[o['version'] == 'V2']).pvalue))
L['juste'] = (L['jug'] == np.where(L['ia'] == 1, 'IA', 'H')).astype(int)
gee = smf.gee('juste ~ pos', groups='id', data=L, family=sm.families.Binomial(),
              cov_struct=sm.cov_struct.Exchangeable()).fit()
R['gee_position'] = dict(OR=round(float(np.exp(gee.params['pos'])), 2),
                         lo=round(float(np.exp(gee.conf_int().loc['pos', 0])), 2),
                         hi=round(float(np.exp(gee.conf_int().loc['pos', 1])), 2), p=float(gee.pvalues['pos']))
# auto-évaluation et sensibilité
DTs = o[['DT1', 'DT2']].mean(axis=1).values
r = spearmanr(DTs, dp)
R['rho_DT_dprime'] = dict(rho=round(float(r[0]), 2), p=float(r[1]),
                          ci=boot_ci(lambda i: spearmanr(DTs[i], dp[i])[0], n))
# prime de notation : IC bootstrap, taille d'effet
d = (o[[t + '_note' for t in AI]].mean(axis=1) - o[[t + '_note' for t in HU]].mean(axis=1)).values
R['prime'] = dict(moy=round(float(d.mean()), 2), ci=boot_ci(lambda i: d[i].mean(), n),
                  dz=round(float(d.mean() / d.std(ddof=1)), 2), p=float(wilcoxon(d).pvalue))
# modèle mixte : coefficient de corrélation intraclasse
m0 = smf.mixedlm('note ~ 1', L, groups=L['id']).fit(reml=True)
v_u = float(m0.cov_re.iloc[0, 0])
R['icc_notes'] = round(v_u / (v_u + float(m0.scale)), 2)
m1 = smf.mixedlm('note ~ ia + susp + pos', L, groups=L['id']).fit(reml=True)
R['mixte_sigma'] = dict(enseignant=round(float(np.sqrt(m1.cov_re.iloc[0, 0])), 2), residu=round(float(np.sqrt(m1.scale)), 2))
# sensibilité : modèle mixte sans les 3 enseignants « je ne sais pas » partout
alln = o.index[(o[[t + '_jug' for t in AI + HU]] == 'nsp').all(axis=1)]
m2 = smf.mixedlm('note ~ ia + susp + pos', L[~L['id'].isin(alln)], groups=L.loc[~L['id'].isin(alln), 'id']).fit(reml=True)
R['mixte_sans_nsp'] = dict(ia=round(float(m2.params['ia']), 2), susp=round(float(m2.params['susp']), 2),
                           p_susp=round(float(m2.pvalues['susp']), 3))
# confusion source/qualité : effet du soupçon à texte constant (effets fixes de texte)
mt = smf.mixedlm('note ~ C(t) + susp + pos', L, groups=L['id']).fit(reml=True)
ci_t = mt.conf_int()
R['susp_effets_fixes_texte'] = dict(b=round(float(mt.params['susp']), 2), lo=round(float(ci_t.loc['susp', 0]), 2),
                                    hi=round(float(ci_t.loc['susp', 1]), 2), p=float(mt.pvalues['susp']))
LA = L[L['ia'] == 1]
ma = smf.mixedlm('note ~ C(t) + susp + pos', LA, groups=LA['id']).fit(reml=True)
ci_a = ma.conf_int()
R['susp_textes_IA_seuls'] = dict(b=round(float(ma.params['susp']), 2), lo=round(float(ci_a.loc['susp', 0]), 2),
                                 hi=round(float(ci_a.loc['susp', 1]), 2), p=float(ma.pvalues['susp']), n=int(len(LA)))
# longueur des textes (mots) : IA 165, 161, 156 ; copies 111, 124, 134 -> séparation complète
R['longueur_mots'] = dict(IA=[156, 161, 165], etudiants=[111, 124, 134])
# traitement des « je ne sais pas » : effectif et d' recalculé sur les seuls textes jugés
nsp_tot = int((o[[t + '_jug' for t in AI + HU]] == 'nsp').values.sum())
R['nsp_total'] = dict(n=nsp_tot, part=round(100 * nsp_tot / (6 * n), 1))
hj = sum((o[t + '_jug'] != 'nsp').astype(int) for t in AI).values
fj = sum((o[t + '_jug'] != 'nsp').astype(int) for t in HU).values
ok = (hj > 0) & (fj > 0)
dp2 = norm.ppf((h[ok] + 0.5) / (hj[ok] + 1)) - norm.ppf((fa[ok] + 0.5) / (fj[ok] + 1))
R['dprime_sans_nsp'] = dict(moy=round(float(dp2.mean()), 2), n=int(ok.sum()))

# fiabilité des échelles
o2 = o.copy()
o2['AC3r'] = 6 - o2['AC3']
o2['EV3r'] = 6 - o2['EV3']
R['alpha'] = {'DT (DT1, DT2)': cronbach_ci(o2[['DT1', 'DT2']]), 'AC (AC1, AC2, AC3r)': cronbach_ci(o2[['AC1', 'AC2', 'AC3r']]),
              'MI (MI1-MI3)': cronbach_ci(o2[['MI1', 'MI2', 'MI3']]), 'EV (EV1, EV2, EV3r)': cronbach_ci(o2[['EV1', 'EV2', 'EV3r']]),
              'FO (FO1, FO2)': cronbach_ci(o2[['FO1', 'FO2']])}

# Kendall W : IC bootstrap ; différence MMCM oral - écrit en salle (note globale), IC bootstrap
F = ['QCM', 'Ecrit_salle', 'Domicile', 'Oral', 'Projet', 'Portfolio']
CR = ['F1', 'F2', 'F3', 'F4']
G = {c: 7 - o[[f'{c}_{f}' for f in F]].values for c in CR}


def kendall_w(Rk):
    m, k = Rk.shape
    s_ = Rk.sum(axis=0)
    return 12 * ((s_ - s_.mean()) ** 2).sum() / (m ** 2 * (k ** 3 - k))


R['kendallW_ci'] = {c: boot_ci(lambda i, c=c: kendall_w(7 - G[c][i]), n, B=2000) for c in CR}
roc = {r_: sum(1 / j for j in range(r_, 5)) / 4 for r_ in range(1, 5)}
W = np.vectorize(roc.get)(o[['W_fraude', 'W_effectif', 'W_comprehension', 'W_charge']].values.astype(int))
Gg = sum(W[:, [j]] * G[c] for j, c in enumerate(CR))
jo, je = F.index('Oral'), F.index('Ecrit_salle')
diff = lambda i: mmcm_profile(Gg[i, jo])[0] - mmcm_profile(Gg[i, je])[0]
R['mmcm_oral_moins_ecrit'] = dict(diff=round(float(diff(np.arange(n))), 3), ci=boot_ci(diff, n, B=2000))
# préférence par paire oral contre écrit en salle (note globale individuelle) : test des signes
plus = int((Gg[:, jo] > Gg[:, je]).sum())
moins = int((Gg[:, jo] < Gg[:, je]).sum())
from scipy.stats import binomtest
R['oral_vs_ecrit_signes'] = dict(oral=plus, ecrit=moins, p=float(binomtest(plus, plus + moins, 0.5).pvalue))

# B5 : information des étudiants lorsque l'IA intervient dans une évaluation (parmi les concernés)
from statsmodels.stats.proportion import proportion_confint
conc = o['B5'].isin(['Jamais', 'Parfois', 'Toujours'])
nj = int((o.loc[conc, 'B5'] == 'Jamais').sum())
lo_, hi_ = proportion_confint(nj, int(conc.sum()), method='wilson')
R['B5_jamais_parmi_concernes'] = dict(jamais=nj, concernes=int(conc.sum()), part=round(100 * nj / conc.sum(), 1),
                                      ci=[round(100 * lo_, 1), round(100 * hi_, 1)],
                                      toujours=int((o['B5'] == 'Toujours').sum()))
# B5 parmi les usagers de l'IA pour au moins une tâche d'évaluation (B3_2 à B3_7 >= « parfois »)
ev = o[['B3_2', 'B3_3', 'B3_4', 'B3_5', 'B3_6', 'B3_7']].ge(3).any(axis=1)
nj2 = int((o.loc[ev, 'B5'] == 'Jamais').sum())
lo2, hi2 = proportion_confint(nj2, int(ev.sum()), method='wilson')
R['B5_usagers_evaluation'] = dict(n=int(ev.sum()), repartition=o.loc[ev, 'B5'].value_counts().to_dict(),
                                  ci_jamais=[round(100 * lo2, 1), round(100 * hi2, 1)])
# B5 chez ceux qui reprennent la note du chatbot, et chez ceux qui exigent la déclaration des étudiants (C4)
R['B5_reprend_note'] = o.loc[o['B4'].fillna('').str.startswith('Je soumets la copie et je reprends'), 'B5'].value_counts().to_dict()
dec = o['C4_adapt'].fillna('').str.contains('obligation de la déclarer')
R['B5_exige_declaration'] = dict(n=int(dec.sum()), repartition=o.loc[dec, 'B5'].value_counts().to_dict())
# indices cités dans les commentaires (comptage par mots-clés) ; les commentaires libres ne sont pas
# diffusés (protection des répondants) : ce bloc ne tourne que sur le fichier complet.
import re
if 'T1_motif' in o.columns:
  com = [(j, m.lower()) for t in AI + HU for j, m in zip(o[t + '_jug'], o[t + '_motif']) if isinstance(m, str) and m.strip()]
  pat = {'fautes': r'faute|orthograph|grammai', 'langue_style': r'langu|style|vocab|tournure|registre|soutenu|formel'}
  R['commentaires'] = {k: dict(IA=sum(1 for j, m in com if j == 'IA' and re.search(p, m)), n_IA=sum(j == 'IA' for j, _ in com),
                               H=sum(1 for j, m in com if j == 'H' and re.search(p, m)), n_H=sum(j == 'H' for j, _ in com))
                       for k, p in pat.items()}

with open(os.path.join(out, 'inferences_enseignants.json'), 'w', encoding='utf8') as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
