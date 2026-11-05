"""Nettoyage et recodage des réponses (étudiants ; enseignants V1+V2 fusionnés)."""
import sys, re
import numpy as np
import pandas as pd

DATA = sys.argv[1]   # dossier contenant etu.xlsx, ensV1.xlsx, ensV2.xlsx
OUT = sys.argv[2]

LIK = {"Pas du tout d'accord": 1, "Plutôt pas d'accord": 2, "Ni d'accord ni en désaccord": 3,
       "Plutôt d'accord": 4, "Tout à fait d'accord": 5}
FREQ = {'Jamais': 1, 'Rarement': 2, 'Parfois': 3, 'Souvent': 4, 'Très souvent': 5, 'Toujours': 5}


def col(d, prefix):
    m = [c for c in d.columns if c.startswith(prefix)]
    if len(m) != 1:
        raise KeyError((prefix, m))
    return m[0]


def bracket(c):
    m = re.search(r'\[(.*)\]\s*$', c)
    return m.group(1) if m else c


# ------------------------------------------------------------------ étudiants
e = pd.read_excel(f'{DATA}/etu.xlsx', 0)
e = e[e[col(e, '0.')] == 'Oui'].reset_index(drop=True)
s = pd.DataFrame(index=e.index)
s['etab'] = e[col(e, 'A1.')]
s['domaine'] = e[col(e, 'A2.')].replace({
    'Sciences exactes (mathématiques, physique, informatique, Sciences de la vie et de la terre, chimie )': 'Sciences',
    'Sciences commerciales et administratives': 'Commerce-gestion',
    'Lettres et langues': 'Lettres', "Psychopédagogie et sciences de l'éducation": 'Psychopédagogie',
    'Histoire, géographie et sciences sociales': 'Sciences sociales'})
s['annee'] = e[col(e, 'A4.')]
s['niveau'] = s['annee'].map({'Licence 1': 'L1-L2', 'Licence 2': 'L1-L2', 'Licence 3': 'L3+',
                              'Master 1': 'L3+', 'Master 2': 'L3+', 'Autre (ancien système, agrégation…)': 'L3+'})
s['femme'] = (e[col(e, 'A5.')] == 'Féminin').astype(float)
s['age'] = e[col(e, 'A6.')]
dev = e[col(e, 'A7.')].fillna('')
s['ordinateur'] = dev.str.contains('Ordinateur').astype(int)
s['acces'] = e[col(e, 'A8.')]
s['connex_jour'] = (e[col(e, 'A9.')] == 'Tous les jours').astype(int)
s['depense'] = e[col(e, 'A10.')]
s['usager'] = (e[col(e, 'A11.')] == 'Oui').astype(int)
s['anciennete'] = e[col(e, 'B1.')]
s['outils'] = e[col(e, 'B2.')]
s['outil_principal'] = e[col(e, 'B3.')]
s['version'] = e[col(e, 'B4.')]
s['freq_etudes'] = e[col(e, 'B5.')]
s['appareil'] = e[col(e, 'B6.')]
s['langues'] = e[col(e, 'B7.')]
b8 = [c for c in e.columns if c.startswith('B8.')]
for i, c in enumerate(b8, 1):
    s[f'B8_{i}'] = e[c].map(FREQ)
s['examen_salle'] = e[col(e, 'B9.')]
s['decouverte'] = e[col(e, 'B10.')]
s['C1_reutilisation'] = e[col(e, 'C1.')]
c2 = [c for c in e.columns if c.startswith('C2.')]
for i, c in enumerate(c2, 1):
    s[f'C2_{i}'] = e[c].map(FREQ)
s['C3_erreurs'] = e[col(e, 'C3.')]
s['C4_types'] = e[col(e, 'C4.')]
s['C5_declare'] = e[col(e, 'C5.')]
dcols = [c for c in e.columns if c.startswith('D1.') or c.startswith('D2.')]
codes = ['UT1', 'UT2', 'UT3', 'CF1', 'CF2', 'CF3', 'DP1', 'DP2', 'DP3', 'HM1',
         'HM2', 'HM3', 'CX1', 'CX2', 'CX3', 'IN1', 'IN2', 'IN3', 'FU1', 'FU2']
assert len(dcols) == 20
for k, c in zip(codes, dcols):
    s[k] = e[c].map(LIK)
s['E1_regles'] = e[col(e, 'E1.')]
s['E2_attitude'] = e[col(e, 'E2.')]
s['formation'] = e[col(e, 'E3.')].fillna('').str.contains('Oui').astype(int)
s['E4_souhait'] = e[col(e, 'E4.')]
s['E5_soupcon'] = e[col(e, 'E5.')]
# test de vigilance
key = {'F1': 'err', 'F2': 'ok', 'F3': 'err', 'F5': 'ok', 'F6': 'err'}
lab = {'La réponse est correcte': 'ok', 'La réponse contient une erreur': 'err', 'Je ne sais pas': 'nsp'}
for it, k in key.items():
    a = e[col(e, it + 'a.')].map(lab)
    s[it + '_rep'] = a
    s[it + '_juste'] = (a == k).astype(int)
    s[it + '_cert'] = pd.to_numeric(e[col(e, it + 'b.')], errors='coerce')
    s[it + '_texte'] = e[col(e, it + 'c.')]
s['VIG'] = s[[f'{i}_juste' for i in key]].sum(axis=1)
s['F4'] = e[col(e, 'F4a.')]
s['REF'] = (s['F4'].str.startswith("Je vérifie")).astype(int)
s['mode'] = e[col(e, 'G1.')]
s.to_csv(f'{OUT}/etudiants_codes.csv', index=False)

# ------------------------------------------------------------------ enseignants
parts = []
for v in ('V1', 'V2'):
    d = pd.read_excel(f'{DATA}/ens{v}.xlsx', 0)
    d['version'] = v
    parts.append(d)
# les colonnes T sont dans un ordre différent : on aligne par nom
t = pd.concat(parts, ignore_index=True, sort=False)
t = t[t[col(t, '0.')] == 'Oui'].reset_index(drop=True)
o = pd.DataFrame(index=t.index)
o['version'] = t['version']
o['etab'] = t[col(t, 'A1.')]
o['grade'] = t[col(t, 'A2.')]
o['domaine'] = t[col(t, 'A3.')]
o['anciennete'] = t[col(t, 'A4.')]
o['effectif'] = t[col(t, 'A5.')]
o['copies'] = t[col(t, 'A6.')]
o['femme'] = (t[col(t, 'A7.')] == 'Féminin').astype(float)
o['age'] = t[col(t, 'A8.')]
o['usager'] = (t[col(t, 'A9.')] == 'Oui').astype(int)
o['outils'] = t[col(t, 'B1.')]
o['version_ia'] = t[col(t, 'B2.')]
b3 = [c for c in t.columns if c.startswith('B3.')]
for i, c in enumerate(b3, 1):
    o[f'B3_{i}'] = t[c].map(FREQ)
o['B4'] = t[col(t, 'B4.')]
o['B5'] = t[col(t, 'B5.')]
o['C1_part'] = t[col(t, 'C1.')]
o['C2_indices'] = t[col(t, 'C2.')]
o['C3_reaction'] = t[col(t, 'C3.')]
o['C4_adapt'] = t[col(t, 'C4.')]
o['C5_fiab'] = pd.to_numeric(t[col(t, 'C5.')], errors='coerce')
dc = [c for c in t.columns if c.startswith('D. ')]
dcodes = ['DT1', 'DT2', 'DT3', 'AC1', 'AC2', 'AC3', 'MI1', 'MI2', 'MI3', 'EV1', 'EV2', 'EV3', 'FO1', 'FO2']
assert len(dc) == 14
for k, c in zip(dcodes, dc):
    o[k] = t[c].map(LIK)
# test de détection — clé réelle confirmée : IA = T1, T4, T5 ; étudiants = T2, T3, T6
truth = {'T1': 'IA', 'T2': 'H', 'T3': 'H', 'T4': 'IA', 'T5': 'IA', 'T6': 'H'}
jl = {"Produit en tout ou en partie avec l'IA": 'IA', 'Écrit par un étudiant sans IA': 'H', 'Je ne sais pas': 'nsp'}
for k, v in truth.items():
    o[k + '_jug'] = t[col(t, k + 'a.')].map(jl)
    o[k + '_juste'] = (o[k + '_jug'] == v).astype(int)
    o[k + '_cert'] = pd.to_numeric(t[col(t, k + 'b.')], errors='coerce')
    o[k + '_note'] = pd.to_numeric(t[col(t, k + 'c.')], errors='coerce')
    o[k + '_motif'] = t[col(t, k + 'd.')]
o['DET'] = o[[k + '_juste' for k in truth]].sum(axis=1)
# position d'affichage (V1 : T1..T6 ; V2 : T6..T1)
for k in truth:
    n = int(k[1])
    o[k + '_pos'] = np.where(o['version'] == 'V1', n, 7 - n)
# classements
fmts = ['QCM', 'Ecrit_salle', 'Domicile', 'Oral', 'Projet', 'Portfolio']
for crit in ('F1', 'F2', 'F3', 'F4'):
    cs = [c for c in t.columns if c.startswith(crit + '.')]
    assert len(cs) == 6, cs
    for f_, c in zip(fmts, cs):
        o[f'{crit}_{f_}'] = t[c].astype(str).str.extract(r'(\d)')[0].astype(float)
f5 = [c for c in t.columns if c.startswith('F5.')]
for k, c in zip(['W_fraude', 'W_effectif', 'W_comprehension', 'W_charge'], f5):
    o[k] = t[c].astype(str).str.extract(r'(\d)')[0].astype(float)
o['F6_choix'] = t[col(t, 'F6.')]
o['G1'] = t[col(t, 'G1.')]
o['G2'] = t[col(t, 'G2.')]
o['G3'] = t[col(t, 'G3.')]
o['G4'] = t[col(t, 'G4.')]
o.to_csv(f'{OUT}/enseignants_codes.csv', index=False)
print('etudiants', s.shape, 'enseignants', o.shape)
print(o['version'].value_counts().to_dict())
