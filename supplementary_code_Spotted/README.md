# Supplementary material: detection, grading and assessment formats (Congolese teacher educators)

This folder reproduces every number, table and figure of the submitted paper.

## Contents

- `clean.py`: recodes and merges the two versions of the teacher questionnaire (raw exports not distributed).
- `mmcm.py`: Mean-Median Compromise Method (Ngoie et al., Mathematics 2024, 12, 3631; correction 2026, 14, 3260). Running `python mmcm.py` checks the worked examples of the original article.
- `analyse_enseignants.py`: descriptive statistics, signal detection indices (log-linear correction), paired Wilcoxon test and linear mixed model of marks, ROC-weighted aggregation of format rankings with MMCM, range voting, median grade and Copeland's rule, Kendall's W, bootstrap stability, Figures 1 and 2. Writes `results/resultats_enseignants.json`.
- `inferences_enseignants.py`: cluster-bootstrap intervals for hit and false-alarm rates, d' and the grading premium; order-effect tests (Mann-Whitney, GEE); intraclass correlation; suspicion effect with text fixed effects (quality held constant) and on AI texts only; treatment of "don't know" answers; sensitivity analysis; Cronbach's alpha with Feldt intervals; bootstrap intervals of Kendall's W; sign test and MMCM difference between the two leading formats. Writes `results/inferences_enseignants.json`.
- `results/enseignants_codes.csv`: anonymized coded responses (n = 47, version 1: 25, version 2: 22; free-text comments removed).
- `results/notes_format_long.csv`: marks in long format (282 rows) used by the mixed model.

## Running

```
pip install -r requirements.txt
python mmcm.py
python analyse_enseignants.py results/enseignants_codes.csv results
python inferences_enseignants.py results/enseignants_codes.csv results
```

Seed 20261025 (bootstrap). Answer key of the detection test: AI texts T1, T4, T5; student copies T2, T3, T6. Version 1 of the form displayed T1 to T6, version 2 displayed T6 to T1.
