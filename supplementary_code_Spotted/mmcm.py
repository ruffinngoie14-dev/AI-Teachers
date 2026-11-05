"""Mean-Median Compromise Method (Ngoie et al., Mathematics 2024, 12, 3631; correction 2026, 14, 3260).

Pour un candidat noté par n juges :
  1. trier les notes par ordre décroissant ;
  2. pour un degré de division k, phi = n / 2^k et, pour p = 1..2^k - 1, r(p) = ceil(phi * p) ;
     on retient la note de rang r(p) si ce rang n'a pas déjà été retenu ;
  3. le score est la moyenne arithmétique des notes intermédiaires retenues.
Départage : on commence à k = 2 ; en cas d'égalité on augmente k jusqu'à nu = floor(log2 n) + 1
(indice de division maximale, formule corrigée), où le score coïncide avec la moyenne.
"""
import math
import numpy as np


def nu(n):
    return int(math.floor(math.log2(n))) + 1


def intermediate_ranks(n, k):
    ranks, seen = [], set()
    for p in range(1, 2 ** k):
        r = -(-n * p // 2 ** k)          # plafond exact de n*p/2^k, sans erreur d'arrondi
        if r not in seen:
            seen.add(r)
            ranks.append(r)
    return ranks


def mmcm_score(grades, k):
    g = np.sort(np.asarray(grades, dtype=float))[::-1]
    return float(np.mean([g[r - 1] for r in intermediate_ranks(len(g), k)]))


def mmcm_profile(grades, k0=2):
    """Vecteur lexicographique des scores pour k = k0..nu (utilisé pour le départage)."""
    n = len(grades)
    return tuple(mmcm_score(grades, k) for k in range(k0, max(nu(n), k0) + 1))


def mmcm_ranking(G, names):
    """G : tableau (juges x candidats) de notes, plus grand = meilleur. Retourne les noms classés."""
    prof = {nm: mmcm_profile(G[:, j]) for j, nm in enumerate(names)}
    return sorted(names, key=lambda nm: prof[nm], reverse=True), prof


if __name__ == '__main__':
    # Exemple de l'article (candidat C, six juges) : k = 2 -> 4 ; k = 3 -> 23/6 (après correction)
    C = [5, 5, 4, 4, 3, 2]
    print(mmcm_score(C, 2), mmcm_score(C, 3), 23 / 6)
    assert abs(mmcm_score(C, 2) - 4.0) < 1e-12 and abs(mmcm_score(C, 3) - 23 / 6) < 1e-12
    assert intermediate_ranks(11, 2) == [3, 6, 9] and intermediate_ranks(11, 3) == [2, 3, 5, 6, 7, 9, 10]
    assert nu(8) == 4 and nu(11) == 4 and nu(41) == 6
    print('MMCM : exemples de l\'article reproduits.')
