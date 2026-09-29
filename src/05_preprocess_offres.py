"""
05 - Pretraitement des series journalieres d'activations Blue One.
Trous de collecte, valeurs aberrantes (filtre de Hampel), choix du regime stable.
Sorties : outputs/series_blueone.csv, outputs/preprocess_offres.md, figures/series_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)

FORFAITS = ["Blue One S", "Blue One M", "Blue One L"]
DEBUT_STABLE = "2023-10-01"   # fin de la phase de montee en charge (cf. moyennes trimestrielles)
FIN_FIABLE = "2025-08-26"     # derniere journee renseignee du classeur

long = pd.read_csv(OUT / "offres_long.csv", parse_dates=["date"])
log, series, diag = [], {}, []

log.append(f"Regime retenu pour la modelisation : {DEBUT_STABLE} -> {FIN_FIABLE}")
log.append("Justification : avant octobre 2023 les forfaits Blue One sont en phase de "
           "lancement (la moyenne journaliere de Blue One M passe de 107 a 7 260 entre "
           "T2 2022 et T4 2023). Melanger lancement et regime etabli fausse toute prevision.")

for offre in FORFAITS:
    s = (long[long["offre"] == offre]
         .set_index("date")["activations"]
         .asfreq("D"))
    n_trous = int(s.isna().sum())

    # --- filtre de Hampel : mediane glissante + ecart absolu median ---
    med = s.rolling(15, center=True, min_periods=5).median()
    mad = (s - med).abs().rolling(15, center=True, min_periods=5).median()
    seuil = 3 * 1.4826 * mad
    aberrant = (s - med).abs() > seuil.replace(0, np.nan)
    aberrant = aberrant.fillna(False)
    n_aber = int(aberrant.sum())

    s_corr = s.copy()
    s_corr[aberrant] = med[aberrant]

    # --- trous : interpolation lineaire puis remplissage des bords ---
    s_corr = s_corr.interpolate("linear", limit_direction="both")
    s_corr = s_corr.round().clip(lower=0)

    dates_aber = list(s.index[aberrant].strftime("%Y-%m-%d"))
    log.append(f"{offre} : {len(s)} jours, {n_trous} trou(s) comble(s), "
               f"{n_aber} valeur(s) aberrante(s) remplacee(s) par la mediane locale")
    if dates_aber:
        log.append(f"    dates concernees : {', '.join(dates_aber[:12])}"
                   + (" ..." if len(dates_aber) > 12 else ""))

    series[offre] = pd.DataFrame({"brut": s, "corrige": s_corr})
    diag.append(pd.DataFrame({"date": s.index, "offre": offre, "brut": s.values,
                              "corrige": s_corr.values, "aberrant": aberrant.values,
                              "manquant": s.isna().values}))

    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(s.index, s.values, lw=0.7, alpha=0.45, label="brut")
    ax.plot(s_corr.index, s_corr.values, lw=0.9, label="corrige")
    ax.scatter(s.index[aberrant], s[aberrant], s=18, c="crimson", zorder=5,
               label="aberrant detecte")
    ax.axvline(pd.Timestamp(DEBUT_STABLE), ls="--", c="k", lw=0.9)
    ax.text(pd.Timestamp(DEBUT_STABLE), ax.get_ylim()[1] * 0.9, " debut regime stable",
            fontsize=8)
    ax.set(title=f"Activations journalieres : {offre}", ylabel="activations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / f"serie_{offre.replace(' ', '_')}.png", dpi=150)
    plt.close()

pd.concat(diag).to_csv(OUT / "series_diagnostic.csv", index=False)

tab = pd.concat({o: d["corrige"] for o, d in series.items()}, axis=1)
tab.index.name = "date"
tab = tab.loc[DEBUT_STABLE:FIN_FIABLE]
tab.to_csv(OUT / "series_blueone.csv")

log.append(f"Table finale : {len(tab)} jours x {tab.shape[1]} forfaits "
           f"({tab.index.min().date()} -> {tab.index.max().date()})")
log.append("Moyennes journalieres sur le regime retenu : "
           + ", ".join(f"{c} = {tab[c].mean():,.0f}" for c in tab.columns))

# saisonnalite hebdomadaire (utile pour justifier les variables calendaires)
jours = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
prof = tab["Blue One M"].groupby(tab.index.dayofweek).mean()
prof = (prof / prof.mean() * 100).round(1)
log.append("Profil hebdomadaire Blue One M (base 100) : "
           + ", ".join(f"{jours[i]} {prof[i]}" for i in prof.index))

rapport = "# Pretraitement des series d'offres\n\n" + "\n".join(f"- {x}" for x in log)
(OUT / "preprocess_offres.md").write_text(rapport, encoding="utf-8")
print(rapport)
