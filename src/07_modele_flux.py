"""
07 - Modele de flux : jonction du modele de churn et du modele de demande.

Identite de stock :   N_{o,t+1} = N_{o,t} + alpha * A_{o,t} - p_o * N_{o,t}

N : base active du forfait o ; A : activations prevues (modele de demande) ;
alpha : part des activations correspondant a de nouvelles souscriptions ;
p_o : taux de depart mensuel du forfait o (modele de churn).

Deux precautions, sans lesquelles la projection n'a pas de sens :

(1) L'echantillon churn n'est pas un tirage aleatoire de la base. Il contient 44 % de
    churners, ce qui est incompatible avec une base observee stable. Le modele fournit
    donc un CLASSEMENT fiable du risque, pas son NIVEAU absolu. On recale le niveau sur
    un taux de reference p*, en conservant les ecarts relatifs entre forfaits estimes
    par le modele.

(2) alpha n'est pas observable dans les donnees fournies (le classeur ne distingue pas
    souscription et renouvellement). Sur une base stable, l'identite impose
    alpha ~ p* + g, ou g est la croissance mensuelle observee. alpha est donc deduit,
    pas suppose.

Sorties : outputs/modele_flux.csv, outputs/scenarios_flux.csv, outputs/bilan_flux.md,
          figures/flux_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"

PRIX = {"Blue One S": 2000, "Blue One M": 3000, "Blue One L": 5000}
SCENARIOS = {"prudent (p*=2%)": 0.02, "central (p*=5%)": 0.05, "haut (p*=10%)": 0.10,
             "echantillon brut (non recale)": None}
SCEN_CENTRAL = "central (p*=5%)"
HORIZON_MOIS = 3

scores = pd.read_csv(OUT / "scores_churn.csv")
prev = pd.read_csv(OUT / "previsions_demande.csv", parse_dates=["date"])
series = pd.read_csv(OUT / "series_blueone.csv", parse_dates=["date"]).set_index("date")
log = []

# ------------------------------------------------------------------
# 1. Risque relatif par forfait (modele de churn)
# ------------------------------------------------------------------
p_brut_60j = scores.groupby("PLAN_NAME")["proba_churn"].mean()
p_brut_mois = 1 - (1 - p_brut_60j) ** 0.5          # fenetre 60 j -> mensuel
risque_relatif = p_brut_mois / p_brut_mois.mean()
log.append("Churn moyen predit par forfait (fenetre 60 j) : "
           + ", ".join(f"{k} {v:.3f}" for k, v in p_brut_60j.items()))
log.append("Equivalent mensuel : "
           + ", ".join(f"{k} {v:.3f}" for k, v in p_brut_mois.items()))
log.append("Risque relatif conserve lors du recalage : "
           + ", ".join(f"{k} x{v:.2f}" for k, v in risque_relatif.items()))

# ------------------------------------------------------------------
# 2. Base active et croissance observee
# ------------------------------------------------------------------
mensuel = series.resample("ME").sum()
mensuel_complet = mensuel.loc[:"2025-07-31"]       # aout 2025 incomplet dans le classeur
base = mensuel_complet.iloc[-3:].mean()
g = float(mensuel_complet["Blue One M"].pct_change().loc["2024-01":].mean())
log.append("Base active estimee (moyenne mai-juillet 2025, un abonne actif genere "
           "environ une activation par mois) : "
           + ", ".join(f"{k} {v:,.0f}" for k, v in base.items()))
log.append(f"Croissance mensuelle observee (Blue One M, 2024-2025) : {g:+.2%}")
log.append("La stabilite de la base impose alpha ~ p* + g : alpha est deduit de "
           "l'identite de stock, il n'est pas postule.")

# ------------------------------------------------------------------
# 3. Activations prevues (modele de demande), mois complets uniquement
# ------------------------------------------------------------------
prev["mois"] = prev["date"].dt.to_period("M")
jours = prev.groupby("mois")["date"].nunique()
mois_complets = [m for m in jours.index if jours[m] >= 28][:HORIZON_MOIS]
act_prev = (prev[prev["mois"].isin(mois_complets)]
            .groupby(["mois", "offre"])["prevision"].sum().unstack())
log.append("Mois de prevision retenus (complets) : "
           + ", ".join(str(m) for m in mois_complets))
log.append("Activations prevues :\n" + act_prev.round(0).to_string())


# ------------------------------------------------------------------
# 4. Projection
# ------------------------------------------------------------------
def projeter(nom_scenario, p_cible):
    if p_cible is None:                       # niveau issu directement du modele
        p = p_brut_mois.copy()
        alpha = float(p.mean() + g)
    else:                                     # recalage sur p*, ecarts relatifs conserves
        p = (risque_relatif * p_cible).clip(upper=0.95)
        alpha = float(p_cible + g)
    lignes, N = [], base.copy()
    for mois in act_prev.index:
        for o in base.index:
            A = float(act_prev.loc[mois, o])
            entrees = alpha * A
            sorties = float(N[o]) * float(p[o])
            N_fin = N[o] + entrees - sorties
            lignes.append({
                "scenario": nom_scenario, "alpha": round(alpha, 4),
                "mois": str(mois), "forfait": o,
                "taux_depart_mensuel": round(float(p[o]), 4),
                "base_debut": round(float(N[o])),
                "activations_prevues": round(A),
                "entrees": round(entrees),
                "departs_attendus": round(sorties),
                "base_fin": round(float(N_fin)),
                "variation_%": round(100 * (N_fin - N[o]) / N[o], 2),
                "revenu_a_risque_fcfa": round(sorties * PRIX[o]),
                "revenu_base_fcfa": round(float(N[o]) * PRIX[o]),
            })
            N[o] = N_fin
    return pd.DataFrame(lignes)


scen = pd.concat([projeter(k, v) for k, v in SCENARIOS.items()], ignore_index=True)
scen.to_csv(OUT / "scenarios_flux.csv", index=False)
flux = scen[scen["scenario"] == SCEN_CENTRAL].copy()
flux.to_csv(OUT / "modele_flux.csv", index=False)

rar = flux.groupby("mois")["revenu_a_risque_fcfa"].sum()
log.append("Revenu mensuel a risque, scenario central : "
           + ", ".join(f"{k} {v/1e6:.1f} M FCFA" for k, v in rar.items()))

# ------------------------------------------------------------------
# 5. Ciblage : valeur operationnelle du classement
# ------------------------------------------------------------------
s = scores.sort_values("proba_churn", ascending=False).reset_index(drop=True)
s["rang_%"] = (s.index + 1) / len(s) * 100
s["capture_%"] = s["churn_reel"].cumsum() / s["churn_reel"].sum() * 100
taux_base = float(scores["churn_reel"].mean())
lift_max = 1 / taux_base
capture = {}
for p_ in [5, 10, 20, 30, 50]:
    c = float(s.loc[s["rang_%"] <= p_, "capture_%"].max())
    capture[f"top_{p_}%"] = {"capture_%": round(c, 1), "lift": round(c / p_, 2),
                             "lift_max": round(min(lift_max, 100 / p_), 2)}
log.append(f"Taux de churn de l'echantillon : {taux_base:.1%} "
           f"=> lift maximal atteignable {lift_max:.2f}")
log.append("Ciblage : " + json.dumps(capture))

# ------------------------------------------------------------------
# 6. Figures
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.3))
for nom in SCENARIOS:
    d = scen[(scen["scenario"] == nom) & (scen["forfait"] == "Blue One M")]
    vals = [float(base["Blue One M"])] + list(d["base_fin"])
    ax[0].plot(range(len(vals)), np.array(vals) / 1000, "o-", label=nom)
ax[0].axhline(base["Blue One M"] / 1000, ls="--", c="k", lw=0.8)
ax[0].set(title="Blue One M : base active projetee", ylabel="milliers d'abonnes",
          xlabel="mois d'horizon")
ax[0].legend(fontsize=7)
ax[0].grid(alpha=0.3)

piv = flux.pivot(index="mois", columns="forfait", values="revenu_a_risque_fcfa") / 1e6
piv.plot(kind="bar", stacked=True, ax=ax[1])
ax[1].set(title="Revenu mensuel a risque (scenario central)",
          ylabel="millions FCFA", xlabel="")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, axis="y")

ax[2].plot(s["rang_%"], s["capture_%"], c="#2a6f97", lw=1.7, label="modele")
ax[2].plot([0, 100], [0, 100], "k--", lw=0.8, label="ciblage aleatoire")
ax[2].plot([0, 100 * taux_base, 100], [0, 100, 100], c="green", ls=":", lw=1,
           label="ciblage parfait")
ax[2].set(title="Courbe de ciblage des churners", xlabel="% d'abonnes contactes",
          ylabel="% de churners captes")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "flux_synthese.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(11.5, 5.4))
ax.axis("off")
boites = {
    "Donnees abonnes\n13 937 lignes (HSS/NGBSS)": (0.04, 0.76, "#dbe9f4"),
    "Donnees offres\n41 feuilles, 123 offres": (0.04, 0.22, "#dbe9f4"),
    "Audit + nettoyage\nfuite, censure, aberrants": (0.28, 0.76, "#f6e7c8"),
    "Audit + nettoyage\ntrous, Hampel, regime": (0.28, 0.22, "#f6e7c8"),
    "Modele de churn\nLogit, RF, XGBoost\n+ Kaplan-Meier / Cox": (0.52, 0.76, "#d9efd9"),
    "Modele de demande\nSARIMA, Prophet, XGBoost\n+ naif saisonnier": (0.52, 0.22, "#d9efd9"),
    "Modele de flux\nN(t+1) = N(t) + aA - pN": (0.775, 0.49, "#f7d6d6"),
}
for txt, (x, y, c) in boites.items():
    ax.add_patch(plt.Rectangle((x, y - 0.10), 0.195, 0.20, fc=c, ec="#444", lw=1.1,
                               transform=ax.transAxes, zorder=2))
    ax.text(x + 0.0975, y, txt, ha="center", va="center", fontsize=8.2,
            transform=ax.transAxes, zorder=3)
for x1, y1, x2, y2 in [(0.235, 0.76, 0.28, 0.76), (0.235, 0.22, 0.28, 0.22),
                       (0.475, 0.76, 0.52, 0.76), (0.475, 0.22, 0.52, 0.22),
                       (0.715, 0.76, 0.775, 0.57), (0.715, 0.22, 0.775, 0.41)]:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2))
ax.text(0.8725, 0.18, "Sorties decisionnelles\n• base active projetee\n"
                      "• revenu mensuel a risque\n• liste d'abonnes a cibler\n"
                      "• scenarios p* et alpha",
        ha="center", va="center", fontsize=7.8, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#888"))
ax.set_title("Architecture du systeme de prediction CAMTEL", fontsize=12.5, pad=12)
plt.tight_layout()
plt.savefig(FIG / "flux_architecture.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
dernier = scen["mois"].iloc[-1]
comp = (scen[(scen["forfait"] == "Blue One M") & (scen["mois"] == dernier)]
        [["scenario", "alpha", "taux_depart_mensuel", "base_fin", "variation_%",
          "revenu_a_risque_fcfa"]])

rapport = ["# Modele de flux : churn et demande", ""] + [f"- {x}" for x in log]
rapport += ["", "## Projection, scenario central", "",
            flux.drop(columns=["scenario"]).to_markdown(index=False)]
rapport += ["", "## Comparaison des scenarios (dernier mois, Blue One M)", "",
            comp.to_markdown(index=False)]
(OUT / "bilan_flux.md").write_text("\n".join(rapport), encoding="utf-8")

print("\n".join(log))
print("\n--- Scenario central ---")
print(flux.drop(columns=["scenario", "alpha"]).to_string(index=False))
print("\n--- Comparaison des scenarios (dernier mois, Blue One M) ---")
print(comp.to_string(index=False))
