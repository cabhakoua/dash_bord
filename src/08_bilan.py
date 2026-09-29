"""
08 - Bilan consolide : tableaux de performance prets a inserer dans le memoire.
Sorties : outputs/bilan_general.md, outputs/tableau_churn.csv,
          outputs/tableau_demande.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

rc = json.load(open(OUT / "resultats_churn.json", encoding="utf-8"))
rd = json.load(open(OUT / "resultats_demande.json", encoding="utf-8"))
audit = json.load(open(OUT / "audit_churn.json", encoding="utf-8"))

# ------------------------------------------------------------------ churn
lignes = []
for nom, d in rc.items():
    if nom.startswith("_"):
        continue
    cm = d["seuil_0.5"]["matrice_confusion"]
    rap = d["rapport_classification"]
    lignes.append({
        "Modele": nom,
        "AUC (VC 5 blocs)": f"{d['auc_cv_moyenne']:.4f} ± {d['auc_cv_ecart_type']:.4f}",
        "AUC (test)": d["auc_test"],
        "AUC-PR (test)": d["auc_pr_test"],
        "Exactitude": d["seuil_0.5"]["exactitude"],
        "Precision (churn)": d["seuil_0.5"]["precision"],
        "Rappel (churn)": d["seuil_0.5"]["rappel"],
        "F1 (churn)": d["seuil_0.5"]["f1"],
        "Precision (fidele)": round(rap["fidele"]["precision"], 4),
        "Rappel (fidele)": round(rap["fidele"]["recall"], 4),
        "Brier": d["brier"],
        "VN / FP / FN / VP": f"{cm[0][0]} / {cm[0][1]} / {cm[1][0]} / {cm[1][1]}",
    })
t_churn = pd.DataFrame(lignes)
t_churn.to_csv(OUT / "tableau_churn.csv", index=False)

# deux vues etroites, lisibles une fois exportees en Word
t_disc = t_churn[["Modele", "AUC (VC 5 blocs)", "AUC (test)", "AUC-PR (test)", "Brier"]]
t_clas = t_churn[["Modele", "Exactitude", "Precision (churn)", "Rappel (churn)",
                  "F1 (churn)", "Precision (fidele)", "Rappel (fidele)",
                  "VN / FP / FN / VP"]]

lignes_opt = []
for nom, d in rc.items():
    if nom.startswith("_"):
        continue
    o = d["seuil_optimal"]
    lignes_opt.append({"Modele": nom, "Seuil": o["seuil"], "Exactitude": o["exactitude"],
                       "Precision": o["precision"], "Rappel": o["rappel"], "F1": o["f1"]})
t_opt = pd.DataFrame(lignes_opt)

# ------------------------------------------------------------------ demande
lignes = []
for offre, modeles in rd.items():
    for nom, m in modeles.items():
        if nom.startswith("_"):
            continue
        lignes.append({"Forfait": offre, "Modele": nom, **m,
                       "Retenu": "oui" if modeles.get("_modele_retenu") == nom else ""})
t_dem = pd.DataFrame(lignes)
t_dem.to_csv(OUT / "tableau_demande.csv", index=False)

# ------------------------------------------------------------------ rapport
flux = pd.read_csv(OUT / "modele_flux.csv")
scen = pd.read_csv(OUT / "scenarios_flux.csv")
imp = pd.read_csv(OUT / "importances_churn.csv").head(8)
cox = pd.read_csv(OUT / "cox_churn.csv")

md = [
    "# Bilan general", "",
    "## 1. Qualite des donnees", "",
    f"- Jeu churn : {audit['n_lignes']} lignes, {audit['n_colonnes']} colonnes, "
    f"aucune valeur manquante, aucun doublon.",
    f"- {len(audit['colonnes_constantes'])} colonnes constantes et "
    f"{len(audit['colonnes_quasi_constantes'])} quasi-constantes retirees.",
    f"- Fuite d'information : la regle « purge anterieure de plus de 60 jours a "
    f"{audit['date_ref'][:10]} » reproduit l'etiquette dans "
    f"{audit['regle_60j_concordance']:.4%} des cas "
    f"({audit['regle_60j_exceptions']} exceptions). Les variables derivees de la purge "
    "sont donc exclues de l'apprentissage.",
    f"- Censure a droite : {audit['n_censures']} abonnes crees apres "
    f"{audit['seuil_censure'][:10]} ne pouvaient pas cumuler 60 jours d'inactivite "
    f"({audit['churners_parmi_censures']} churner parmi eux). Ils sont retires de la "
    "classification et conserves comme observations censurees pour l'analyse de survie.",
    f"- Identite verifiee a {audit['total_charge_identite_verifiee']:.0%} : "
    "TOTAL_CHARGE = MONTHLY_CHARGE x RECHARGE_FREQUENCY (colonne redondante).",
    "- Jeu offres : 41 feuilles harmonisees, 123 offres apres normalisation des libelles "
    "(SPOT/Blue mo S, COOL/Blue mo M, BLUE Booster/Blue mo L, KOLO/Blue mo XL, "
    "SWIM/Blue mo XXL fusionnes).",
    "", "## 2. Modeles de churn", "",
    "Jeu de test : 25 % des observations, stratifie. Seuil de decision 0,5.", "",
    "### Pouvoir discriminant", "",
    t_disc.to_markdown(index=False), "",
    "### Metriques de classification au seuil 0,5", "",
    t_clas.to_markdown(index=False), "",
    "(VN / FP / FN / VP : vrais negatifs, faux positifs, faux negatifs, vrais positifs.)",
    "", "### Au seuil optimisant le F1", "",
    t_opt.to_markdown(index=False), "",
    "### Etude d'ablation (AUC du meilleur modele, variable retiree)", "",
    pd.DataFrame(list(rc["_ablation_auc"].items()),
                 columns=["Variante", "AUC"]).to_markdown(index=False), "",
    "### Importance des variables (permutation, chute d'AUC)", "",
    imp.round(4).to_markdown(index=False), "",
    "### Analyse de survie", "",
    f"- Indice de concordance du modele de Cox : {rc['_survie']['concordance_cox']}",
    "- Rapports de risque (reference : Blue One L) :", "",
    cox.to_markdown(index=False), "",
    "- Duree mediane de survie : "
    + ", ".join(f"{k} = {'non atteinte sur la fenetre' if (v is None or v != v or v == float('inf')) else f'{v:.0f} jours'}"
                for k, v in rc["_survie"]["mediane_survie_jours"].items()),
    "", "## 3. Modeles de demande", "",
    "Test hors echantillon : 60 derniers jours. Validation par origine glissante : "
    "3 blocs de 30 jours.", "",
    t_dem.to_markdown(index=False), "",
    "## 4. Modele de flux", "",
    "Scenario central (taux de depart de reference p* = 5 % par mois, ecarts entre "
    "forfaits issus du modele de churn) :", "",
    flux.drop(columns=["scenario"]).to_markdown(index=False), "",
    "### Sensibilite", "",
    scen[scen["forfait"] == "Blue One M"][
        ["scenario", "alpha", "taux_depart_mensuel", "mois", "base_fin",
         "variation_%", "revenu_a_risque_fcfa"]].to_markdown(index=False), "",
    "## 5. Limites", "",
    "- L'echantillon churn (44 % de churners) n'est pas un tirage aleatoire de la base : "
    "le modele fournit un classement fiable, mais le niveau absolu du risque doit etre "
    "recale sur une reference externe.",
    "- Le mix des forfaits differe fortement entre l'echantillon churn (68 % de M) et "
    "les activations reelles de 2025 (93 % de M).",
    "- La part des activations correspondant a de nouvelles souscriptions n'est pas "
    "observable dans le classeur ; elle est deduite de l'identite de stock, sous "
    "l'hypothese d'un abonne generant une activation par mois. Une confirmation aupres "
    "de CAMTEL leverait cette incertitude.",
    "- La fenetre du fichier churn (janvier-juillet 2025) est courte : sept mois "
    "d'historique ne permettent pas d'observer de saisonnalite annuelle du churn.",
]
(OUT / "bilan_general.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
