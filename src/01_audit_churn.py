"""
01 - Audit du jeu de donnees churn.
Objectif : documenter la qualite des donnees AVANT tout modele.
Sorties : outputs/audit_churn.json, outputs/audit_churn.md
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "churn_database2.csv")
audit = {}

audit["n_lignes"] = int(len(df))
audit["n_colonnes"] = int(df.shape[1])
audit["doublons_complets"] = int(df.duplicated().sum())
audit["doublons_msisdn"] = int(df["MSISDN"].duplicated().sum())
audit["valeurs_manquantes"] = int(df.isna().sum().sum())

# --- colonnes constantes (aucune information) ---
const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
audit["colonnes_constantes"] = const

# --- quasi-constantes (>99.5% d'une seule modalite) ---
quasi = []
for c in df.columns:
    if c in const:
        continue
    top = df[c].value_counts(normalize=True, dropna=False)
    if len(top) > 0 and top.iloc[0] > 0.995:
        quasi.append({"colonne": c, "modalite": str(top.index[0]),
                      "part": round(float(top.iloc[0]), 5)})
audit["colonnes_quasi_constantes"] = quasi

# --- dates ---
purge = pd.to_datetime(df["PURGE_TIME_ATSGSN"], format="%d/%m/%Y %H:%M")
crea = pd.to_datetime(df["SUB_CREATE_TIME"], format="%d/%m/%Y %H:%M")
REF = purge.max()
audit["date_ref"] = str(REF)
audit["purge_min"] = str(purge.min())
audit["creation_min"] = str(crea.min())
audit["creation_max"] = str(crea.max())
audit["purge_avant_creation"] = int((purge < crea).sum())

# --- test de la regle des 60 jours (fuite d'information) ---
gap = (REF - purge).dt.days
regle = (gap >= 60).astype(int)
concord = float((regle == df["CHURN_STATUS"]).mean())
audit["regle_60j_concordance"] = round(concord, 6)
audit["regle_60j_exceptions"] = int((regle != df["CHURN_STATUS"]).sum())

# --- coherence MONTHS_SINCE_PURGE / churn ---
tab = pd.crosstab(df["MONTHS_SINCE_PURGE"], df["CHURN_STATUS"])
audit["months_since_purge_vs_churn"] = tab.to_dict()

# --- censure a droite : abonnes trop recents pour avoir pu churner ---
seuil_censure = REF - pd.Timedelta(days=60)
censures = crea > seuil_censure
audit["seuil_censure"] = str(seuil_censure)
audit["n_censures"] = int(censures.sum())
audit["churners_parmi_censures"] = int(df.loc[censures, "CHURN_STATUS"].sum())

# --- coherence forfait / prix ---
coh = df.groupby("PLAN_NAME")["MONTHLY_CHARGE"].nunique().to_dict()
audit["prix_par_forfait_nunique"] = {k: int(v) for k, v in coh.items()}
audit["prix_par_forfait"] = df.groupby("PLAN_NAME")["MONTHLY_CHARGE"].first().to_dict()

# --- TOTAL_CHARGE = MONTHLY_CHARGE * RECHARGE_FREQUENCY ? ---
attendu = df["MONTHLY_CHARGE"] * df["RECHARGE_FREQUENCY"]
audit["total_charge_identite_verifiee"] = float((attendu == df["TOTAL_CHARGE"]).mean())

# --- geo : une seule coordonnee par region => centroide ---
geo = df.groupby("REGION")[["Latitude", "Longitude"]].nunique()
audit["coord_uniques_par_region"] = int(geo.max().max())

# --- valeurs aberrantes numeriques (methode IQR) ---
num = ["TENURE_MONTHS", "RECHARGE_FREQUENCY", "AGE", "TOTAL_CHARGE", "MONTHLY_CHARGE"]
ab = {}
for c in num:
    q1, q3 = df[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    ab[c] = {"min": float(df[c].min()), "max": float(df[c].max()),
             "borne_basse": float(lo), "borne_haute": float(hi),
             "n_hors_bornes": int(((df[c] < lo) | (df[c] > hi)).sum())}
audit["aberrantes_iqr"] = ab

# --- impossibilites metier ---
audit["age_hors_15_120"] = int(((df["AGE"] < 15) | (df["AGE"] > 120)).sum())
audit["tenure_negative"] = int((df["TENURE_MONTHS"] < 0).sum())
audit["recharge_negative"] = int((df["RECHARGE_FREQUENCY"] < 0).sum())

# --- taux de churn ---
audit["taux_churn_global"] = round(float(df["CHURN_STATUS"].mean()), 4)
audit["churn_par_forfait"] = df.groupby("PLAN_NAME")["CHURN_STATUS"].mean().round(4).to_dict()
audit["repartition_forfaits_churn"] = df["PLAN_NAME"].value_counts(normalize=True).round(4).to_dict()

with open(OUT / "audit_churn.json", "w", encoding="utf-8") as f:
    json.dump(audit, f, ensure_ascii=False, indent=2, default=str)

# --- rapport lisible ---
lignes = [
    "# Audit du jeu de donnees churn", "",
    f"- Lignes : {audit['n_lignes']} | Colonnes : {audit['n_colonnes']}",
    f"- Doublons complets : {audit['doublons_complets']} | doublons MSISDN : {audit['doublons_msisdn']}",
    f"- Valeurs manquantes : {audit['valeurs_manquantes']}",
    f"- Taux de churn : {audit['taux_churn_global']:.2%}",
    "",
    "## Colonnes sans information",
    f"Constantes ({len(const)}) : {', '.join(const) if const else 'aucune'}",
    f"Quasi-constantes : {', '.join(q['colonne'] for q in quasi) if quasi else 'aucune'}",
    "",
    "## Fuite d'information",
    f"Regle 'purge > 60 jours avant {REF.date()}' reproduit l'etiquette dans "
    f"{concord:.4%} des cas ({audit['regle_60j_exceptions']} exceptions).",
    "",
    "## Censure a droite",
    f"Abonnes crees apres {seuil_censure.date()} : {audit['n_censures']} "
    f"dont {audit['churners_parmi_censures']} churners.",
    "",
    "## Coherence",
    f"TOTAL_CHARGE = MONTHLY_CHARGE x RECHARGE_FREQUENCY : {audit['total_charge_identite_verifiee']:.2%} des lignes",
    f"Coordonnees distinctes par region : {audit['coord_uniques_par_region']} (=> centroide regional)",
]
(OUT / "audit_churn.md").write_text("\n".join(lignes), encoding="utf-8")

print("\n".join(lignes))
print("\n--- aberrantes IQR ---")
print(json.dumps(ab, indent=2))
print("\n--- MONTHS_SINCE_PURGE x CHURN ---")
print(tab)
