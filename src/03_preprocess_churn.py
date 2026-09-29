"""
03 - Pretraitement du jeu churn.
Chaine : suppression des colonnes vides d'information -> retrait des variables de fuite
-> traitement de la censure a droite -> valeurs aberrantes -> variables derivees.
Sorties : outputs/churn_clean.csv, outputs/churn_survie.csv, outputs/preprocess_churn.md
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

df = pd.read_csv(ROOT / "data" / "churn_database2.csv")
n0 = len(df)
log = []

# ------------------------------------------------------------------
# 1. Dates de reference
# ------------------------------------------------------------------
df["purge_dt"] = pd.to_datetime(df["PURGE_TIME_ATSGSN"], format="%d/%m/%Y %H:%M")
df["crea_dt"] = pd.to_datetime(df["SUB_CREATE_TIME"], format="%d/%m/%Y %H:%M")
REF = df["purge_dt"].max()
SEUIL_INACTIVITE = 60  # jours, regle metier reconstituee
log.append(f"Date de reference (derniere purge observee) : {REF.date()}")

# ------------------------------------------------------------------
# 2. Colonnes sans information
# ------------------------------------------------------------------
constantes = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
quasi = [c for c in ["TGPPAMBRMAXUL", "TGPPAMBRMAXDL"]
         if df[c].value_counts(normalize=True).iloc[0] > 0.995]
df = df.drop(columns=constantes + quasi)
log.append(f"Colonnes constantes supprimees ({len(constantes)}) : {', '.join(constantes)}")
log.append(f"Colonnes quasi-constantes supprimees ({len(quasi)}) : {', '.join(quasi)}")

# ------------------------------------------------------------------
# 3. Variables de fuite
#    L'etiquette est definie par la date de purge : tout ce qui en derive
#    revele la cible et doit sortir du jeu d'apprentissage.
# ------------------------------------------------------------------
fuite = ["PURGE_TIME_ATSGSN", "MONTHS_SINCE_PURGE", "PURGEDONMME"]
fuite = [c for c in fuite if c in df.columns]
df = df.drop(columns=fuite)
log.append(f"Variables de fuite supprimees : {', '.join(fuite)}")

# ------------------------------------------------------------------
# 4. Redondances exactes
#    TOTAL_CHARGE = MONTHLY_CHARGE x RECHARGE_FREQUENCY (verifie a 100 %)
#    MONTHLY_CHARGE est en bijection avec PLAN_NAME.
# ------------------------------------------------------------------
df = df.drop(columns=["TOTAL_CHARGE"])
log.append("TOTAL_CHARGE supprimee (identite exacte MONTHLY_CHARGE x RECHARGE_FREQUENCY)")

# Latitude/Longitude = centroide regional, redondant avec REGION
df = df.drop(columns=["Latitude", "Longitude"])
log.append("Latitude/Longitude supprimees (centroide regional, redondant avec REGION)")

# ------------------------------------------------------------------
# 5. Valeurs aberrantes et impossibilites metier
# ------------------------------------------------------------------
avant = len(df)
masque_valide = (
    df["AGE"].between(15, 100)
    & (df["TENURE_MONTHS"] >= 0)
    & (df["RECHARGE_FREQUENCY"] >= 0)
    & (df["purge_dt"] >= df["crea_dt"])
)
df = df[masque_valide].copy()
log.append(f"Lignes retirees pour incoherence metier : {avant - len(df)}")

# Winsorisation des queues numeriques a 1 % / 99 % (conserve la ligne, borne la valeur)
for c in ["AGE", "TENURE_MONTHS", "RECHARGE_FREQUENCY"]:
    lo, hi = df[c].quantile([0.01, 0.99])
    n_wins = int(((df[c] < lo) | (df[c] > hi)).sum())
    df[c] = df[c].clip(lo, hi)
    log.append(f"Winsorisation {c} sur [{lo:g}, {hi:g}] : {n_wins} valeurs bornees")

# ------------------------------------------------------------------
# 6. Jeu de survie (AVANT filtrage de la censure : la censure y est une information)
# ------------------------------------------------------------------
surv = df.copy()
surv["duree_jours"] = np.where(
    surv["CHURN_STATUS"] == 1,
    (surv["purge_dt"] - surv["crea_dt"]).dt.days + SEUIL_INACTIVITE,
    (REF - surv["crea_dt"]).dt.days,
)
surv["duree_jours"] = surv["duree_jours"].clip(lower=1)
surv["evenement"] = surv["CHURN_STATUS"]
surv[["PLAN_NAME", "REGION", "GENDER", "AGE", "RECHARGE_FREQUENCY",
      "duree_jours", "evenement"]].to_csv(OUT / "churn_survie.csv", index=False)
log.append(f"Jeu de survie ecrit : {len(surv)} lignes, "
           f"{int(surv['evenement'].sum())} evenements, "
           f"{int((1 - surv['evenement']).sum())} censures")

# ------------------------------------------------------------------
# 7. Censure a droite pour la classification
#    Un abonne cree apres REF - 60 j ne peut pas avoir cumule 60 j d'inactivite :
#    son churn = 0 est un artefact de la fenetre, pas de la fidelite.
# ------------------------------------------------------------------
seuil_crea = REF - pd.Timedelta(days=SEUIL_INACTIVITE)
avant = len(df)
df = df[df["crea_dt"] <= seuil_crea].copy()
log.append(f"Abonnes crees apres {seuil_crea.date()} retires (censure a droite) : "
           f"{avant - len(df)} lignes")

# ------------------------------------------------------------------
# 8. Variables derivees
# ------------------------------------------------------------------
df["anciennete_jours"] = (REF - df["crea_dt"]).dt.days
df["intensite_recharge"] = df["RECHARGE_FREQUENCY"] / df["TENURE_MONTHS"].clip(lower=1)
df["revenu_cumule"] = df["MONTHLY_CHARGE"] * df["RECHARGE_FREQUENCY"]
df["jamais_recharge"] = (df["RECHARGE_FREQUENCY"] == 0).astype(int)
df["mois_creation"] = df["crea_dt"].dt.month
df["tranche_age"] = pd.cut(df["AGE"], [0, 20, 25, 30, 120],
                           labels=["<=20", "21-25", "26-30", ">30"]).astype(str)
log.append("Variables derivees : anciennete_jours, intensite_recharge, revenu_cumule, "
           "jamais_recharge, mois_creation, tranche_age")

df = df.drop(columns=["SUB_CREATE_TIME", "purge_dt", "crea_dt", "MSISDN"])
df.to_csv(OUT / "churn_clean.csv", index=False)

log.append(f"Jeu final : {len(df)} lignes ({len(df)/n0:.1%} de l'original), "
           f"{df.shape[1] - 2} variables explicatives, "
           f"taux de churn {df['CHURN_STATUS'].mean():.2%}")

rapport = "# Pretraitement du jeu churn\n\n" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(log))
(OUT / "preprocess_churn.md").write_text(rapport, encoding="utf-8")
print(rapport)
print("\nColonnes conservees :", list(df.columns))
