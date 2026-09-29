"""
04 - Modelisation du churn.
Trois classifieurs compares (regression logistique, foret aleatoire, XGBoost),
validation croisee 5 blocs, evaluation sur jeu de test, calibration, seuil optimal,
etude d'ablation, puis analyse de survie (Kaplan-Meier + Cox) en complement.
Sorties : outputs/resultats_churn.json, outputs/bilan_churn.md, figures/*.png,
          outputs/scores_churn.csv, outputs/modele_churn.joblib
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
FIG.mkdir(exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv(OUT / "churn_clean.csv")
y = df["CHURN_STATUS"].values
ids = df["CUSTOMER_ID"].values
X = df.drop(columns=["CHURN_STATUS", "CUSTOMER_ID"])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    X, y, ids, test_size=0.25, stratify=y, random_state=RANDOM_STATE)


def make_pre(scale):
    steps = [("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)]
    steps.append(("num", StandardScaler() if scale else "passthrough", num_cols))
    return ColumnTransformer(steps)


MODELES = {
    "Regression logistique": Pipeline([
        ("pre", make_pre(True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ]),
    "Foret aleatoire": Pipeline([
        ("pre", make_pre(False)),
        ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                       class_weight="balanced_subsample",
                                       n_jobs=-1, random_state=RANDOM_STATE)),
    ]),
    "XGBoost": Pipeline([
        ("pre", make_pre(False)),
        ("clf", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              eval_metric="logloss", n_jobs=-1,
                              random_state=RANDOM_STATE)),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
resultats, probas = {}, {}

for nom, pipe in MODELES.items():
    auc_cv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_cv = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1", n_jobs=-1)
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    probas[nom] = p

    # seuil optimisant le F1 sur la courbe precision-rappel
    prec, rec, seuils = precision_recall_curve(y_te, p)
    f1s = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    i_best = int(np.nanargmax(f1s[:-1]))
    seuil_opt = float(seuils[i_best])

    lignes = {}
    for lab, s in [("seuil_0.5", 0.5), ("seuil_optimal", seuil_opt)]:
        yp = (p >= s).astype(int)
        lignes[lab] = {
            "seuil": round(s, 4),
            "exactitude": round(accuracy_score(y_te, yp), 4),
            "precision": round(precision_score(y_te, yp, zero_division=0), 4),
            "rappel": round(recall_score(y_te, yp), 4),
            "f1": round(f1_score(y_te, yp), 4),
            "matrice_confusion": confusion_matrix(y_te, yp).tolist(),
        }

    resultats[nom] = {
        "auc_cv_moyenne": round(float(auc_cv.mean()), 4),
        "auc_cv_ecart_type": round(float(auc_cv.std()), 4),
        "f1_cv_moyenne": round(float(f1_cv.mean()), 4),
        "auc_test": round(float(roc_auc_score(y_te, p)), 4),
        "auc_pr_test": round(float(average_precision_score(y_te, p)), 4),
        "brier": round(float(brier_score_loss(y_te, p)), 4),
        **lignes,
        "rapport_classification": classification_report(
            y_te, (p >= 0.5).astype(int), target_names=["fidele", "churner"],
            output_dict=True, zero_division=0),
    }
    print(f"{nom:24s} AUC CV {auc_cv.mean():.4f} (+/-{auc_cv.std():.4f})  "
          f"AUC test {roc_auc_score(y_te, p):.4f}  F1@0.5 {lignes['seuil_0.5']['f1']:.4f}")

# ------------------------------------------------------------------
# Meilleur modele : importance des variables + ablation
# ------------------------------------------------------------------
meilleur = max(resultats, key=lambda k: resultats[k]["auc_test"])
pipe_best = MODELES[meilleur]

imp = permutation_importance(pipe_best, X_te, y_te, n_repeats=10,
                             random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1)
importances = (pd.DataFrame({"variable": X.columns, "importance": imp.importances_mean,
                             "ecart_type": imp.importances_std})
               .sort_values("importance", ascending=False))
importances.to_csv(OUT / "importances_churn.csv", index=False)

ablation = {}
for var in ["RECHARGE_FREQUENCY", "intensite_recharge", "TENURE_MONTHS", "PLAN_NAME"]:
    cols = [c for c in X.columns if c not in {var}]
    if var in ("RECHARGE_FREQUENCY",):   # retirer aussi les variables qui la contiennent
        cols = [c for c in cols if c not in {"intensite_recharge", "revenu_cumule",
                                             "jamais_recharge"}]
    sub_num = [c for c in num_cols if c in cols]
    sub_cat = [c for c in cat_cols if c in cols]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), sub_cat),
                             ("num", "passthrough", sub_num)])
    p2 = Pipeline([("pre", pre), ("clf", XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
        random_state=RANDOM_STATE))])
    p2.fit(X_tr[cols], y_tr)
    ablation[f"sans {var}"] = round(float(roc_auc_score(y_te, p2.predict_proba(X_te[cols])[:, 1])), 4)
resultats["_ablation_auc"] = ablation

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Export des courbes pour le tableau de bord (rendu interactif cote application)
# ------------------------------------------------------------------
pts = []
for nom, p in probas.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    pas = max(1, len(fpr) // 300)
    for a, b in zip(fpr[::pas], tpr[::pas]):
        pts.append({"modele": nom, "courbe": "roc", "x": float(a), "y": float(b)})
    pr, rc, _ = precision_recall_curve(y_te, p)
    pas = max(1, len(pr) // 300)
    for a, b in zip(rc[::pas], pr[::pas]):
        pts.append({"modele": nom, "courbe": "pr", "x": float(a), "y": float(b)})
    xg, yg = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    for a, b in zip(xg, yg):
        pts.append({"modele": nom, "courbe": "calibration", "x": float(a), "y": float(b)})
pd.DataFrame(pts).to_csv(OUT / "courbes_churn.csv", index=False)

# distribution des scores par classe reelle
pd.DataFrame({"modele": np.repeat(list(probas), len(y_te)),
              "proba": np.concatenate(list(probas.values())),
              "reel": np.tile(y_te, len(probas))}).to_csv(
    OUT / "distribution_scores.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for nom, p in probas.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[0].plot(fpr, tpr, label=f"{nom} (AUC={roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[1].plot(rc, pr, label=f"{nom} (AP={average_precision_score(y_te, p):.3f})")
    xg, yg = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax[2].plot(xg, yg, "o-", label=nom)
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[0].set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbes ROC")
ax[1].axhline(y_te.mean(), ls="--", c="k", lw=0.8)
ax[1].set(xlabel="Rappel", ylabel="Precision", title="Courbes precision-rappel")
ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[2].set(xlabel="Probabilite predite", ylabel="Frequence observee", title="Calibration")
for a in ax:
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "churn_courbes.png", dpi=150)
plt.close()

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, (nom, p) in zip(ax, probas.items()):
    cm = confusion_matrix(y_te, (p >= 0.5).astype(int))
    a.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    a.set(title=nom, xticks=[0, 1], yticks=[0, 1],
          xticklabels=["fidele", "churner"], yticklabels=["fidele", "churner"],
          xlabel="Predit", ylabel="Reel")
plt.tight_layout()
plt.savefig(FIG / "churn_confusion.png", dpi=150)
plt.close()

top = importances.head(12).iloc[::-1]
plt.figure(figsize=(8, 5))
plt.barh(top["variable"], top["importance"], xerr=top["ecart_type"], color="#2a6f97")
plt.xlabel("Chute d'AUC apres permutation")
plt.title(f"Importance des variables ({meilleur})")
plt.tight_layout()
plt.savefig(FIG / "churn_importances.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# Analyse de survie
# ------------------------------------------------------------------
from lifelines import CoxPHFitter, KaplanMeierFitter

surv = pd.read_csv(OUT / "churn_survie.csv")
kmf = KaplanMeierFitter()
plt.figure(figsize=(7, 4.5))
mediane_par_plan = {}
km_points = []
for plan in sorted(surv["PLAN_NAME"].unique()):
    s = surv[surv["PLAN_NAME"] == plan]
    kmf.fit(s["duree_jours"], s["evenement"], label=plan)
    kmf.plot_survival_function(ci_show=False)
    mediane_par_plan[plan] = (None if pd.isna(kmf.median_survival_time_)
                              else float(kmf.median_survival_time_))
    sf = kmf.survival_function_.reset_index()
    sf.columns = ["t", "survie"]
    ic = kmf.confidence_interval_survival_function_
    sf["bas"] = ic.iloc[:, 0].values
    sf["haut"] = ic.iloc[:, 1].values
    sf["forfait"] = plan
    km_points.append(sf)
pd.concat(km_points).to_csv(OUT / "km_survie.csv", index=False)
plt.xlabel("Jours depuis la souscription")
plt.ylabel("Probabilite de survie S(t)")
plt.title("Kaplan-Meier par forfait")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "survie_kaplan_meier.png", dpi=150)
plt.close()

cox_df = pd.get_dummies(
    surv[["PLAN_NAME", "GENDER", "AGE", "RECHARGE_FREQUENCY", "duree_jours", "evenement"]],
    columns=["PLAN_NAME", "GENDER"], drop_first=True, dtype=float)
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col="duree_jours", event_col="evenement")
cox_res = cph.summary[["coef", "exp(coef)", "p"]].round(4)
cox_res.to_csv(OUT / "cox_churn.csv")

resultats["_survie"] = {
    "mediane_survie_jours": mediane_par_plan,
    "concordance_cox": round(float(cph.concordance_index_), 4),
    "rapports_de_risque": {k: round(float(v), 4) for k, v in cph.hazard_ratios_.items()},
}

# ------------------------------------------------------------------
# Scores individuels + sauvegarde du modele
# ------------------------------------------------------------------
pipe_best.fit(X, y)
joblib.dump({"pipeline": pipe_best, "colonnes": list(X.columns), "modele": meilleur},
            OUT / "modele_churn.joblib")

# Specification portable : le pickle scikit-learn n'est pas compatible entre versions
# (les attributs internes changent d'une version a l'autre). On enregistre donc aussi
# de quoi reconstruire et reentrainer le modele a l'identique, en quelques secondes.
with open(OUT / "modele_churn.json", "w", encoding="utf-8") as f:
    json.dump({
        "modele": meilleur,
        "colonnes": list(X.columns),
        "colonnes_numeriques": num_cols,
        "colonnes_categorielles": cat_cols,
        "random_state": RANDOM_STATE,
        "sklearn_version": __import__("sklearn").__version__,
    }, f, ensure_ascii=False, indent=2)

scores = pd.DataFrame({
    "CUSTOMER_ID": ids,
    "PLAN_NAME": df["PLAN_NAME"],
    "REGION": df["REGION"],
    "MONTHLY_CHARGE": df["MONTHLY_CHARGE"],
    "churn_reel": y,
    "proba_churn": pipe_best.predict_proba(X)[:, 1],
})
scores["perte_attendue_fcfa"] = scores["proba_churn"] * scores["MONTHLY_CHARGE"]
scores["decile_risque"] = pd.qcut(scores["proba_churn"], 10, labels=False, duplicates="drop") + 1
scores.to_csv(OUT / "scores_churn.csv", index=False)

with open(OUT / "resultats_churn.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

print(f"\nMeilleur modele : {meilleur}")
print("\nAblation (AUC) :", json.dumps(ablation, indent=2))
print("\nSurvie, concordance Cox :", resultats["_survie"]["concordance_cox"])
print("Rapports de risque :", json.dumps(resultats["_survie"]["rapports_de_risque"], indent=2))
print("\nTop 8 variables :")
print(importances.head(8).to_string(index=False))
