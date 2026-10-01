"""
06 - Prevision de la demande en offres (activations journalieres Blue One).
Trois modeles compares (SARIMA, Prophet, XGBoost sur retards + calendrier) plus une
reference naive saisonniere. Validation par origine glissante, test hors echantillon,
puis prevision a 90 jours.
Sorties : outputs/resultats_demande.json, outputs/previsions_demande.csv,
          figures/demande_*.png
"""
import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
from prophet import Prophet  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "figures"
HORIZON_TEST = 60      # jours reserves a l'evaluation
HORIZON_PREV = 90      # jours de prevision au-dela des donnees

tab = pd.read_csv(OUT / "series_blueone.csv", parse_dates=["date"]).set_index("date")
tab = tab.asfreq("D")


# ------------------------------------------------------------------ metriques
def mape(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yh[m]) / y[m])) * 100)


def smape(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    d = (np.abs(y) + np.abs(yh)) / 2
    m = d != 0
    return float(np.mean(np.abs(y[m] - yh[m]) / d[m]) * 100)


def metriques(y, yh):
    return {
        "MAE": round(float(mean_absolute_error(y, yh)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, yh))), 1),
        "MAPE_%": round(mape(y, yh), 2),
        "sMAPE_%": round(smape(y, yh), 2),
        "biais": round(float(np.mean(np.asarray(yh, float) - np.asarray(y, float))), 1),
    }


# ------------------------------------------------------------- variables XGB
def features(s, lags=(1, 2, 3, 7, 14, 21, 28), fen=(7, 14, 28)):
    d = pd.DataFrame({"y": s})
    for l in lags:
        d[f"lag_{l}"] = d["y"].shift(l)
    for f in fen:
        d[f"moy_{f}"] = d["y"].shift(1).rolling(f).mean()
        d[f"ect_{f}"] = d["y"].shift(1).rolling(f).std()
    idx = d.index
    d["jour_sem"] = idx.dayofweek
    d["jour_mois"] = idx.day
    d["mois"] = idx.month
    d["semaine"] = idx.isocalendar().week.astype(int)
    d["weekend"] = (idx.dayofweek >= 5).astype(int)
    d["debut_mois"] = (idx.day <= 5).astype(int)
    d["fin_mois"] = (idx.day >= 26).astype(int)
    d["tendance"] = np.arange(len(d))
    return d


# ------------------------------------------------------------------ modeles
def prev_naive(train, h, index_fut):
    """Reference : meme jour de la semaine, derniere semaine observee."""
    derniers = train.iloc[-7:].values
    return pd.Series([derniers[i % 7] for i in range(h)], index=index_fut)


def prev_sarima(train, h, index_fut):
    m = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False)
    r = m.fit(disp=False)
    return pd.Series(np.asarray(r.forecast(h)), index=index_fut)


def prev_prophet(train, h, index_fut):
    d = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05,
                seasonality_mode="multiplicative")
    m.fit(d)
    fut = pd.DataFrame({"ds": index_fut})
    return pd.Series(m.predict(fut)["yhat"].values, index=index_fut)


def prev_xgb(train, h, index_fut):
    """Prevision recursive : chaque pas reinjecte la valeur predite."""
    hist = train.copy()
    d = features(hist).dropna()
    Xc = [c for c in d.columns if c != "y"]
    mod = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                       random_state=42)
    mod.fit(d[Xc], d["y"])
    out = []
    for date in index_fut:
        hist.loc[date] = np.nan
        ligne = features(hist).loc[[date], Xc]
        p = float(mod.predict(ligne)[0])
        hist.loc[date] = p
        out.append(p)
    return pd.Series(out, index=index_fut), mod, Xc


MODELES = {
    "Naif saisonnier (reference)": prev_naive,
    "SARIMA (2,1,2)(1,1,1)7": prev_sarima,
    "Prophet": prev_prophet,
    "XGBoost (retards + calendrier)": None,   # traite a part (retourne le modele)
    
}

resultats, previsions_test = {}, {}

for offre in tab.columns:
    s = tab[offre].dropna()
    train, test = s.iloc[:-HORIZON_TEST], s.iloc[-HORIZON_TEST:]
    resultats[offre] = {}
    previsions_test[offre] = pd.DataFrame({"reel": test})

    for nom, fn in MODELES.items():
        if nom.startswith("XGBoost"):
            yh, _, _ = prev_xgb(train, HORIZON_TEST, test.index)
        else:
            yh = fn(train, HORIZON_TEST, test.index)
        yh = yh.clip(lower=0)
        resultats[offre][nom] = metriques(test.values, yh.values)
        previsions_test[offre][nom] = yh.values

    # --- validation par origine glissante (3 blocs de 30 jours) sur le meilleur cadre ---
    glissant = {n: [] for n in MODELES}
    for k in (3, 2, 1):
        fin = len(s) - 30 * (k - 1)
        tr = s.iloc[:fin - 30]
        te = s.iloc[fin - 30:fin]
        for nom, fn in MODELES.items():
            try:
                yh = (prev_xgb(tr, 30, te.index)[0] if nom.startswith("XGBoost")
                      else fn(tr, 30, te.index)).clip(lower=0)
                glissant[nom].append(mape(te.values, yh.values))
            except Exception:
                glissant[nom].append(np.nan)
    for nom in MODELES:
        resultats[offre][nom]["MAPE_origine_glissante_%"] = round(
            float(np.nanmean(glissant[nom])), 2)

    print(f"\n--- {offre} ---")
    print(pd.DataFrame(resultats[offre]).T.to_string())

# ------------------------------------------------------------------ figures
export = []
for offre, d in previsions_test.items():
    e = d.reset_index().melt("date", var_name="serie", value_name="valeur")
    e["offre"] = offre
    export.append(e)
pd.concat(export).to_csv(OUT / "previsions_test.csv", index=False)

fig, axes = plt.subplots(len(tab.columns), 1, figsize=(12, 3.4 * len(tab.columns)),
                         squeeze=False)
for ax, offre in zip(axes[:, 0], tab.columns):
    d = previsions_test[offre]
    ctx = tab[offre].iloc[-HORIZON_TEST - 90:-HORIZON_TEST]
    ax.plot(ctx.index, ctx.values, c="grey", lw=0.8, label="historique")
    ax.plot(d.index, d["reel"], c="black", lw=1.4, label="reel")
    for col in d.columns[1:]:
        ax.plot(d.index, d[col], lw=1.0, ls="--", label=col)
    ax.set(title=f"{offre} : test hors echantillon ({HORIZON_TEST} j)",
           ylabel="activations/jour")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "demande_test.png", dpi=150)
plt.close()

# ------------------------------------------------------- prevision a 90 jours
lignes_prev = []
for offre in tab.columns:
    s = tab[offre].dropna()
    index_fut = pd.date_range(s.index[-1] + pd.Timedelta(days=1),
                              periods=HORIZON_PREV, freq="D")
    meilleur = min(
        (n for n in MODELES if not n.startswith("Naif")),
        key=lambda n: resultats[offre][n]["MAPE_%"])
    if meilleur.startswith("XGBoost"):
        yh, _, _ = prev_xgb(s, HORIZON_PREV, index_fut)
    else:
        yh = MODELES[meilleur](s, HORIZON_PREV, index_fut)
    yh = yh.clip(lower=0)

    # intervalle empirique a partir de l'erreur du modele sur le test
    err = previsions_test[offre]["reel"].values - previsions_test[offre][meilleur].values
    q = np.quantile(np.abs(err), 0.9)
    for date, v in yh.items():
        lignes_prev.append({"date": date, "offre": offre, "modele": meilleur,
                            "prevision": round(float(v), 1),
                            "borne_basse": round(float(max(v - q, 0)), 1),
                            "borne_haute": round(float(v + q), 1)})
    resultats[offre]["_modele_retenu"] = meilleur

prev = pd.DataFrame(lignes_prev)
prev.to_csv(OUT / "previsions_demande.csv", index=False)

fig, axes = plt.subplots(len(tab.columns), 1, figsize=(12, 3.4 * len(tab.columns)),
                         squeeze=False)
for ax, offre in zip(axes[:, 0], tab.columns):
    hist = tab[offre].iloc[-180:]
    p = prev[prev["offre"] == offre].set_index("date")
    ax.plot(hist.index, hist.values, c="#2a6f97", lw=0.9, label="observe")
    ax.plot(p.index, p["prevision"], c="crimson", lw=1.2, label="prevision")
    ax.fill_between(p.index, p["borne_basse"], p["borne_haute"], color="crimson",
                    alpha=0.15, label="intervalle 90 %")
    ax.set(title=f"{offre} : prevision a {HORIZON_PREV} jours "
                 f"({resultats[offre]['_modele_retenu']})", ylabel="activations/jour")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "demande_prevision.png", dpi=150)
plt.close()

with open(OUT / "resultats_demande.json", "w", encoding="utf-8") as f:
    json.dump(resultats, f, ensure_ascii=False, indent=2)

print("\n=== Modeles retenus ===")
for offre in tab.columns:
    print(f"{offre:14s} -> {resultats[offre]['_modele_retenu']}")
print("\nTotal prevu sur 90 jours :")
print(prev.groupby("offre")["prevision"].sum().round(0).to_string())
