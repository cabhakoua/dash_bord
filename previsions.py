"""
Prévision de la demande en offres : Prophet, XGBoost, LSTM (+ SARIMA en option),
comparés à une référence naïve saisonnière.

Toutes les fonctions de prévision ont la même signature :
    f(train: pd.Series, index_fut: DatetimeIndex) -> pd.Series
ce qui permet de les brancher indifféremment dans le banc d'essai `lancer()`.
"""
import logging
import warnings
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)
logging.getLogger("prophet").setLevel(logging.CRITICAL)

REFERENCE = "Naïf saisonnier"
MODELES_DISPONIBLES = ["Prophet", "XGBoost", "LSTM", "SARIMA"]


# ------------------------------------------------------------------ métriques
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
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return {
        "MAE": round(float(mean_absolute_error(y, yh)), 1),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, yh))), 1),
        "MAPE (%)": round(mape(y, yh), 2),
        "sMAPE (%)": round(smape(y, yh), 2),
        "Biais": round(float(np.mean(yh - y)), 1),
    }


# ------------------------------------------------------- variables calendaires
def _calendrier(idx):
    """Jour de la semaine et jour du mois, codés en sinus / cosinus."""
    w = 2 * np.pi * idx.dayofweek / 7
    m = 2 * np.pi * (idx.day - 1) / 31
    return np.column_stack([np.sin(w), np.cos(w), np.sin(m), np.cos(m)])


def _variables_xgb(s, lags=(1, 2, 3, 7, 14, 21, 28), fen=(7, 14, 28)):
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


# ------------------------------------------------------------------- modèles
def prev_naive(train, index_fut):
    """Référence : on rejoue la dernière semaine observée, jour pour jour."""
    derniers = train.iloc[-7:].values
    return pd.Series([derniers[i % 7] for i in range(len(index_fut))], index=index_fut)


def prev_sarima(train, index_fut):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    r = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return pd.Series(np.asarray(r.forecast(len(index_fut))), index=index_fut)


def prev_prophet(train, index_fut):
    from prophet import Prophet
    d = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.05,
                seasonality_mode="multiplicative")
    m.fit(d)
    fut = pd.DataFrame({"ds": index_fut})
    return pd.Series(m.predict(fut)["yhat"].values, index=index_fut)


def prev_xgb(train, index_fut):
    """Prévision récursive : chaque valeur prédite est réinjectée comme retard."""
    from xgboost import XGBRegressor
    hist = train.copy()
    d = _variables_xgb(hist).dropna()
    cols = [c for c in d.columns if c != "y"]
    mod = XGBRegressor(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    mod.fit(d[cols], d["y"])
    sortie = []
    for date in index_fut:
        hist.loc[date] = np.nan
        ligne = _variables_xgb(hist).loc[[date], cols]
        p = float(mod.predict(ligne)[0])
        hist.loc[date] = p
        sortie.append(p)
    return pd.Series(sortie, index=index_fut)


def prev_lstm(train, index_fut, fenetre=28, epochs=80, graine=42):
    """
    LSTM à une couche, prévision récursive.
    Entrée : `fenetre` derniers jours (valeur normalisée + calendrier du jour à prédire).
    Sortie : la valeur normalisée du lendemain.
    """
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras

    keras.utils.set_random_seed(graine)

    z = np.log1p(train.values.astype(float))
    mu, sd = z.mean(), z.std() or 1.0
    z = (z - mu) / sd

    idx_tout = train.index.append(index_fut)
    cal = _calendrier(idx_tout + pd.Timedelta(days=1))      # calendrier du jour suivant

    def fenetre_a(pos, serie_z):
        v = serie_z[pos - fenetre + 1: pos + 1].reshape(-1, 1)
        return np.hstack([v, cal[pos - fenetre + 1: pos + 1]])

    X = np.stack([fenetre_a(p, z) for p in range(fenetre - 1, len(z) - 1)])
    y = z[fenetre:]

    modele = keras.Sequential([
        keras.layers.Input((fenetre, X.shape[2])),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1),
    ])
    modele.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    modele.fit(X, y, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0,
               callbacks=[keras.callbacks.EarlyStopping(
                   patience=8, restore_best_weights=True)])

    zz = list(z)
    sortie = []
    for k in range(len(index_fut)):
        pos = len(train) - 1 + k
        x = fenetre_a(pos, np.asarray(zz))[None, ...]
        zp = float(modele(x, training=False).numpy()[0, 0])
        zz.append(zp)
        sortie.append(np.expm1(zp * sd + mu))
    return pd.Series(sortie, index=index_fut)


def fabrique(fenetre=28, epochs=80):
    return {
        REFERENCE: prev_naive,
        "SARIMA": prev_sarima,
        "Prophet": prev_prophet,
        "XGBoost": prev_xgb,
        "LSTM": partial(prev_lstm, fenetre=fenetre, epochs=epochs),
    }


# ------------------------------------------------------------------ banc d'essai
def lancer(serie, choix, h_test=60, h_prev=90, fenetre=28, epochs=80, rappel=None):
    """
    Entraîne chaque modèle deux fois : sur la série privée de ses `h_test` derniers
    jours (pour l'évaluer), puis sur la série entière (pour prévoir `h_prev` jours).
    Retourne un dictionnaire prêt à afficher.
    """
    serie = serie.dropna().asfreq("D").interpolate()
    fonctions = fabrique(fenetre, epochs)
    noms = [REFERENCE] + [c for c in choix if c in fonctions]

    train, test = serie.iloc[:-h_test], serie.iloc[-h_test:]
    idx_fut = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=h_prev)

    res = {"metriques": {}, "test": pd.DataFrame({"Réel": test}),
           "futur": {}, "bornes": {}, "erreurs": {}}
    total = len(noms)
    for i, nom in enumerate(noms):
        if rappel:
            rappel(i / total, f"{nom} : entraînement et test")
        try:
            yt = fonctions[nom](train, test.index).clip(lower=0)
            res["test"][nom] = yt.values
            res["metriques"][nom] = metriques(test.values, yt.values)
            if rappel:
                rappel((i + 0.5) / total, f"{nom} : prévision à {h_prev} jours")
            yf = fonctions[nom](serie, idx_fut).clip(lower=0)
            res["futur"][nom] = yf
            q = float(np.quantile(np.abs(test.values - yt.values), 0.9))
            res["bornes"][nom] = q
        except Exception as e:                       # un modèle en échec ne bloque pas les autres
            res["erreurs"][nom] = f"{type(e).__name__}: {e}"
    if rappel:
        rappel(1.0, "Terminé")

    candidats = [n for n in res["metriques"] if n != REFERENCE]
    res["retenu"] = (min(candidats, key=lambda n: res["metriques"][n]["MAPE (%)"])
                     if candidats else None)
    res["h_test"], res["h_prev"] = h_test, h_prev
    return res


def tableau_futur(res, serie_nom):
    """Prévision du modèle retenu avec son intervalle empirique à 90 %."""
    n = res["retenu"]
    f, q = res["futur"][n], res["bornes"][n]
    return pd.DataFrame({
        "date": f.index, "offre": serie_nom, "modele": n,
        "prevision": f.values.round(1),
        "borne_basse": np.maximum(f.values - q, 0).round(1),
        "borne_haute": (f.values + q).round(1)})
