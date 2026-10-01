"""
Console de retention CAMTEL : churn et demande en offres.
Lancement :  streamlit run app.py
"""
import io
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import departs as DP
import pipeline as PL
import previsions as PV
import theme as T

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "outputs", ROOT / "figures"
PRIX = {"Blue One S": 2000, "Blue One M": 3000, "Blue One L": 5000}

st.set_page_config(page_title="CAMTEL, rétention", layout="wide",
                   page_icon="◆", initial_sidebar_state="expanded")
T.appliquer()



# ================================================================== chargement
@st.cache_data(show_spinner=False)
def charger():
    lire = lambda n, **k: pd.read_csv(OUT / n, **k)
    d = {
        "scores": lire("scores_churn.csv"),
        "series": lire("series_blueone.csv", parse_dates=["date"]).set_index("date"),
        "diag": lire("series_diagnostic.csv", parse_dates=["date"]),
        "prev": lire("previsions_demande.csv", parse_dates=["date"]),
        "prev_test": lire("previsions_test.csv", parse_dates=["date"]),
        "flux": lire("modele_flux.csv"),
        "scen": lire("scenarios_flux.csv"),
        "t_churn": lire("tableau_churn.csv"),
        "t_dem": lire("tableau_demande.csv"),
        "imp": lire("importances_churn.csv"),
        "cox": lire("cox_churn.csv"),
        "courbes": lire("courbes_churn.csv"),
        "distrib": lire("distribution_scores.csv"),
        "km": lire("km_survie.csv"),
    }
    d["audit"] = json.load(open(OUT / "audit_churn.json", encoding="utf-8"))
    d["rc"] = json.load(open(OUT / "resultats_churn.json", encoding="utf-8"))
    return d


@st.cache_resource(show_spinner=False)
def charger_modele():
    """Le pickle scikit-learn n'est pas portable entre versions : on le valide,
    et on reconstruit le modele depuis les donnees nettoyees s'il est illisible."""
    spec_path = OUT / "modele_churn.json"
    spec = json.load(open(spec_path, encoding="utf-8")) if spec_path.exists() else None
    try:
        m = joblib.load(OUT / "modele_churn.joblib")
        m["pipeline"].predict_proba(pd.read_csv(OUT / "churn_clean.csv", nrows=1)[m["colonnes"]])
        m["origine"] = "charge"
        return m
    except Exception as e:
        erreur = e
    if spec is None:
        st.error(f"Modèle illisible et spécification absente : {erreur}")
        return None

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from xgboost import XGBClassifier

    df = pd.read_csv(OUT / "churn_clean.csv")
    y, X = df["CHURN_STATUS"].values, df[spec["colonnes"]]
    num, cat, rs = spec["colonnes_numeriques"], spec["colonnes_categorielles"], spec["random_state"]
    pre = lambda sc: ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat),
        ("num", StandardScaler() if sc else "passthrough", num)])
    fab = {
        "Regression logistique": lambda: Pipeline([("pre", pre(True)), ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs))]),
        "Foret aleatoire": lambda: Pipeline([("pre", pre(False)), ("clf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=rs))]),
        "XGBoost": lambda: Pipeline([("pre", pre(False)), ("clf", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=rs))]),
    }
    pipe = fab[spec["modele"]]()
    pipe.fit(X, y)
    return {"pipeline": pipe, "colonnes": spec["colonnes"], "modele": spec["modele"],
            "origine": "reentraine"}


D = charger()

# ================================================================== navigation
PAGES_PRINCIPALES = ["Importer les données", "Prévision des offres", "Scorer un abonné",
                     "Situation", "Ciblage"]
PAGES_AVANCEES = ["Modèles de churn"]
LIBELLE_PAGE = {
    "Importer les données": "Importer mes données",
    "Prévision des offres": "Prévision des offres",
    "Scorer un abonné": "Évaluer un client",
    "Situation": "Vue d'ensemble",
    "Ciblage": "Clients à risque",
    "Modèles de churn": "Comparaison des modèles de départ",
}
ss_nav = st.session_state
ss_nav.setdefault("nav_principal", PAGES_PRINCIPALES[0])
ss_nav.setdefault("nav_avance", None)


def _nav_principal():
    ss_nav["nav_avance"] = None


def _nav_avance():
    ss_nav["nav_principal"] = None


with st.sidebar:
    st.markdown(
        "<div class='marque'><div class='nom'>CAMTEL</div>"
        "<div class='sous'>Tableau de bord des offres et des départs d'abonnés</div></div>",
        unsafe_allow_html=True)
    st.radio("Tableau de bord", PAGES_PRINCIPALES, index=None, key="nav_principal",
             on_change=_nav_principal, format_func=LIBELLE_PAGE.get,
             label_visibility="collapsed")
    with st.expander("Analyse détaillée",
                     expanded=ss_nav.get("nav_avance") is not None):
        st.radio("Analyse détaillée", PAGES_AVANCEES, index=None, key="nav_avance",
                 on_change=_nav_avance, format_func=LIBELLE_PAGE.get,
                 label_visibility="collapsed")
    page = ss_nav.get("nav_principal") or ss_nav.get("nav_avance") or PAGES_PRINCIPALES[0]
    st.markdown(
        "<div class='pied'>Extrait HSS de janvier à juillet 2025, 13 937 abonnés.<br>"
        "Activations d'avril 2022 à août 2025, 41 relevés mensuels.</div>",
        unsafe_allow_html=True)

flux = D["flux"]
scores = st.session_state.get("scores_import", D["scores"])


def avertir_reference(pages_concernees):
    """Rappelle qu'une page reste calculée sur l'analyse de référence après un import."""
    if "series_import" in st.session_state or "scores_import" in st.session_state:
        T.note("Vous avez importé vos propres données, mais " + pages_concernees +
               " Cette page continue d'afficher l'analyse de référence du projet.")


@st.cache_data(show_spinner=False)
def lire_offres_cache(nom, data):
    return PL.lire_offres(nom, data)


@st.cache_data(show_spinner=False)
def charger_offres_long():
    """Toutes les offres du classeur (table date, offre, activations), ou None."""
    try:
        p = OUT / "offres_long.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["date"])
            df = df.groupby(["date", "offre"], as_index=False)["activations"].sum()
        else:
            xl = ROOT / "data" / "offres.xlsx"
            if not xl.exists():
                return None
            df, _ = PL.lire_offres(xl.name, xl.read_bytes())
        df, _ = PL.unifier_libelles(df)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def charger_departs():
    """
    Un départ par ligne (date, région, forfait), calculé automatiquement depuis
    churn_database2.csv (nettoyage inclus, voir `pipeline.departs_churn`).
    Retourne (table, message d'erreur) : l'un des deux est None.
    """
    dossiers = [ROOT / "data", OUT, ROOT, ROOT.parent / "data", Path.cwd(), Path.cwd() / "data"]
    for d in dossiers:
        brut = d / "churn_database2.csv"
        if brut.exists():
            try:
                dep = PL.departs_churn(pd.read_csv(brut))
                return dep.dropna(subset=["date_depart"]), None
            except Exception as e:
                return None, f"`{brut}` n'a pas pu être traité : {e}"
    p = OUT / "departs_abonnes.csv"          # à défaut, table déjà calculée
    if p.exists():
        try:
            return pd.read_csv(p, parse_dates=["date_depart"]).dropna(subset=["date_depart"]), None
        except Exception as e:
            return None, f"`{p}` n'a pas pu être lu : {e}"
    return None, ("Le fichier `churn_database2.csv` est introuvable. Placez-le dans le "
                  "dossier `data` à côté de `app.py` : il sera lu et nettoyé automatiquement.")


@st.cache_data(show_spinner=False)
def series_offres_cache(long, offres, debut, fin):
    return PL.series_propres(long, list(offres), debut, fin)


def afficher_resultat(res, serie, offre):
    """Cartes, graphiques et tableaux d'un résultat de PV.lancer()."""
    if res["erreurs"]:
        T.note("Certains modèles n'ont pas pu tourner : " + " ; ".join(
            f"<b>{n}</b> ({m})" for n, m in res["erreurs"].items())
               + ". Pour le LSTM, vérifiez que <code>tensorflow</code> est installé ; "
                 "pour Prophet, <code>prophet</code>.", alerte=True)
    r = res["retenu"]
    if r is None:
        return
    ref, best = res["metriques"][PV.REFERENCE], res["metriques"][r]
    futur = res["futur"][r]
    gain = 1 - best["MAPE (%)"] / ref["MAPE (%)"]

    k = st.columns(4)
    with k[0]:
        T.tuile("Modèle retenu", r, "plus faible erreur relative sur le test", "vert")
    with k[1]:
        T.tuile("Erreur moyenne", T.dec(best["MAPE (%)"], 2, "%"),
                f"MAPE sur {res['h_test']} jours")
    with k[2]:
        T.tuile("Écart avec la référence", ("\u2212" if gain >= 0 else "+") + T.pct(abs(gain), 0),
                "d'erreur relative, référence : " + T.dec(ref["MAPE (%)"], 1, "%"),
                "ambre" if gain > 0 else "brique")
    with k[3]:
        T.tuile(f"Prévu sur {res['h_prev']} jours", T.nb(futur.sum()),
                T.fcfa(futur.sum() * PRIX[offre]) if offre in PRIX else "activations")

    couleur = {"Réel": T.ENCRE, PV.REFERENCE: T.TRAIT}
    for i, n in enumerate(PV.MODELES_DISPONIBLES):
        couleur[n] = T.SEQUENCE[(i + 1) % len(T.SEQUENCE)]

    T.section("Test hors échantillon", f"{res['h_test']} derniers jours, cachés aux modèles")
    t = res["test"]
    fig = go.Figure()
    for n in t.columns:
        if n == "Réel":
            continue
        fig.add_trace(go.Scatter(
            x=t.index, y=t[n], name=n,
            line=dict(color=couleur[n], width=1.4,
                      dash="dot" if n == PV.REFERENCE else None)))
    fig.add_trace(go.Scatter(x=t.index, y=t["Réel"], name="réalisé",
                             line=dict(color=T.ENCRE, width=2.2)))
    fig.update_yaxes(title="activations / jour")
    fig.update_layout(legend=dict(font=dict(size=11)))
    T.axe_mois(fig, t.index)
    T.afficher(fig, 340)

    T.section("Comparaison des modèles")
    m = pd.DataFrame(res["metriques"]).T
    m.index.name = "Modèle"
    m = m.reset_index()
    m["Retenu"] = np.where(m["Modèle"] == r, "oui", "")
    st.dataframe(m.sort_values("MAPE (%)"), width="stretch", hide_index=True)
    st.caption("MAE, RMSE et biais sont en activations par jour. Un biais négatif "
               "signifie que le modèle sous-estime. Un écart d'un ou deux points de "
               "MAPE entre deux modèles n'est pas décisif : le classement vaut pour "
               "ces jours précis.")

    T.section(f"Prévision à {res['h_prev']} jours", f"{r}, intervalle empirique à 90 %")
    hist = serie.iloc[-180:]
    band = PV.tableau_futur(res, offre).set_index("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=band.index, y=band["borne_haute"], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band.index, y=band["borne_basse"], fill="tonexty",
                             fillcolor="rgba(223,162,60,.16)", line=dict(width=0),
                             name="intervalle 90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="observé",
                             line=dict(color=T.BLEU, width=1.5)))
    for n, f_ in res["futur"].items():
        if n == r:
            continue
        fig.add_trace(go.Scatter(x=f_.index, y=f_.values, name=n, visible="legendonly",
                                 line=dict(color=couleur[n], width=1.2, dash="dot")))
    fig.add_trace(go.Scatter(x=band.index, y=band["prevision"], name=f"prévu ({r})",
                             line=dict(color=T.AMBRE, width=2)))
    fig.add_vline(x=hist.index[-1], line=dict(color=T.TRAIT, width=1, dash="dot"))
    fig.update_yaxes(title="activations / jour")
    T.axe_mois(fig, list(hist.index) + list(band.index))
    T.afficher(fig, 340)
    st.caption("Les autres modèles se réaffichent d'un clic dans la légende.")

    mens = futur.groupby(futur.index.to_period("M")).agg(["sum", "count"])
    mens.columns = ["Activations prévues", "Jours couverts"]
    mens.insert(0, "Mois", [T.etiquette_mois(str(p)) for p in mens.index])
    st.dataframe(mens.reset_index(drop=True), width="stretch", hide_index=True)
    st.caption("Un mois qui compte moins de jours que le calendrier est un mois "
               "partiel : ne le comparez pas tel quel aux mois complets.")
    st.download_button("Exporter la prévision (CSV)",
                       PV.tableau_futur(res, offre).to_csv(index=False).encode("utf-8"),
                       f"prevision_{re.sub(r'[^A-Za-z0-9]+', '_', offre).strip('_')}_{res['h_prev']}j.csv",
                       "text/csv")


# ============================================================ IMPORT DES DONNÉES
if page == "Importer les données":
    T.entete("Importer mes données",
             "Déposez vos fichiers depuis votre ordinateur. L'application les nettoie avec "
             "les mêmes règles que l'analyse de référence, puis alimente les pages qui en "
             "dépendent.")
    ss = st.session_state

    c = st.columns(2)
    with c[0]:
        if "series_import" in ss:
            s = ss["series_import"]
            T.tuile("Offres en cours d'usage", f"{s.shape[1]} offre(s)",
                    f"{len(s)} jours, du {s.index.min():%d/%m/%Y} au {s.index.max():%d/%m/%Y}",
                    "vert")
        else:
            T.tuile("Offres en cours d'usage", "Données de référence",
                    "classeur du projet, avril 2022 à août 2025")
    with c[1]:
        if "scores_import" in ss:
            T.tuile("Abonnés en cours d'usage", T.nb(len(ss["scores_import"])),
                    f"vos données, AUC de contrôle {ss['churn_auc']:.3f}", "vert")
        else:
            T.tuile("Abonnés en cours d'usage", T.nb(len(D["scores"])),
                    "extrait HSS de référence, janvier à juillet 2025")
    st.write("")

    onglets = st.tabs(["Offres (activations)", "Abonnés (churn)"])

    # ---------------------------------------------------------------- offres
    with onglets[0]:
        st.markdown(
            "Le classeur mensuel d'origine (une feuille par mois) est accepté tel quel. "
            "Un CSV quotidien l'est aussi : soit trois colonnes "
            "<code>date, offre, activations</code>, soit une colonne de dates suivie d'une "
            "colonne par offre.", unsafe_allow_html=True)
        f = st.file_uploader("Classeur des offres (.xlsx) ou série quotidienne (.csv)",
                             type=["xlsx", "xlsm", "csv"], key="up_offres")
        if f is not None:
            long = None
            try:
                with st.spinner("Lecture du fichier, cela peut prendre une minute pour un classeur complet."):
                    long, jr = lire_offres_cache(f.name, f.getvalue())
            except Exception as e:
                st.error(f"Ce fichier n'a pas pu être lu : {e}")
            if long is not None:
                st.success(jr[-1])
                volume = long.groupby("offre")["activations"].sum().sort_values(ascending=False)
                defaut = [o for o in PL.FORFAITS_DEFAUT if o in volume.index] \
                    or list(volume.index[:3])
                dmin, dmax = long["date"].min().date(), long["date"].max().date()
                deb0 = pd.Timestamp("2023-10-01").date()
                deb0 = deb0 if dmin <= deb0 <= dmax else dmin
                cc = st.columns([3, 1.4, 1.4])
                toutes = st.checkbox("Conserver toutes les offres du fichier",
                                     help="Sinon, choisissez les offres à conserver ci-dessous.")
                offres = cc[0].multiselect("Offres à conserver", list(volume.index),
                                           default=list(volume.index) if toutes else defaut)
                debut = cc[1].date_input("Début de la période", deb0,
                                         min_value=dmin, max_value=dmax)
                fin = cc[2].date_input("Fin de la période", dmax,
                                       min_value=dmin, max_value=dmax)
                st.caption("Écartez la phase de lancement d'une offre, qui fausse la "
                           "prévision, et les derniers jours si le mois n'est pas complet. "
                           "La page « Prévision des offres » propose de toute façon toutes les "
                           "offres du fichier : ce choix-ci concerne les autres pages.")
                if st.button("Utiliser ces données", type="primary", disabled=not offres):
                    tab, jr2 = PL.series_propres(long, offres, debut, fin)
                    ss["series_import"] = tab
                    ss["long_import"] = long
                    ss["import_journal"] = jr2
                    for k_ in ("prev_live", "prev_multi", "sel_offres"):
                        ss.pop(k_, None)
                    st.rerun()
        if "import_journal" in ss:
            with st.expander("Ce qui a été fait aux séries"):
                st.markdown("\n".join(f"- {x}" for x in ss["import_journal"]))
            if len(ss["series_import"]) < 200:
                st.warning("Moins de 200 jours : la prévision sera peu fiable, surtout "
                           "pour le LSTM.")

    # ----------------------------------------------------------------- churn
    with onglets[1]:
        st.markdown(
            "Le fichier doit avoir la structure de <code>churn_database2.csv</code> "
            "(identifiant, dates de création et de purge, forfait, recharges, région...).",
            unsafe_allow_html=True)
        f = st.file_uploader("Fichier des abonnés (.csv)", type=["csv"], key="up_churn")
        if f is not None:
            try:
                brut = pd.read_csv(io.BytesIO(f.getvalue()))
            except Exception as e:
                brut = None
                st.error(f"Ce fichier n'a pas pu être lu : {e}")
            if brut is not None:
                manque = PL.verifier_churn(brut)
                if manque:
                    st.error("Colonnes absentes : " + ", ".join(manque))
                else:
                    st.success(f"{len(brut):,} lignes, {brut.shape[1]} colonnes".replace(",", "\u202f"))
                    if st.button("Préparer et entraîner le modèle", type="primary"):
                        try:
                            with st.spinner("Nettoyage puis entraînement d'un XGBoost..."):
                                propre, jr = PL.preparer_churn(brut)
                                sc, auc = PL.entrainer_churn(propre)
                            ss["scores_import"], ss["churn_auc"] = sc, auc
                            ss["churn_journal"] = jr
                            st.rerun()
                        except Exception as e:
                            st.error(f"Le traitement a échoué : {e}")
        if "churn_journal" in ss:
            with st.expander("Ce qui a été fait au fichier"):
                st.markdown("\n".join(f"- {x}" for x in ss["churn_journal"]))
            T.note("Les scores individuels alimentent les pages <b>Situation</b> (risque par "
                   "décile) et <b>Ciblage</b>. Les départs par période et par région de la "
                   "page <b>Évaluer un client</b> se calculent tout seuls depuis "
                   "<code>churn_database2.csv</code>. Le scoreur d'un abonné et la page des "
                   "modèles gardent le modèle de référence.")

    if "series_import" in ss or "scores_import" in ss:
        st.write("")
        if st.button("Revenir aux données de référence"):
            for k in ["series_import", "long_import", "import_journal", "scores_import",
                      "churn_auc", "churn_journal", "prev_live", "prev_multi",
                      "sel_offres"]:
                ss.pop(k, None)
            st.rerun()


# ================================================================== SITUATION
elif page == "Situation":
    T.entete("Vue d'ensemble de la base Blue One",
             "Ce que le modèle attend pour le mois à venir : combien d'abonnés entrent, "
             "combien partent, et ce que représentent ces départs en revenu mensuel.")

    m = flux[flux["mois"] == flux["mois"].iloc[0]]
    debut, entrees = m["base_debut"].sum(), m["entrees"].sum()
    sorties, fin = m["departs_attendus"].sum(), m["base_fin"].sum()
    var = (fin - debut) / debut

    T.grand_livre([
        ("Base active au 1er du mois", T.nb(debut), "abonnés Blue One S, M et L", "", ""),
        ("Nouvelles souscriptions", T.nb(entrees), "part des activations prévues",
         "gl-entree", "+"),
        ("Départs attendus", T.nb(sorties), T.fcfa(m["revenu_a_risque_fcfa"].sum())
         + " de revenu mensuel", "gl-sortie", "−"),
        ("Base en fin de mois", T.nb(fin),
         ("+" if var >= 0 else "\u2212") + T.pct(abs(var), 2) + " sur le mois",
         "gl-resultat", "="),
    ])

    T.section("Activations quotidiennes", "observé puis prévu")
    offre = st.radio("Forfait", list(D["series"].columns), index=1,
                     horizontal=True, label_visibility="collapsed")
    hist = D["series"][offre].iloc[-200:]
    p = D["prev"][D["prev"]["offre"] == offre].set_index("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p["borne_haute"], line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=p.index, y=p["borne_basse"], fill="tonexty",
                             fillcolor="rgba(223,162,60,.16)", line=dict(width=0),
                             name="intervalle 90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="observé",
                             line=dict(color=T.BLEU, width=1.4)))
    fig.add_trace(go.Scatter(x=p.index, y=p["prevision"], name="prévu",
                             line=dict(color=T.AMBRE, width=1.8)))
    fig.add_vline(x=hist.index[-1], line=dict(color=T.TRAIT, width=1, dash="dot"))
    fig.update_yaxes(title="activations / jour")
    T.axe_mois(fig, list(hist.index) + list(p.index))
    T.afficher(fig, 330)

    T.section("Où se concentre le risque")
    c = st.columns([3, 2])

    with c[0]:
        dec = scores.groupby("decile_risque").agg(
            taux=("churn_reel", "mean"), n=("CUSTOMER_ID", "count"),
            perte=("perte_attendue_fcfa", "sum")).reset_index()
        fig = go.Figure(go.Bar(
            x=dec["decile_risque"], y=dec["taux"],
            marker=dict(color=[T.BRIQUE if d >= 8 else T.BLEU
                               for d in dec["decile_risque"]], line=dict(width=0)),
            customdata=np.stack([dec["n"], dec["perte"] / 1e6], -1),
            hovertemplate="Décile %{x}<br>%{y:.1%} de churn observé<br>"
                          "%{customdata[0]:,.0f} abonnés<br>"
                          "%{customdata[1]:.1f} M FCFA exposés<extra></extra>"))
        fig.update_yaxes(tickformat=".0%", title="part réellement partie")
        fig.update_xaxes(title="décile de score (10 = risque le plus élevé)", dtick=1)
        fig.update_layout(hovermode="closest")
        T.afficher(fig, 320)

    with c[1]:
        piv = flux.pivot(index="mois", columns="forfait", values="revenu_a_risque_fcfa")
        etiq = [T.etiquette_mois(m) for m in piv.index]
        fig = go.Figure()
        for col in piv.columns:
            fig.add_trace(go.Bar(x=etiq, y=piv[col] / 1e6, name=col,
                                 marker_color=T.COULEUR_FORFAIT[col],
                                 hovertemplate="%{y:.1f} M FCFA<extra>" + col + "</extra>"))
        fig.update_layout(barmode="stack", bargap=.55)
        fig.update_yaxes(title="M FCFA à risque")
        fig.update_xaxes(type="category")
        T.afficher(fig, 320)

    T.note(
        "Le score sert à classer les abonnés du plus au moins exposé, pas à mesurer le "
        "niveau réel du churn. Notre extrait compte 44 % de partants, un chiffre qu'une "
        "base stable ne peut pas tenir. Les montants affichés viennent donc du scénario "
        "central du modèle de flux, pas du taux brut de l'extrait.")

    # ---- Qualité des données (intégrée à la vue d'ensemble)
    a = D["audit"]
    T.section("Qualité des données",
              "ce que contiennent les fichiers sources, et les corrections appliquées")

    c = st.columns(4)
    with c[0]:
        T.tuile("Abonnés", T.nb(a["n_lignes"]), "aucune valeur manquante")
    with c[1]:
        T.tuile("Retenus après nettoyage", T.nb(12626), "90,6 % de l'extrait", "vert")
    with c[2]:
        T.tuile("Churn dans l'extrait", f"{a['taux_churn_global']:.1%}",
                "non représentatif de la base", "ambre")
    with c[3]:
        T.tuile("Colonnes écartées", "10", "constantes, redondantes ou fuitantes", "brique")

    T.section("Deux pièges à éviter avant de modéliser")
    c = st.columns(2)
    with c[0]:
        T.note(
            f"<b>Une fuite d'information.</b> La règle « dernière purge datant de plus "
            f"de 60 jours au {a['date_ref'][:10]} » retrouve l'étiquette dans "
            f"{a['regle_60j_concordance']:.2%} des cas, pour "
            f"{a['regle_60j_exceptions']} exceptions. Si on garde les variables tirées "
            f"de la purge, l'AUC grimpe à 0,996 sans que le modèle ait rien appris : il "
            f"redécouvre simplement la définition de la cible.", alerte=True)
    with c[1]:
        T.note(
            f"<b>Une censure à droite.</b> {a['n_censures']} abonnés créés après le "
            f"{a['seuil_censure'][:10]} n'avaient pas eu le temps de cumuler 60 jours "
            f"d'inactivité avant la fin de la fenêtre, et on n'y trouve que "
            f"{a['churners_parmi_censures']} partant. Leur étiquette reflète la durée "
            f"d'observation, pas leur fidélité. On les retire de la classification et "
            f"on les garde pour l'analyse de survie.")

    T.section("Écart de composition entre les deux sources")
    mix_e = scores["PLAN_NAME"].value_counts(normalize=True)
    mens = D["series"].loc["2025-01-01":].sum()
    mix_r = mens / mens.sum()
    forfaits = ["Blue One S", "Blue One M", "Blue One L"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=forfaits, x=[mix_e.get(f, 0) for f in forfaits],
                         orientation="h", name="extrait churn", marker_color=T.CIEL,
                         hovertemplate="%{x:.1%} de l'extrait<extra></extra>"))
    fig.add_trace(go.Bar(y=forfaits, x=[mix_r.get(f, 0) for f in forfaits],
                         orientation="h", name="activations réelles 2025",
                         marker_color=T.BLEU,
                         hovertemplate="%{x:.1%} des activations<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="part du forfait")
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(showgrid=False)
    T.afficher(fig, 250)
    st.caption("Blue One M pèse 68 % de l'extrait mais 93 % des activations réelles. "
               "Tout calcul de revenu doit être repondéré.")

    T.section("Séries d'activations", "aberrants détectés par filtre de Hampel")
    offre = st.radio("Forfait", ["Blue One M", "Blue One L", "Blue One S"],
                     horizontal=True, label_visibility="collapsed", key="forfait_qualite")
    d = D["diag"][D["diag"]["offre"] == offre]
    ab = d[d["aberrant"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["brut"], name="relevé brut",
                             line=dict(color=T.TRAIT, width=1)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["corrige"], name="série corrigée",
                             line=dict(color=T.BLEU, width=1.2)))
    fig.add_trace(go.Scatter(x=ab["date"], y=ab["brut"], mode="markers",
                             name=f"{len(ab)} points écartés",
                             marker=dict(color=T.BRIQUE, size=6, line=dict(width=0))))
    fig.add_vrect(x0=d["date"].min(), x1=pd.Timestamp("2023-10-01"),
                  fillcolor="#000", opacity=.04, line_width=0,
                  annotation_text="montée en charge, exclue de la modélisation",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color=T.GRIS))
    fig.update_yaxes(title="activations / jour")
    T.axe_mois(fig, d["date"], pas=3)
    T.afficher(fig, 330)


# ============================================================= MODÈLES CHURN
elif page == "Modèles de churn":
    T.entete("Comparaison des modèles de départ",
             "Trois classifieurs entraînés sur les mêmes variables, évalués sur 25 % des "
             "observations mises de côté. Validation croisée à cinq blocs sur le reste.")

    avertir_reference("les modèles de churn ont été entraînés sur l'extrait de référence.")
    t = D["t_churn"]
    onglets = st.tabs(["Performance", "Discrimination", "Ce qui compte", "Survie"])

    with onglets[0]:
        c = st.columns(3)
        for i, r in t.iterrows():
            with c[i]:
                T.tuile(r["Modele"], T.dec(r["AUC (test)"], 3),
                        f"rappel {T.dec(r['Rappel (churn)'], 2)}, précision "
                        f"{T.dec(r['Precision (churn)'], 2)}",
                        "vert" if i == 0 else "",
                        jauge=(r["AUC (test)"] - 0.5) / 0.5 * 100)

        T.section("Métriques détaillées", "seuil de décision 0,5")
        vue = st.radio("Vue", ["Pouvoir discriminant", "Classification"],
                       horizontal=True, label_visibility="collapsed")
        if vue == "Pouvoir discriminant":
            st.dataframe(t[["Modele", "AUC (VC 5 blocs)", "AUC (test)",
                            "AUC-PR (test)", "Brier"]], width="stretch",
                         hide_index=True)
            st.caption("Le score de Brier mesure l'écart entre le risque annoncé et ce "
                       "qui s'est vraiment passé. Plus il est bas, plus le score "
                       "s'utilise tel quel, sans correction.")
            st.latex(r"\mathrm{Brier} = \frac{1}{n}\sum_{i=1}^{n}\bigl(\hat p_i - y_i\bigr)^2")
        else:
            st.dataframe(t[["Modele", "Exactitude", "Precision (churn)",
                            "Rappel (churn)", "F1 (churn)", "Precision (fidele)",
                            "Rappel (fidele)", "VN / FP / FN / VP"]],
                         width="stretch", hide_index=True)
            st.caption("VN / FP / FN / VP : vrais négatifs, faux positifs, faux "
                       "négatifs, vrais positifs sur les 3 157 abonnés du jeu de test.")
            st.latex(r"\text{précision} = \frac{VP}{VP + FP}, \quad "
                     r"\text{rappel} = \frac{VP}{VP + FN}, \quad "
                     r"F_1 = \frac{2\cdot\text{précision}\cdot\text{rappel}}"
                     r"{\text{précision} + \text{rappel}}")

        T.section("Compromis précision / rappel")
        c = st.columns([2, 3])
        with c[0]:
            seuil = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.01)
            modele = st.selectbox("Modèle", t["Modele"].tolist())
            dd = D["distrib"]
            dd = dd[dd["modele"] == modele]
            yp = (dd["proba"] >= seuil).astype(int)
            vp = int(((yp == 1) & (dd["reel"] == 1)).sum())
            fp = int(((yp == 1) & (dd["reel"] == 0)).sum())
            fn = int(((yp == 0) & (dd["reel"] == 1)).sum())
            vn = int(((yp == 0) & (dd["reel"] == 0)).sum())
            prec = vp / max(vp + fp, 1)
            rapp = vp / max(vp + fn, 1)
            f1 = 2 * prec * rapp / max(prec + rapp, 1e-9)
            k = st.columns(3)
            with k[0]:
                T.tuile("Précision", T.dec(prec, 3), f"{fp} fausses alertes", "ambre")
            with k[1]:
                T.tuile("Rappel", T.dec(rapp, 3), f"{fn} départs manqués", "brique")
            with k[2]:
                T.tuile("F1", T.dec(f1, 3), f"{vp} départs détectés", "vert")
        with c[1]:
            fig = go.Figure()
            for lab, val, coul in [("abonnés fidèles", 0, T.VERT),
                                   ("abonnés partis", 1, T.BRIQUE)]:
                fig.add_trace(go.Histogram(
                    x=dd[dd["reel"] == val]["proba"], name=lab, nbinsx=44,
                    marker=dict(color=coul, line=dict(width=0)), opacity=.72))
            fig.add_vline(x=seuil, line=dict(color=T.ENCRE, width=1.5, dash="dash"))
            fig.update_layout(barmode="overlay", hovermode="closest")
            fig.update_xaxes(title="score de risque attribué")
            fig.update_yaxes(title="abonnés")
            T.afficher(fig, 330)

    with onglets[1]:
        cb = D["courbes"]
        c = st.columns(3)
        for col, (code, titre, xl, yl) in zip(c, [
                ("roc", "Courbe ROC", "faux positifs", "vrais positifs"),
                ("pr", "Précision-rappel", "rappel", "précision"),
                ("calibration", "Calibration", "score prédit", "fréquence observée")]):
            fig = go.Figure()
            if code == "roc":
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(
                    color=T.TRAIT, width=1, dash="dash"), showlegend=False,
                    hoverinfo="skip"))
            if code == "calibration":
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(
                    color=T.TRAIT, width=1, dash="dash"), showlegend=False,
                    hoverinfo="skip"))
            for i, m in enumerate(cb["modele"].unique()):
                s = cb[(cb["modele"] == m) & (cb["courbe"] == code)]
                fig.add_trace(go.Scatter(
                    x=s["x"], y=s["y"], name=m, mode="lines+markers" if
                    code == "calibration" else "lines",
                    marker=dict(size=5), line=dict(color=T.SEQUENCE[i], width=1.8)))
            fig.update_xaxes(title=xl)
            fig.update_yaxes(title=yl)
            fig.update_layout(hovermode="closest",
                              legend=dict(font=dict(size=10.5)),
                              title=dict(text=titre))
            with col:
                T.afficher(fig, 340)
        st.caption("La calibration compte autant que la discrimination : un score qu'on "
                   "multiplie par un prix pour obtenir un revenu à risque doit être "
                   "calibré, pas seulement bien ordonné.")

    with onglets[2]:
        c = st.columns([3, 2])
        with c[0]:
            imp = D["imp"].head(10).sort_values("importance")
            fig = go.Figure(go.Bar(
                x=imp["importance"], y=imp["variable"], orientation="h",
                error_x=dict(array=imp["ecart_type"], color=T.GRIS, thickness=1, width=3),
                marker=dict(color=T.BLEU, line=dict(width=0)),
                hovertemplate="%{y}<br>−%{x:.4f} d'AUC<extra></extra>"))
            fig.update_xaxes(title="chute d'AUC après permutation")
            fig.update_yaxes(showgrid=False)
            fig.update_layout(hovermode="closest")
            T.afficher(fig, 360)
        with c[1]:
            abl = pd.DataFrame(list(D["rc"]["_ablation_auc"].items()),
                               columns=["Variante", "AUC"])
            fig = go.Figure(go.Bar(
                x=abl["AUC"], y=abl["Variante"], orientation="h",
                marker=dict(color=[T.BRIQUE if v < 0.7 else T.CIEL for v in abl["AUC"]],
                            line=dict(width=0)),
                text=[f"{v:.3f}" for v in abl["AUC"]], textposition="inside",
                insidetextfont=dict(color="#fff"),
                hovertemplate="%{y} : AUC %{x:.3f}<extra></extra>"))
            fig.add_vline(x=D["t_churn"]["AUC (test)"].max(),
                          line=dict(color=T.ENCRE, width=1, dash="dot"))
            fig.update_xaxes(title="AUC", range=[0.5, 0.95])
            fig.update_yaxes(showgrid=False)
            fig.update_layout(hovermode="closest")
            T.afficher(fig, 360)
        T.note(
            "Sans la fréquence de recharge, l'AUC tombe à 0,57. Ce qui limite la "
            "prédiction, c'est l'information dont on dispose, pas l'algorithme : un modèle "
            "plus sophistiqué n'y changerait presque rien, alors que des données d'usage "
            "(volume consommé, appels au service client, réclamations) feraient une vraie "
            "différence.")

    with onglets[3]:
        c = st.columns([3, 2])
        with c[0]:
            km = D["km"]
            fig = go.Figure()
            for f in ["Blue One S", "Blue One M", "Blue One L"]:
                s = km[km["forfait"] == f]
                if s.empty:
                    continue
                coul = T.COULEUR_FORFAIT[f]
                rgb = tuple(int(coul.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
                fig.add_trace(go.Scatter(x=s["t"], y=s["haut"], line=dict(width=0),
                                         showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=s["t"], y=s["bas"], fill="tonexty",
                                         fillcolor=f"rgba{rgb + (0.13,)}",
                                         line=dict(width=0), showlegend=False,
                                         hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=s["t"], y=s["survie"], name=f,
                                         line=dict(color=coul, width=1.8, shape="hv")))
            fig.add_hline(y=0.5, line=dict(color=T.TRAIT, width=1, dash="dot"))
            fig.update_xaxes(title="jours depuis la souscription")
            fig.update_yaxes(title="part d'abonnés encore actifs", tickformat=".0%")
            T.afficher(fig, 360)
        with c[1]:
            med = D["rc"]["_survie"]["mediane_survie_jours"]
            for f, v in med.items():
                lisible = ("non atteinte sur la fenêtre"
                           if v is None or v != v or v == float("inf") else f"{v:.0f} jours")
                T.tuile(f, lisible, "durée médiane avant départ",
                        "brique" if isinstance(v, float) and v == v and v < 150 else "")
                st.write("")
            st.caption(f"Concordance du modèle de Cox : "
                       f"{D['rc']['_survie']['concordance_cox']}")
        T.section("Rapports de risque", "référence : Blue One L")
        cox = D["cox"].copy()
        cox["signif"] = np.where(cox["p"] < 0.05, "oui", "non")
        fig = go.Figure(go.Bar(
            x=cox["exp(coef)"], y=cox["covariate"], orientation="h",
            marker=dict(color=[T.BLEU if s == "oui" else T.TRAIT for s in cox["signif"]],
                        line=dict(width=0)),
            text=[f"×{v:.2f}" for v in cox["exp(coef)"]], textposition="outside",
            hovertemplate="%{y}<br>risque ×%{x:.3f}<extra></extra>"))
        fig.add_vline(x=1, line=dict(color=T.ENCRE, width=1))
        fig.update_xaxes(title="rapport de risque")
        fig.update_yaxes(showgrid=False)
        fig.update_layout(hovermode="closest")
        T.afficher(fig, 260)
        st.caption("En gris, les coefficients non significatifs au seuil de 5 %.")
        with st.expander("Les formules derrière ces courbes"):
            st.markdown("La courbe de survie de Kaplan-Meier estime la part d'abonnés "
                        "encore actifs à l'instant t. d<sub>i</sub> départs sur n<sub>i</sub> "
                        "abonnés à risque à chaque date t<sub>i</sub> :",
                        unsafe_allow_html=True)
            st.latex(r"\hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right)")
            st.markdown("Le modèle de Cox relie le risque instantané de départ aux "
                        "caractéristiques x de l'abonné. Le rapport de risque d'une "
                        "variable est l'exponentielle de son coefficient :")
            st.latex(r"h(t \mid x) = h_0(t)\,\exp\bigl(\beta^{\top}x\bigr), \qquad "
                     r"\mathrm{RR}_j = e^{\beta_j}")


# ===================================================================== CIBLAGE
elif page == "Ciblage":
    T.entete("Clients à risque : qui appeler cette semaine",
             "Le modèle classe les abonnés par risque. Fixez la capacité de la campagne, "
             "la liste s'ajuste et s'exporte.")

    c = st.columns([1.6, 2.2, 2.2, 1.6])
    budget = c[0].slider("Part de la base contactée", 1, 50, 10, format="%d %%")
    forfaits = c[1].multiselect("Forfaits retenus", sorted(scores["PLAN_NAME"].unique()),
                                default=sorted(scores["PLAN_NAME"].unique()),
                                placeholder="Tous les forfaits")
    regions = c[2].multiselect("Régions", sorted(scores["REGION"].unique()),
                               placeholder="Toutes les régions")
    cout = c[3].number_input("Coût d'un contact (FCFA)", 0, 5000, 150, 50)

    f = scores[scores["PLAN_NAME"].isin(forfaits)]
    if regions:
        f = f[f["REGION"].isin(regions)]
    if f.empty:
        st.warning("Aucun abonné ne correspond à ces filtres. Élargissez la sélection.")
        st.stop()

    f = f.sort_values("proba_churn", ascending=False).reset_index(drop=True)
    n = max(1, int(len(f) * budget / 100))
    cible = f.head(n)
    captes = int(cible["churn_reel"].sum())
    total = int(f["churn_reel"].sum())
    expose = float(cible["perte_attendue_fcfa"].sum())

    T.grand_livre([
        ("Abonnés contactés", T.nb(n), f"{budget} % de {T.nb(len(f))}", "", ""),
        ("Départs interceptés", T.nb(captes),
         T.pct(captes / max(total, 1), 0) + " des départs de la sélection",
         "gl-entree", "\u2192"),
        ("Revenu mensuel exposé", T.fcfa(expose), "somme des pertes attendues", "gl-sortie", ""),
        ("Coût de la campagne", T.fcfa(n * cout),
         f"soit {expose/max(n*cout,1):.0f}\u202f× le coût si tout est retenu",
         "gl-resultat", "="),
    ])

    T.section("Rendement du ciblage")
    c = st.columns([3, 2])
    with c[0]:
        g = f.copy()
        g["rang"] = (g.index + 1) / len(g) * 100
        g["capture"] = g["churn_reel"].cumsum() / max(total, 1) * 100
        pas = max(1, len(g) // 600)
        gg = g.iloc[::pas]
        taux = g["churn_reel"].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="tirage aléatoire",
                                 line=dict(color=T.TRAIT, width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=[0, 100 * taux, 100], y=[0, 100, 100],
                                 name="ciblage parfait",
                                 line=dict(color=T.VERT, width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=gg["rang"], y=gg["capture"], name="modèle",
                                 line=dict(color=T.BLEU, width=2.2)))
        fig.add_trace(go.Scatter(x=[budget], y=[g.loc[g["rang"] <= budget, "capture"].max()],
                                 mode="markers", name="réglage actuel",
                                 marker=dict(color=T.AMBRE, size=11,
                                             line=dict(color="#fff", width=2))))
        fig.update_xaxes(title="% de la base contactée")
        fig.update_yaxes(title="% des départs interceptés")
        fig.update_layout(hovermode="closest")
        T.afficher(fig, 340)
    with c[1]:
        rep = cible.groupby("PLAN_NAME")["perte_attendue_fcfa"].sum().reset_index()
        fig = go.Figure(go.Pie(
            labels=rep["PLAN_NAME"], values=rep["perte_attendue_fcfa"], hole=.58,
            marker=dict(colors=[T.COULEUR_FORFAIT[p] for p in rep["PLAN_NAME"]],
                        line=dict(color="#fff", width=2)),
            texttemplate="%{percent:.0%}", textfont=dict(color="#fff", size=13),
            hovertemplate="%{label}<br>%{value:,.0f} FCFA<extra></extra>"))
        fig.update_layout(hovermode="closest",
                          annotations=[dict(text="revenu<br>exposé", showarrow=False,
                                            font=dict(size=13, color=T.GRIS))])
        T.afficher(fig, 340)

    T.section("Liste d'appel", f"{T.nb(n)} abonnés, triés par risque décroissant")
    liste = cible[["CUSTOMER_ID", "PLAN_NAME", "REGION", "proba_churn",
                   "perte_attendue_fcfa"]].head(400).copy()
    liste["perte_attendue_fcfa"] = liste["perte_attendue_fcfa"].map(T.fcfa)
    st.dataframe(
        liste, width="stretch", hide_index=True,
        column_config={
            "CUSTOMER_ID": "Identifiant",
            "PLAN_NAME": "Forfait",
            "REGION": "Région",
            "proba_churn": st.column_config.ProgressColumn(
                "Score de risque", min_value=0, max_value=1, format="%.2f"),
            "perte_attendue_fcfa": "Revenu exposé"})
    st.caption("Les 400 premiers abonnés sont affichés ; l'export contient la liste "
               "entière.")
    st.download_button("Exporter la liste complète",
                       cible.to_csv(index=False).encode("utf-8"),
                       f"appel_retention_top{budget}pct.csv", "text/csv",
                       type="primary")


# ========================================================= PRÉVISION DES OFFRES
elif page == "Prévision des offres":
    T.entete("Prévision des offres",
             "Trois modèles apprennent l'historique quotidien des offres, sont jugés sur "
             "les derniers jours qu'on leur a cachés, puis regardent devant. Choisissez une "
             "offre, plusieurs, ou toutes. Ici, uniquement les offres : le churn n'intervient "
             "que plus loin, dans le modèle de flux.")

    ss = st.session_state
    LONG, source = ss.get("long_import"), "importées"
    if LONG is None:
        LONG, source = charger_offres_long(), "de référence"
    if LONG is None:
        LONG = D["series"].reset_index().melt("date", var_name="offre",
                                              value_name="activations")
        source = "de référence (trois forfaits Blue One seulement)"
        st.warning("Le fichier <offres_long.csv> est introuvable dans le dossier outputs : "
                   "seuls les forfaits Blue One sont proposés. Lancez 02_parse_offres.py "
                   "ou importez votre classeur dans la première section.")

    resume = PL.resume_offres(LONG)
    dispo = list(resume.index)                       # de la plus vendue à la moins vendue
    fam = PL.familles(dispo)
    gammes = list(dict.fromkeys(fam[o] for o in dispo))
    dmin, dmax = LONG["date"].min().date(), LONG["date"].max().date()
    st.caption(f"Données {source} : {len(dispo)} offres, du {dmin:%d/%m/%Y} au {dmax:%d/%m/%Y}. "
               "La référence (répéter la semaine passée) est toujours calculée : elle sert de point de comparaison.")

    # ------------------------------------------------ choix des offres
    ss.setdefault("sel_offres", [o for o in ["Blue One M"] if o in dispo] or dispo[:1])
    ss["sel_offres"] = [o for o in ss["sel_offres"] if o in dispo]

    def _toutes():
        ss["sel_offres"] = list(dispo)

    def _aucune():
        ss["sel_offres"] = []

    def _ajouter_gamme():
        g = ss.get("sel_gamme")
        if g:
            ss["sel_offres"] = [o for o in dispo if o in ss["sel_offres"] or fam[o] == g]
        ss["sel_gamme"] = ""

    st.multiselect("Offres à prévoir", dispo, key="sel_offres",
                   placeholder="Tapez un nom pour chercher une offre",
                   help="Triées de la plus vendue à la moins vendue.")
    b = st.columns([1.2, 2.6, 2.6])
    b[0].button("Toutes les offres", on_click=_toutes, use_container_width=True)
    b[1].selectbox("Ajouter une gamme entière", [""] + gammes, key="sel_gamme",
                   on_change=_ajouter_gamme, format_func=lambda g: g or "Ajouter une gamme entière",
                   label_visibility="collapsed")
    mode = b[2].radio("Calcul", ["Total des offres choisies", "Chaque offre séparément"],
                      label_visibility="collapsed")
    choix = list(ss["sel_offres"])
    if not choix:
        st.info("Sélectionnez au moins une offre.")
        st.stop()
    separe = mode.startswith("Chaque") and len(choix) > 1

    # ------------------------------------------------ réglages
    h_prev = st.select_slider("Horizon de prévision (jours)", [30, 60, 90, 120, 150, 180], 90)
    deb0 = pd.Timestamp("2023-10-01").date()
    deb0 = deb0 if dmin <= deb0 <= dmax else dmin
    with st.expander("Réglages avancés : période, modèles, test"):
        c = st.columns([1.3, 1.3, 1.1, 2.4])
        debut = c[0].date_input("Début de la période", deb0, min_value=dmin, max_value=dmax)
        fin = c[1].date_input("Fin de la période", dmax, min_value=dmin, max_value=dmax)
        h_test = c[2].slider("Jours de test", 30, 120, 60, 10,
                             help="Derniers jours cachés aux modèles pour les évaluer.")
        modeles = c[3].multiselect(
            "Modèles", PV.MODELES_DISPONIBLES,
            default=["Prophet", "XGBoost"] if separe and len(choix) > 5
            else ["Prophet", "XGBoost", "LSTM"],
            help="SARIMA est facultatif : c'est le plus lent des quatre. Le LSTM ajoute "
                 "environ une minute par offre.")
        e = st.columns(2)
        fenetre = e[0].slider("LSTM : fenêtre d'entrée (jours)", 14, 56, 28, 7)
        epochs = e[1].slider("LSTM : époques maximum", 20, 200, 80, 10,
                             help="L'entraînement s'arrête plus tôt si l'erreur de "
                                  "validation cesse de baisser.")

    with st.expander("Pourquoi ces modèles ?"):
        st.markdown(
            "Le **naïf saisonnier** rejoue la dernière semaine observée. C'est la barre à "
            "franchir : un modèle sophistiqué qui ne le bat pas ne sert à rien. "
            "**SARIMA** est le classique des séries au rythme hebdomadaire. **Prophet**, "
            "**XGBoost** et le **LSTM** représentent trois familles plus récentes : "
            "décomposition en tendance et saisons, arbres de décision, réseau récurrent.\n\n"
            "Avec moins de deux ans de données quotidiennes, le LSTM a peu de matière pour "
            "apprendre. Il mérite d'être essayé, sans garantie de gagner, et un résultat "
            "où le naïf reste devant est une information utile, pas un échec.")

    tab, jr = series_offres_cache(LONG, tuple(choix), debut, fin)
    with st.expander("Ce qui a été fait aux séries"):
        st.markdown("\n".join(f"- {x}" for x in jr))
    besoin = h_test + fenetre + 60
    signature = (len(LONG), float(LONG["activations"].sum()), str(debut), str(fin))
    params = (h_test, h_prev, tuple(sorted(modeles)), fenetre, epochs)

    # ------------------------------------------------ total (ou une seule offre)
    if not separe:
        total = tab.sum(axis=1)
        nom = choix[0] if len(choix) == 1 else f"Total de {len(choix)} offres"
        serie, motif = PL.plage_active(total, tab.index.max())
        if motif is None and len(serie) < besoin:
            motif = (f"historique trop court ({len(serie)} jours) pour {h_test} jours de "
                     "test et cette fenêtre : réduisez-les ou avancez le début de la période")
        if motif:
            st.warning(f"{nom} : {motif}.")
            st.stop()
        if len(choix) > 1:
            part = (tab.sum() / tab.sum().sum()).sort_values(ascending=False)
            with st.expander(f"Poids des {len(choix)} offres dans ce total"):
                st.dataframe(pd.DataFrame({"Offre": part.index,
                                           "Activations": tab.sum()[part.index].values,
                                           "Part": (part.values * 100).round(1)}),
                             width="stretch", hide_index=True)
            if len(serie) < len(total):
                st.caption(f"Le total est calculé à partir du {serie.index[0]:%d/%m/%Y}, "
                           "date de la première activation.")

        cle = (tuple(choix), signature) + params
        cache = ss.setdefault("prev_live", {})
        if st.button("Lancer la prévision", type="primary", disabled=not modeles):
            barre = st.progress(0.0, text="Préparation")
            cache[cle] = PV.lancer(
                serie, modeles, h_test, h_prev, fenetre, epochs,
                rappel=lambda p, m: barre.progress(min(float(p), 1.0), text=m))
            barre.empty()
        res = cache.get(cle)
        if res is None:
            st.info("Choisissez les offres et les modèles, puis lancez la prévision. Comptez "
                    "une à deux minutes avec le LSTM.")
        else:
            afficher_resultat(res, serie, nom)

    # ------------------------------------------------ une prévision par offre
    else:
        eligibles, exclues = [], {}
        for o in choix:
            s_o, motif = PL.plage_active(tab[o], tab.index.max())
            if motif is None and len(s_o) < besoin:
                motif = f"historique trop court ({len(s_o)} jours) pour ces réglages"
            if motif:
                exclues[o] = motif
            else:
                eligibles.append((o, s_o))
        if exclues:
            with st.expander(f"{len(exclues)} offre(s) écartée(s) de la prévision individuelle"):
                st.markdown("\n".join(f"- **{o}** : {m}" for o, m in exclues.items()))
                st.caption("Ces offres restent utilisables dans le mode « Total des offres "
                           "choisies ».")
        if not eligibles:
            st.warning("Aucune des offres choisies n'a assez de données pour une prévision "
                       "individuelle.")
            st.stop()

        cle = (tuple(o for o, _ in eligibles), signature) + params
        cache = ss.setdefault("prev_multi", {})
        secondes = len(eligibles) * (2 + 2 * ("Prophet" in modeles) + 3 * ("XGBoost" in modeles)
                                     + 45 * ("LSTM" in modeles) + 30 * ("SARIMA" in modeles))
        st.caption(f"{len(eligibles)} offre(s) seront calculées, environ "
                   + (f"{round(secondes)} seconde(s)" if secondes < 90
                      else f"{round(secondes / 60)} minute(s)")
                   + " avec ces modèles (estimation).")
        if st.button(f"Lancer la prévision pour {len(eligibles)} offre(s)", type="primary",
                     disabled=not modeles):
            barre = st.progress(0.0, text="Préparation")
            n = len(eligibles)
            sortie = {}
            for i, (o, s_o) in enumerate(eligibles):
                try:
                    sortie[o] = PV.lancer(
                        s_o, modeles, h_test, h_prev, fenetre, epochs,
                        rappel=lambda p, m, i=i, o=o: barre.progress(
                            min((i + float(p)) / n, 1.0), text=f"{o} ({i + 1}/{n}) : {m}"))
                except Exception as e_:
                    sortie[o] = {"erreurs": {"Calcul": str(e_)}, "retenu": None}
            barre.empty()
            cache[cle] = sortie
        multi = cache.get(cle)
        if multi is None:
            st.info("Choisissez les modèles, puis lancez la prévision. Les offres sont "
                    "calculées l'une après l'autre.")
        else:
            lignes, bandes = [], []
            for o, r_ in multi.items():
                if r_.get("retenu") is None:
                    lignes.append({"Offre": o, "Modèle retenu": "échec", "MAPE (%)": np.nan,
                                   "MAPE référence (%)": np.nan, "Activations prévues": np.nan})
                    continue
                m_ = r_["metriques"]
                lignes.append({
                    "Offre": o, "Modèle retenu": r_["retenu"],
                    "MAPE (%)": m_[r_["retenu"]]["MAPE (%)"],
                    "MAPE référence (%)": m_[PV.REFERENCE]["MAPE (%)"],
                    "Activations prévues": round(float(r_["futur"][r_["retenu"]].sum())),
                })
                bandes.append(PV.tableau_futur(r_, o))
            T.section("Vue d'ensemble", f"{len(multi)} offre(s), prévision à {h_prev} jours")
            synth = pd.DataFrame(lignes)
            synth["Meilleur que la référence"] = np.where(synth["MAPE (%)"] < synth["MAPE référence (%)"],
                                            "oui", "non")
            st.dataframe(synth.sort_values("Activations prévues", ascending=False),
                         width="stretch", hide_index=True)
            st.caption("Pour les offres à ventes espacées, la MAPE est peu parlante (elle "
                       "ignore les jours à zéro) : lisez-la avec prudence.")
            if bandes:
                st.download_button("Exporter toutes les prévisions (CSV)",
                                   pd.concat(bandes).to_csv(index=False).encode("utf-8"),
                                   f"previsions_offres_{h_prev}j.csv", "text/csv")
            T.section("Détail d'une offre")
            voir = st.selectbox("Offre à détailler", list(multi), label_visibility="collapsed")
            serie_v = dict(eligibles)[voir]
            afficher_resultat(multi[voir], serie_v, voir)

    with st.expander("Les modèles en formules"):
        f_tabs = st.tabs(["Prophet", "XGBoost", "LSTM", "Références", "Mesures d'erreur"])
        with f_tabs[0]:
            st.markdown("Prophet découpe la série en une tendance et une saison. En mode "
                        "multiplicatif, la saison agit comme un coefficient sur la tendance :")
            st.latex(r"y(t) = g(t)\,\bigl(1 + s(t)\bigr) + \varepsilon_t")
            st.markdown("La tendance est une droite par morceaux, dont la pente change à "
                        "quelques dates de rupture :")
            st.latex(r"g(t) = \bigl(k + \mathbf{a}(t)^{\top}\boldsymbol{\delta}\bigr)\,t "
                     r"+ \bigl(m + \mathbf{a}(t)^{\top}\boldsymbol{\gamma}\bigr)")
            st.markdown("La saison, de période P (7 jours pour la semaine, 365,25 pour "
                        "l'année), s'écrit comme une série de Fourier :")
            st.latex(r"s(t) = \sum_{n=1}^{N}\Bigl(a_n\cos\frac{2\pi n t}{P} "
                     r"+ b_n\sin\frac{2\pi n t}{P}\Bigr)")
        with f_tabs[1]:
            st.markdown("XGBoost additionne des arbres construits l'un après l'autre, chacun "
                        "corrigeant les erreurs des précédents. Ici, il devine l'activation "
                        "du jour à partir des jours d'avant et du calendrier :")
            st.latex(r"\hat y_t = \sum_{k=1}^{K} f_k(\mathbf{x}_t), \qquad "
                     r"\mathbf{x}_t = \bigl(y_{t-1},\dots,y_{t-28},\ \bar y^{(7)}_{t-1},\ "
                     r"\bar y^{(28)}_{t-1},\ \text{jour},\ \text{mois},\dots\bigr)")
            st.markdown("Les arbres sont choisis pour minimiser l'erreur, avec une pénalité "
                        "sur leur complexité :")
            st.latex(r"\mathcal{L} = \sum_t \bigl(y_t - \hat y_t\bigr)^2 + \sum_k \Omega(f_k), "
                     r"\qquad \Omega(f) = \gamma\,T_f + \tfrac{1}{2}\lambda\,\lVert w\rVert^2")
            st.markdown("Pour prévoir plusieurs jours de suite, chaque valeur prédite "
                        "redevient un retard du jour suivant. Les erreurs s'accumulent donc "
                        "avec l'horizon.")
        with f_tabs[2]:
            st.markdown("Le LSTM lit la fenêtre des derniers jours pas à pas. Il garde une "
                        "mémoire interne c<sub>t</sub> qu'il met à jour grâce à trois portes "
                        "(oubli, entrée, sortie) :", unsafe_allow_html=True)
            st.latex(r"\begin{aligned}"
                     r"f_t &= \sigma\bigl(W_f\,[h_{t-1},x_t] + b_f\bigr)\\"
                     r"i_t &= \sigma\bigl(W_i\,[h_{t-1},x_t] + b_i\bigr)\\"
                     r"\tilde c_t &= \tanh\bigl(W_c\,[h_{t-1},x_t] + b_c\bigr)\\"
                     r"c_t &= f_t \odot c_{t-1} + i_t \odot \tilde c_t\\"
                     r"o_t &= \sigma\bigl(W_o\,[h_{t-1},x_t] + b_o\bigr)\\"
                     r"h_t &= o_t \odot \tanh(c_t)"
                     r"\end{aligned}")
            st.markdown("Avant l'apprentissage, la série passe par un logarithme puis une "
                        "standardisation. On revient à l'échelle réelle après la prévision :")
            st.latex(r"z_t = \frac{\ln(1+y_t) - \mu}{\sigma}, \qquad "
                     r"\hat y_t = \exp\bigl(\sigma\,\hat z_t + \mu\bigr) - 1")
        with f_tabs[3]:
            st.markdown("Le naïf saisonnier répète la dernière semaine observée, T étant "
                        "le dernier jour connu :")
            st.latex(r"\hat y_{T+h} = y_{T-6+((h-1)\bmod 7)}")
            st.markdown("SARIMA(2,1,2)(1,1,1)<sub>7</sub> combine une différenciation "
                        "simple et une différenciation hebdomadaire, B étant l'opérateur "
                        "de retard :", unsafe_allow_html=True)
            st.latex(r"\Phi(B^{7})\,\phi(B)\,(1-B)(1-B^{7})\,y_t = "
                     r"\Theta(B^{7})\,\theta(B)\,\varepsilon_t")
        with f_tabs[4]:
            st.latex(r"\mathrm{MAPE} = \frac{100}{n}\sum_{t}\frac{|y_t - \hat y_t|}{|y_t|}"
                     r"\qquad \mathrm{sMAPE} = \frac{100}{n}\sum_{t}"
                     r"\frac{|y_t - \hat y_t|}{(|y_t| + |\hat y_t|)/2}")
            st.latex(r"\mathrm{MAE} = \frac{1}{n}\sum_t |y_t - \hat y_t|, \qquad "
                     r"\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_t (y_t - \hat y_t)^2}, \qquad "
                     r"\text{biais} = \frac{1}{n}\sum_t (\hat y_t - y_t)")
            st.markdown("L'intervalle à 90 % ajoute et retire à chaque prévision le quantile "
                        "à 90 % des erreurs absolues observées sur le test :")
            st.latex(r"\bigl[\hat y_t - q_{0{,}9},\ \hat y_t + q_{0{,}9}\bigr], \qquad "
                     r"q_{0{,}9} = \text{quantile}_{90\,\%}\bigl(|y_t - \hat y_t|\bigr)")


# ================================================================== SCORER
else:
    T.entete("Départs du réseau et évaluation d'un client",
             "D'abord, combien d'abonnés ont quitté le réseau, par jour, semaine, mois et "
             "an, région par région. Ensuite, le score d'un profil précis : il sert à "
             "prioriser un appel, pas à estimer une probabilité de départ absolue.")

    # ---- départs par période et par région
    T.section("Abonnés partis du réseau", "par jour, semaine, mois et an")
    DEPARTS, err_dep = charger_departs()
    if DEPARTS is None:
        st.error(err_dep)
    elif DEPARTS.empty:
        st.warning("Aucun départ daté dans le fichier des abonnés.")
    else:
        DP.afficher(DEPARTS, "du fichier des abonnés")

    # ---- évaluation d'un client
    T.section("Évaluer un client")
    MOD = charger_modele()
    if MOD is None:
        st.stop()
    if MOD.get("origine") == "reentraine":
        T.note("Le fichier <code>modele_churn.joblib</code> vient d'une autre version de "
               "scikit-learn. Le modèle a été reconstruit depuis les données nettoyées : "
               "résultats identiques, aucune action requise.")

    c = st.columns([2, 3])
    with c[0]:
        plan = st.selectbox("Forfait", ["Blue One S", "Blue One M", "Blue One L"], index=1)
        region = st.selectbox("Région", sorted(scores["REGION"].unique()))
        genre = st.selectbox("Genre", ["Male", "Female"],
                             format_func=lambda g: "Homme" if g == "Male" else "Femme")
        age = st.slider("Âge", 15, 60, 25)
        tenure = st.slider("Ancienneté (mois)", 1, 12, 4)
        recharges = st.slider("Recharges effectuées", 0, 10, 2)

    prix = PRIX[plan]
    ligne = pd.DataFrame([{
        "TENURE_MONTHS": tenure, "PLAN_NAME": plan, "MONTHLY_CHARGE": prix,
        "RECHARGE_FREQUENCY": recharges, "REGION": region, "GENDER": genre, "AGE": age,
        "anciennete_jours": tenure * 30,
        "intensite_recharge": recharges / max(tenure, 1),
        "revenu_cumule": prix * recharges,
        "jamais_recharge": int(recharges == 0), "mois_creation": 3,
        "tranche_age": ("<=20" if age <= 20 else "21-25" if age <= 25
                        else "26-30" if age <= 30 else ">30"),
    }])[MOD["colonnes"]]
    p = float(MOD["pipeline"].predict_proba(ligne)[0, 1])
    rang = float((scores["proba_churn"] < p).mean())
    niveau, ton = (("élevé", "brique") if p > .6 else
                   ("moyen", "ambre") if p > .35 else ("faible", "vert"))

    with c[1]:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100,
            number={"suffix": "\u202f%", "valueformat": ".0f",
                    "font": {"size": 46, "color": T.ENCRE, "family": T.POLICE_TITRE}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0, "tickcolor": T.TRAIT,
                         "tickfont": {"size": 11, "color": T.GRIS}},
                "bar": {"color": T.ENCRE, "thickness": .22},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [{"range": [0, 35], "color": "#E4EFE8"},
                          {"range": [35, 60], "color": "#FAF0DC"},
                          {"range": [60, 100], "color": "#F6DEDC"}]}))
        T.afficher(fig, 260)
        k = st.columns(3)
        with k[0]:
            T.tuile("Niveau de risque", niveau.capitalize(), "lecture relative", ton)
        with k[1]:
            T.tuile("Rang dans la base", T.pct(rang, 0),
                    "d'abonnés moins exposés", jauge=rang * 100)
        with k[2]:
            T.tuile("Revenu mensuel exposé", T.fcfa(p * prix),
                    "forfait à " + T.nb(prix, "FCFA"))

    T.section("Effet de la fréquence de recharge", "variable la plus décisive du modèle")
    courbe = []
    for r in range(0, 11):
        l2 = ligne.copy()
        l2["RECHARGE_FREQUENCY"] = r
        l2["intensite_recharge"] = r / max(tenure, 1)
        l2["revenu_cumule"] = prix * r
        l2["jamais_recharge"] = int(r == 0)
        courbe.append(float(MOD["pipeline"].predict_proba(l2)[0, 1]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(11)), y=courbe, line=dict(color=T.BLEU, width=2.4),
                             name="score", hovertemplate="%{x} recharges : %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[recharges], y=[p], mode="markers", name="profil saisi",
                             marker=dict(color=T.AMBRE, size=12,
                                         line=dict(color="#fff", width=2))))
    fig.update_xaxes(title="recharges effectuées", dtick=1)
    fig.update_yaxes(title="score de risque", tickformat=".0%")
    fig.update_layout(hovermode="closest")
    T.afficher(fig, 280)
    st.caption("Toutes les autres caractéristiques restent celles saisies à gauche.")
