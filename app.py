"""
Console de retention CAMTEL : churn et demande en offres.
Lancement :  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
with st.sidebar:
    st.markdown(
        "<div class='marque'><div class='nom'>Rétention Blue One</div>"
        "<div class='sous'>Console d'analyse du churn et de la demande</div></div>",
        unsafe_allow_html=True)
    page = st.radio("Section", [
        "Situation", "Qualité des données", "Modèles de churn", "Ciblage",
        "Demande en offres", "Modèle de flux", "Scorer un abonné"],
        label_visibility="collapsed")
    st.markdown(
        "<div class='pied'>Extrait HSS de janvier à juillet 2025, 13 937 abonnés.<br>"
        "Activations d'avril 2022 à août 2025, 41 relevés mensuels.</div>",
        unsafe_allow_html=True)

flux, scores = D["flux"], D["scores"]


# ================================================================== SITUATION
if page == "Situation":
    T.entete("Situation de la base Blue One",
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
        "Le score classe les abonnés par risque relatif ; il n'estime pas le niveau "
        "absolu du churn. L'extrait d'apprentissage contient 44 % de churners, bien "
        "au-delà de ce qu'une base stable peut supporter. Les montants ci-dessus "
        "reposent donc sur le scénario central du modèle de flux, pas sur le taux brut "
        "de l'échantillon.")


# ============================================================ QUALITÉ DONNÉES
elif page == "Qualité des données":
    a = D["audit"]
    T.entete("Qualité des données",
             "Ce que les deux fichiers sources contiennent réellement, et les corrections "
             "appliquées avant toute modélisation.")

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

    T.section("Deux défauts qui invalident un modèle naïf")
    c = st.columns(2)
    with c[0]:
        T.note(
            f"<b>Fuite d'information.</b> La règle « dernière purge antérieure de plus "
            f"de 60 jours au {a['date_ref'][:10]} » reproduit l'étiquette dans "
            f"{a['regle_60j_concordance']:.2%} des cas, avec "
            f"{a['regle_60j_exceptions']} exceptions. Garder les variables dérivées de "
            f"la purge donne une AUC de 0,996 sans rien prédire : le modèle réapprend "
            f"la définition de la cible.", alerte=True)
    with c[1]:
        T.note(
            f"<b>Censure à droite.</b> {a['n_censures']} abonnés créés après "
            f"{a['seuil_censure'][:10]} ne pouvaient pas cumuler 60 jours d'inactivité "
            f"avant la fin de la fenêtre, et on y compte "
            f"{a['churners_parmi_censures']} churner. Leur étiquette traduit la fenêtre "
            f"d'observation, pas la fidélité. Ils sortent de la classification et "
            f"alimentent l'analyse de survie.")

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
                     horizontal=True, label_visibility="collapsed")
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
    T.entete("Modèles de churn",
             "Trois classifieurs entraînés sur les mêmes variables, évalués sur 25 % des "
             "observations mises de côté. Validation croisée à cinq blocs sur le reste.")

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
            st.caption("Le score de Brier mesure l'écart entre le score annoncé et la "
                       "réalité : plus il est bas, plus le score est exploitable tel quel.")
        else:
            st.dataframe(t[["Modele", "Exactitude", "Precision (churn)",
                            "Rappel (churn)", "F1 (churn)", "Precision (fidele)",
                            "Rappel (fidele)", "VN / FP / FN / VP"]],
                         width="stretch", hide_index=True)
            st.caption("VN / FP / FN / VP : vrais négatifs, faux positifs, faux "
                       "négatifs, vrais positifs sur les 3 157 abonnés du jeu de test.")

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
            "Retirer la fréquence de recharge fait tomber l'AUC à 0,57. C'est "
            "l'information disponible qui plafonne la prédiction, pas l'algorithme : "
            "un modèle plus complexe ne changerait rien, des données d'usage "
            "(volume consommé, appels au service client, réclamations) changeraient tout.")

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


# ===================================================================== CIBLAGE
elif page == "Ciblage":
    T.entete("Qui appeler cette semaine",
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


# ============================================================ DEMANDE EN OFFRES
elif page == "Demande en offres":
    T.entete("Demande en offres",
             "Quatre méthodes de prévision mises en concurrence sur les mêmes 60 derniers "
             "jours. Celle qui gagne sert à projeter les 90 jours suivants.")

    t = D["t_dem"]
    offre = st.radio("Forfait", sorted(t["Forfait"].unique()), index=1, horizontal=True,
                     label_visibility="collapsed")
    sous = t[t["Forfait"] == offre].copy()
    retenu = sous.loc[sous["Retenu"] == "oui", "Modele"]
    retenu = retenu.iloc[0] if len(retenu) else sous.loc[sous["MAPE_%"].idxmin(), "Modele"]

    c = st.columns(4)
    ref = sous[sous["Modele"].str.startswith("Naif")]["MAPE_%"].iloc[0]
    best = sous[sous["Modele"] == retenu].iloc[0]
    with c[0]:
        T.tuile("Modèle retenu", retenu.split(" (")[0], retenu, "vert")
    with c[1]:
        T.tuile("Erreur moyenne", T.dec(best["MAPE_%"], 2, "%"), "MAPE sur 60 jours")
    with c[2]:
        T.tuile("Gain sur la référence", "\u2212" + T.pct(1 - best["MAPE_%"] / ref, 0),
                "naïf saisonnier : " + T.dec(ref, 1, "%"), "ambre")
    with c[3]:
        p90 = D["prev"][D["prev"]["offre"] == offre]
        T.tuile("Prévu sur 90 jours", T.nb(p90["prevision"].sum()),
                T.fcfa(p90["prevision"].sum() * PRIX[offre]))

    T.section("Test hors échantillon", "60 jours jamais vus à l'entraînement")
    pt = D["prev_test"]
    pt = pt[pt["offre"] == offre]
    fig = go.Figure()
    reel = pt[pt["serie"] == "reel"]
    for i, m in enumerate([s for s in pt["serie"].unique() if s != "reel"]):
        s = pt[pt["serie"] == m]
        fig.add_trace(go.Scatter(x=s["date"], y=s["valeur"], name=m,
                                 line=dict(color=T.SEQUENCE[i + 1], width=1.3,
                                           dash="dot" if m.startswith("Naif") else None),
                                 visible=True if m == retenu or m.startswith("Naif")
                                 else "legendonly"))
    fig.add_trace(go.Scatter(x=reel["date"], y=reel["valeur"], name="réalisé",
                             line=dict(color=T.ENCRE, width=2.2)))
    fig.update_yaxes(title="activations / jour")
    fig.update_layout(legend=dict(font=dict(size=11)))
    T.axe_mois(fig, reel["date"])
    T.afficher(fig, 340)
    st.caption("Les courbes masquées se réaffichent d'un clic dans la légende.")

    T.section("Comparaison des méthodes")
    c = st.columns([2, 3])
    with c[0]:
        ordre = sous.sort_values("MAPE_%", ascending=False)
        fig = go.Figure(go.Bar(
            x=ordre["MAPE_%"], y=ordre["Modele"], orientation="h",
            marker=dict(color=[T.VERT if m == retenu else
                               T.TRAIT if m.startswith("Naif") else T.CIEL
                               for m in ordre["Modele"]], line=dict(width=0)),
            text=[T.dec(v, 1, "%") for v in ordre["MAPE_%"]], textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>MAPE %{x:.2f} %<extra></extra>"))
        fig.update_xaxes(title="erreur relative moyenne (MAPE)",
                         range=[0, ordre["MAPE_%"].max() * 1.22])
        fig.update_yaxes(showgrid=False)
        fig.update_layout(hovermode="closest", margin=dict(r=20))
        T.afficher(fig, 290)
    with c[1]:
        tab = sous.drop(columns=["Forfait"]).rename(columns={
            "Modele": "Méthode", "MAPE_%": "MAPE (%)", "sMAPE_%": "sMAPE (%)",
            "biais": "Biais moyen",
            "MAPE_origine_glissante_%": "MAPE origine glissante (%)"})
        st.dataframe(tab, width="stretch", hide_index=True)
        st.caption("MAE et RMSE sont en activations par jour, le biais aussi : négatif, "
                   "le modèle sous-estime. La dernière colonne rejoue la prévision sur "
                   "trois blocs de 30 jours à des dates différentes.")

    T.section("Prévision retenue", "90 jours, intervalle empirique à 90 %")
    agg = st.radio("Pas de temps", ["Jour", "Semaine", "Mois"], horizontal=True,
                   label_visibility="collapsed")
    q = p90.set_index("date")[["prevision", "borne_basse", "borne_haute"]]
    hist = D["series"][offre]
    if agg == "Semaine":
        q, hist = q.resample("W").sum(), hist.resample("W").sum()
    elif agg == "Mois":
        q, hist = q.resample("ME").sum(), hist.resample("ME").sum()
    hist = hist.iloc[-(len(q) * 2):]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q.index, y=q["borne_haute"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q.index, y=q["borne_basse"], fill="tonexty",
                             fillcolor="rgba(223,162,60,.16)", line=dict(width=0),
                             name="intervalle 90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="observé",
                             line=dict(color=T.BLEU, width=1.6)))
    fig.add_trace(go.Scatter(x=q.index, y=q["prevision"], name="prévu",
                             line=dict(color=T.AMBRE, width=2)))
    fig.update_yaxes(title=f"activations / {agg.lower()}")
    T.axe_mois(fig, list(hist.index) + list(q.index), pas=1 if agg != "Mois" else 1)
    T.afficher(fig, 340)


# =============================================================== MODÈLE DE FLUX
elif page == "Modèle de flux":
    T.entete("Modèle de flux",
             "Les deux modèles se rejoignent dans une identité de stock : ce qui entre "
             "vient de la prévision de demande, ce qui sort vient du modèle de churn.")

    st.markdown("""
<div style="background:#fff;border:1px solid var(--trait);border-radius:.5rem;
            padding:1.4rem 1.6rem;margin-bottom:1.1rem">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.12rem;color:var(--encre);
              margin-bottom:1rem;letter-spacing:-.01em">
    N<sub>t+1</sub> &nbsp;=&nbsp; N<sub>t</sub>
    &nbsp;+&nbsp; <span style="color:var(--vert)">&alpha;&thinsp;A<sub>t</sub></span>
    &nbsp;&minus;&nbsp; <span style="color:var(--brique)">p&thinsp;N<sub>t</sub></span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
              gap:1.1rem;font-size:.87rem;color:var(--gris);line-height:1.5">
    <div><b style="color:var(--encre)">N</b> : base active du forfait, estimée par le
         volume mensuel d'activations</div>
    <div><b style="color:var(--vert)">A</b> : activations prévues par le modèle de
         demande</div>
    <div><b style="color:var(--vert)">&alpha;</b> : part des activations qui sont de
         nouvelles souscriptions, le reste étant des renouvellements</div>
    <div><b style="color:var(--brique)">p</b> : taux de départ mensuel, issu du modèle
         de churn après recalage</div>
  </div>
</div>""", unsafe_allow_html=True)

    scen = D["scen"]
    LIBELLE = {
        "prudent (p*=2%)": "Prudent, 2 % de départs par mois",
        "central (p*=5%)": "Central, 5 %",
        "haut (p*=10%)": "Haut, 10 %",
        "echantillon brut (non recale)": "Échantillon brut, borne haute",
    }
    noms = scen["scenario"].unique().tolist()
    choix = st.radio("Scénario de taux de départ", noms, index=1, horizontal=True,
                     format_func=lambda n: LIBELLE.get(n, n))
    f = scen[scen["scenario"] == choix]

    debut = f[f["mois"] == f["mois"].iloc[0]]["base_debut"].sum()
    fin = f[f["mois"] == f["mois"].iloc[-1]]["base_fin"].sum()
    c = st.columns(4)
    with c[0]:
        T.tuile("α déduit", T.pct(f["alpha"].iloc[0], 2),
                "de l'identité de stock, non postulé")
    with c[1]:
        T.tuile("Départs sur l'horizon", T.nb(f["departs_attendus"].sum()),
                "tous forfaits", "brique")
    with c[2]:
        T.tuile("Revenu exposé", T.fcfa(f["revenu_a_risque_fcfa"].sum()),
                "cumul sur l'horizon", "ambre")
    with c[3]:
        T.tuile("Base en fin d'horizon",
                ("+" if fin >= debut else "\u2212") + T.pct(abs(fin / debut - 1), 2),
                T.nb(fin) + " abonnés",
                "vert" if fin >= debut else "brique")

    T.section("Trajectoire de la base selon le scénario")
    c = st.columns([3, 2])
    with c[0]:
        fig = go.Figure()
        for i, nom in enumerate(scen["scenario"].unique()):
            d = scen[(scen["scenario"] == nom) & (scen["forfait"] == "Blue One M")]
            base0 = d["base_debut"].iloc[0]
            y = [base0] + list(d["base_fin"])
            fig.add_trace(go.Scatter(
                x=list(range(len(y))), y=np.array(y) / 1000,
                name=LIBELLE.get(nom, nom),
                line=dict(color=T.SEQUENCE[i], width=2.4 if nom == choix else 1.2,
                          dash=None if nom == choix else "dot"),
                opacity=1 if nom == choix else .55,
                hovertemplate="mois %{x} : %{y:.1f} k abonnés<extra>"
                              + LIBELLE.get(nom, nom) + "</extra>"))
        fig.add_hline(y=scen["base_debut"].iloc[0] / 1000,
                      line=dict(color=T.TRAIT, width=1))
        toutes = scen[scen["forfait"] == "Blue One M"]
        lo = min(toutes["base_fin"].min(), toutes["base_debut"].min()) / 1000
        hi = max(toutes["base_fin"].max(), toutes["base_debut"].max()) / 1000
        marge = max((hi - lo) * 0.28, 1)
        fig.update_xaxes(title="mois d'horizon", dtick=1)
        fig.update_yaxes(title="base Blue One M (milliers)",
                         range=[lo - marge, hi + marge])
        fig.update_layout(hovermode="closest", legend=dict(font=dict(size=10.5)))
        T.afficher(fig, 340)
    with c[1]:
        d = f.groupby("forfait").agg(
            entrees=("entrees", "sum"), departs=("departs_attendus", "sum"),
            base=("base_debut", "first")).reset_index()
        d["ent_pct"] = d["entrees"] / d["base"]
        d["dep_pct"] = -d["departs"] / d["base"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=d["forfait"], y=d["ent_pct"], name="entrées",
                             marker_color=T.VERT, customdata=d["entrees"],
                             hovertemplate="+%{customdata:,.0f} abonnés"
                                           " (%{y:.1%})<extra></extra>"))
        fig.add_trace(go.Bar(x=d["forfait"], y=d["dep_pct"], name="départs",
                             marker_color=T.BRIQUE, customdata=d["departs"],
                             hovertemplate="−%{customdata:,.0f} abonnés"
                                           " (%{y:.1%})<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=T.ENCRE, width=1))
        fig.update_yaxes(title="part de la base, cumul sur l'horizon", tickformat=".0%")
        fig.update_layout(barmode="relative", hovermode="closest")
        T.afficher(fig, 340)
        st.caption("Rapporté à la taille de chaque base, pour rendre les trois forfaits "
                   "comparables malgré leurs volumes très différents.")

    T.section("Détail mois par mois")
    detail = f.drop(columns=["scenario"]).copy()
    detail["mois"] = detail["mois"].map(T.etiquette_mois)
    detail = detail.rename(columns={
        "alpha": "α", "mois": "Mois", "forfait": "Forfait",
        "taux_depart_mensuel": "Taux de départ", "base_debut": "Base au 1er",
        "activations_prevues": "Activations prévues", "entrees": "Entrées",
        "departs_attendus": "Départs", "base_fin": "Base en fin de mois",
        "variation_%": "Variation (%)", "revenu_a_risque_fcfa": "Revenu exposé (FCFA)",
        "revenu_base_fcfa": "Revenu de la base (FCFA)"})
    st.dataframe(detail, width="stretch", hide_index=True)

    T.note(
        "Le scénario <b>échantillon brut</b> applique tel quel le niveau de risque estimé "
        "sur l'extrait fourni, qui contient 44 % de churners. Il sert de borne haute, pas "
        "de prévision. Le paramètre α reste la principale inconnue : le classeur "
        "d'activations ne distingue pas une souscription d'un renouvellement. Une "
        "confirmation auprès de CAMTEL lèverait cette incertitude et réduirait l'écart "
        "entre les scénarios.")


# ================================================================== SCORER
else:
    MOD = charger_modele()
    T.entete("Scorer un abonné",
             "Le score situe un profil par rapport aux autres abonnés. Il sert à "
             "prioriser un appel, pas à estimer une probabilité de départ absolue.")
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
