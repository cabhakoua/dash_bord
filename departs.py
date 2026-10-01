"""
Abonnés partis du réseau : par jour, par semaine, par mois et par an, avec choix des régions.

Entrée : une table à un départ par ligne, colonnes `REGION` et `date_depart`
(voir `pipeline.departs_churn`).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme as T

# (titre de l'onglet, fréquence pandas, nom de la période, lissage, libellé du lissage)
ONGLETS = [
    ("Par jour", "D", "jour", 7, "moyenne mobile 7 jours"),
    ("Par semaine", "W-SUN", "semaine", 4, "moyenne mobile 4 semaines"),
    ("Par mois", "M", "mois", None, None),
    ("Par an", "Y", "an", None, None),
]
MAX_COURBES = 6                 # au-delà, les couleurs se répètent : on garde le total
PALE = "rgba(178,58,53,.42)"    # BRIQUE atténué : période incomplète


# ------------------------------------------------------------------ calculs
def compter(dep, freq, regions, bornes):
    """Tableau période x région du nombre de départs, sans trou (zéro les jours calmes)."""
    plein = pd.period_range(pd.Period(bornes[0], freq), pd.Period(bornes[1], freq), freq=freq)
    if dep.empty:
        return pd.DataFrame(0, index=plein, columns=regions)
    p = dep["date_depart"].dt.to_period(freq)
    return (pd.crosstab(p, dep["REGION"])
            .reindex(index=plein, columns=regions, fill_value=0).astype(int))


def etiquette(per, freq):
    if freq == "D":
        return per.start_time.strftime("%d/%m/%Y")
    if freq.startswith("W"):
        return "sem. du " + per.start_time.strftime("%d/%m/%y")
    if freq == "M":
        return T.etiquette_mois(str(per))
    return str(per.year)


def _partiel(index, bornes):
    """Vrai pour une période qui déborde de la fenêtre où les départs sont observés."""
    debut, fin = index.start_time, index.end_time.normalize()
    return np.asarray((debut < bornes[0]) | (fin > bornes[1]))


def _variation(a, b):
    return None if b == 0 else (a - b) / b


# ------------------------------------------------------------------ affichage
def _tuiles(total, partiel, freq, nom, bornes):
    complet = total[~partiel]
    base = complet if len(complet) else total
    if freq == "D":
        dern, prec = total.iloc[-7:].sum(), total.iloc[-14:-7].sum()
        lib = "7 derniers jours"
        ok = len(total) >= 14
    else:
        ok = len(complet) >= 2
        dern = complet.iloc[-1] if len(complet) else 0
        prec = complet.iloc[-2] if ok else 0
        lib = f"dernier {nom} complet" if nom != "semaine" else "dernière semaine complète"
    var = _variation(dern, prec) if ok else None
    pic = total.idxmax()

    k = st.columns(4)
    with k[0]:
        T.tuile("Départs sur la période", T.nb(total.sum()),
                f"du {bornes[0]:%d/%m/%Y} au {bornes[1]:%d/%m/%Y}", "brique")
    with k[1]:
        T.tuile(f"Moyenne par {nom}", T.dec(base.mean(), 1),
                "périodes complètes seulement" if len(complet) else "période incomplète")
    with k[2]:
        T.tuile(f"Pic ({nom})", T.nb(total.max()), etiquette(pic, freq), "ambre")
    with k[3]:
        if var is None:
            T.tuile(lib.capitalize(), T.nb(dern), "comparaison impossible")
        else:
            T.tuile(lib.capitalize(), T.nb(dern),
                    ("+" if var >= 0 else "\u2212") + T.pct(abs(var), 0) + " sur la précédente",
                    "brique" if var > 0 else "vert")


def _graphique(wide, partiel, freq, lisse, lib_lisse, par_region):
    idx = wide.index
    labels = [etiquette(p, freq) for p in idx]
    cat = freq in ("M", "Y")
    x = labels if cat else list(idx.start_time)
    fig = go.Figure()
    if par_region:
        for i, r in enumerate(wide.columns):
            y = wide[r].rolling(lisse, min_periods=1).mean() if freq == "D" else wide[r]
            fig.add_trace(go.Scatter(
                x=x, y=y, name=r, customdata=labels,
                mode="lines+markers" if len(idx) < 30 else "lines",
                line=dict(color=T.SEQUENCE[i % len(T.SEQUENCE)], width=1.8),
                hovertemplate="%{customdata}<br>%{y:.1f} départs<extra>" + r + "</extra>"))
        fig.update_yaxes(title="départs / jour (moyenne 7 jours)" if freq == "D" else "départs")
        fig.update_layout(hovermode="x unified")
    else:
        total = wide.sum(axis=1)
        fig.add_trace(go.Bar(
            x=x, y=total.values, name="départs",
            marker=dict(color=[PALE if p else T.BRIQUE for p in partiel], line=dict(width=0)),
            customdata=np.stack([labels, np.where(partiel, "période incomplète", "")], -1),
            hovertemplate="%{customdata[0]}<br>%{y:,.0f} départs<br>%{customdata[1]}"
                          "<extra></extra>"))
        if lisse:
            fig.add_trace(go.Scatter(
                x=x, y=total.rolling(lisse, min_periods=1).mean().values, name=lib_lisse,
                customdata=labels, line=dict(color=T.ENCRE, width=1.8),
                hovertemplate="%{customdata}<br>%{y:.1f} en moyenne<extra></extra>"))
        fig.update_yaxes(title="départs")
        fig.update_layout(hovermode="closest", bargap=.35 if cat else .15)
    if cat:
        fig.update_xaxes(type="category")
    else:
        T.axe_mois(fig, x)
    T.afficher(fig, 340)


def _tableau(wide, partiel, freq, par_region):
    total = wide.sum(axis=1)
    brut = pd.DataFrame({"Période": [etiquette(p, freq) for p in wide.index]})
    if par_region:
        for r in wide.columns:
            brut[r] = wide[r].values
    brut["Départs"] = total.values
    var = total.pct_change().replace([np.inf, -np.inf], np.nan)
    brut["Écart avec la précédente (%)"] = (var * 100).round(1).values
    brut["Statut"] = np.where(partiel, "incomplète", "")
    brut = brut.iloc[::-1].reset_index(drop=True)          # le plus récent en premier
    aff = brut.copy()
    aff["Écart avec la précédente (%)"] = [
        "" if pd.isna(v) else ("+" if v > 0 else "") + T.dec(v, 1) + "\u202f%"
        for v in brut["Écart avec la précédente (%)"]]
    return brut, aff


def _onglet(dep, regions, bornes, cfg, par_region):
    _, freq, nom, lisse, lib_lisse = cfg
    wide = compter(dep, freq, regions, bornes)
    partiel = _partiel(wide.index, bornes)
    total = wide.sum(axis=1)
    _tuiles(total, partiel, freq, nom, bornes)
    st.write("")
    _graphique(wide, partiel, freq, lisse, lib_lisse, par_region)
    if partiel.any():
        st.caption("Les barres pâles sont des périodes incomplètes : la fenêtre "
                   "d'observation commence ou s'arrête au milieu, ne les comparez pas "
                   "aux périodes pleines.")
    brut, aff = _tableau(wide, partiel, freq, par_region)
    with st.expander(f"Tableau récapitulatif ({len(brut)} lignes)"):
        st.dataframe(aff, width="stretch", hide_index=True)
    st.download_button("Exporter ce tableau (CSV)", brut.to_csv(index=False).encode("utf-8"),
                       f"departs_par_{nom}.csv", "text/csv", key=f"dl_dep_{freq}")


def afficher(dep, source):
    """Section complète : filtre de régions, quatre onglets, répartition régionale."""
    ss = st.session_state
    tous = sorted(dep["REGION"].unique())
    bornes = (dep["date_depart"].min().normalize(), dep["date_depart"].max().normalize())
    ss["dep_regions"] = [r for r in ss.get("dep_regions", []) if r in tous]

    c = st.columns([3, 2])
    choix = c[0].multiselect("Régions", tous, key="dep_regions",
                             placeholder="Toutes les régions")
    regions = choix or tous
    mode = c[1].radio("Courbes", ["Total de la sélection", "Une courbe par région"],
                      horizontal=True, label_visibility="collapsed", key="dep_mode")
    par_region = mode.startswith("Une") and 1 < len(regions) <= MAX_COURBES
    if mode.startswith("Une") and len(regions) > MAX_COURBES:
        st.caption(f"La comparaison est limitée à {MAX_COURBES} régions pour rester lisible : "
                   "le total est affiché. Réduisez la sélection pour comparer.")

    sel = dep[dep["REGION"].isin(regions)]
    st.caption(f"Départs {source} : {T.nb(len(sel))} abonnés partis"
               f"{'' if len(regions) == len(tous) else f' dans {len(regions)} région(s)'}, "
               f"du {bornes[0]:%d/%m/%Y} au {bornes[1]:%d/%m/%Y}.")

    onglets = st.tabs([o[0] for o in ONGLETS])
    for tab, cfg in zip(onglets, ONGLETS):
        with tab:
            _onglet(sel, regions, bornes, cfg, par_region)

    # ---- répartition régionale
    T.section("Répartition par région", "sur toute la période")
    c = st.columns([2, 3])
    par = sel.groupby("REGION").size().reindex(regions, fill_value=0).sort_values()
    with c[0]:
        part = par / max(par.sum(), 1)
        fig = go.Figure(go.Bar(
            x=par.values, y=par.index, orientation="h",
            marker=dict(color=T.BRIQUE, line=dict(width=0)),
            customdata=part.values,
            hovertemplate="%{y}<br>%{x:,.0f} départs (%{customdata:.1%})<extra></extra>"))
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(title="départs")
        fig.update_layout(hovermode="closest")
        T.afficher(fig, max(220, 34 * len(par) + 60))
    with c[1]:
        mois = compter(sel, "M", regions, bornes)
        piv = mois.T
        piv.columns = [etiquette(p, "M") for p in mois.index]
        piv["Total"] = piv.sum(axis=1)
        piv.loc["Ensemble"] = piv.sum()
        piv = piv.astype(int)
        piv.index.name = "Région"
        st.dataframe(piv.reset_index(), width="stretch", hide_index=True)
        st.download_button("Exporter (CSV)", piv.reset_index().to_csv(index=False).encode("utf-8"),
                           "departs_region_mois.csv", "text/csv", key="dl_dep_region")
    st.caption("Le premier et le dernier mois peuvent être incomplets : voir la note "
               "ci-dessous sur la façon dont les départs sont datés.")

    with st.expander("Comment sont comptés les départs ?"):
        T.note(
            "<b>Date retenue.</b> Un abonné est compté le jour de sa dernière trace sur le "
            "réseau (champ <code>PURGE_TIME_ATSGSN</code>), à condition d'être classé parti, "
            "c'est-à-dire sans activité depuis plus de 60 jours à la date de l'extrait. "
            f"Par construction, les départs des 60 derniers jours ne sont pas encore "
            f"visibles : la série s'arrête le {bornes[1]:%d/%m/%Y}. "
            "Les effectifs décrivent l'extrait, pas toute la base de CAMTEL.")
