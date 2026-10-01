"""
Habillage de l'application : feuille de style, gabarit de graphique, composants.

Le systeme de couleurs encode l'information plutot que de la decorer :
bleu = valeur observee, ambre = valeur prevue, rouge = depart / risque,
vert = abonne conserve. Aucune couleur n'est utilisee hors de ce role.
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------- jetons
ENCRE = "#0E1C2B"
PAPIER = "#F6F7F5"
BLEU = "#1668A6"
CIEL = "#86B8D8"
AMBRE = "#DFA23C"
BRIQUE = "#B23A35"
VERT = "#3D8A5C"
TRAIT = "#DCE2E5"
GRIS = "#5E7183"

SEQUENCE = [BLEU, AMBRE, VERT, BRIQUE, CIEL, "#6B5B95"]
COULEUR_FORFAIT = {"Blue One S": CIEL, "Blue One M": BLEU, "Blue One L": "#0E426D"}

POLICE = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"
POLICE_TITRE = "Archivo, IBM Plex Sans, system-ui, sans-serif"


# ----------------------------------------------------------------- graphiques
def _gabarit():
    axe = dict(
        showgrid=False, zeroline=False, showline=True, linewidth=1, linecolor=TRAIT,
        ticks="outside", ticklen=4, tickcolor=TRAIT,
        tickfont=dict(family=POLICE, size=11.5, color=GRIS),
        title_font=dict(family=POLICE, size=12, color=GRIS),
    )
    return go.layout.Template(layout=go.Layout(
        colorway=SEQUENCE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=POLICE, size=12.5, color=ENCRE),
        margin=dict(l=8, r=8, t=28, b=8),
        xaxis={**axe},
        yaxis={**axe, "showgrid": True, "gridcolor": "#E7ECEF", "gridwidth": 1,
               "showline": False, "ticks": ""},
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor=TRAIT, font_size=12.5,
                        font_family=POLICE, font_color=ENCRE),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(family=POLICE_TITRE, size=14, color=ENCRE), x=0, xanchor="left"),
    ))


pio.templates["camtel"] = _gabarit()
pio.templates.default = "camtel"


MOIS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août",
           "sept.", "oct.", "nov.", "déc."]


def axe_mois(fig, dates, pas=1):
    """Graduations mensuelles en francais : Plotly n'embarque pas la locale fr."""
    import pandas as pd
    d = pd.to_datetime(pd.Series(list(dates)))
    debuts = pd.date_range(d.min().normalize().replace(day=1),
                           d.max(), freq="MS")[::pas]
    fig.update_xaxes(
        tickmode="array", tickvals=list(debuts),
        ticktext=[f"{MOIS_FR[t.month - 1]} {t.year % 100:02d}" for t in debuts])
    return fig


def etiquette_mois(periode):
    """'2025-09' -> 'sept. 25'"""
    a, m = str(periode).split("-")[:2]
    return f"{MOIS_FR[int(m) - 1]} {a[2:]}"


def afficher(fig, hauteur=340, **kw):
    """Affiche une figure avec les reglages communs."""
    fig.update_layout(height=hauteur, template="camtel", **kw)
    st.plotly_chart(fig, width="stretch", config={
        "displayModeBar": False, "scrollZoom": False, "responsive": True})


# ----------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --encre:#0E1C2B; --papier:#F6F7F5; --bleu:#1668A6; --ciel:#86B8D8;
  --ambre:#DFA23C; --brique:#B23A35; --vert:#3D8A5C; --trait:#DCE2E5; --gris:#5E7183;
}

/* chiffres alignes partout : c'est un outil de lecture de nombres */
html, body, [class*="st-"], .stMarkdown, table { font-variant-numeric: tabular-nums; }

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1380px; }

/* ---------- en-tete de page ---------- */
.entete { margin-bottom: 1.6rem; }
.entete h1 {
  font-family: Archivo, sans-serif; font-weight: 700; font-size: 1.85rem;
  letter-spacing: -0.022em; color: var(--encre); margin: 0 0 .3rem 0; line-height: 1.15;
}
.entete p { color: var(--gris); font-size: .95rem; margin: 0; max-width: 68ch; }

/* ---------- titre de section : filet qui court jusqu'au bord ---------- */
.section { display:flex; align-items:center; gap:.85rem; margin: 2.1rem 0 .9rem 0; }
.section:first-of-type { margin-top: .6rem; }
.section h2 {
  font-family: Archivo, sans-serif; font-weight: 600; font-size: 1.05rem;
  color: var(--encre); margin:0; white-space: nowrap; letter-spacing:-.01em;
}
.section .filet { flex:1; height:1px; background: var(--trait); }
.section .compte { color: var(--gris); font-size:.82rem; white-space: nowrap; }

/* ---------- grand livre : l'equation de flux rendue lisible ---------- */
.grand-livre {
  display:flex; align-items:stretch; background:#fff;
  border:1px solid var(--trait); border-radius:.5rem; overflow:hidden;
}
.gl-poste { flex:1; padding: 1.05rem 1.25rem; border-left:1px solid var(--trait); }
.gl-poste:first-child { border-left:none; }
.gl-etiquette { display:block; font-size:.78rem; color:var(--gris); margin-bottom:.3rem; }
.gl-valeur {
  display:block; font-family:Archivo, sans-serif; font-weight:650;
  font-size:1.62rem; line-height:1.1; color:var(--encre); letter-spacing:-.02em;
}
.gl-note { display:block; font-size:.76rem; color:var(--gris); margin-top:.28rem; }
.gl-op {
  display:flex; align-items:center; justify-content:center; width:34px;
  font-family:Archivo, sans-serif; font-size:1.15rem; color:var(--gris);
  background:#FAFBFB; border-left:1px solid var(--trait);
}
.gl-entree .gl-valeur { color: var(--vert); }
.gl-sortie .gl-valeur { color: var(--brique); }
.gl-resultat { background:#FAFBFC; }

/* ---------- tuiles : filet d'accent a gauche, pas d'ombre ---------- */
.tuile {
  background:#fff; border:1px solid var(--trait); border-left:3px solid var(--bleu);
  border-radius:.4rem; padding:.9rem 1rem; height:100%;
}
.tuile .t-lab { font-size:.78rem; color:var(--gris); display:block; margin-bottom:.28rem; }
.tuile .t-val {
  font-family:Archivo, sans-serif; font-weight:650; font-size:1.4rem;
  color:var(--encre); letter-spacing:-.02em; display:block; line-height:1.15;
}
.tuile .t-sub { font-size:.76rem; color:var(--gris); display:block; margin-top:.25rem; }
.tuile.ambre { border-left-color: var(--ambre); }
.tuile.brique { border-left-color: var(--brique); }
.tuile.vert { border-left-color: var(--vert); }

/* ---------- note de methode ---------- */
.note {
  border:1px solid var(--trait); border-left:3px solid var(--ambre); background:#FFFDF8;
  border-radius:.4rem; padding:.85rem 1.05rem; font-size:.88rem; color:#4A3B22;
  line-height:1.5;
}
.note.alerte { border-left-color: var(--brique); background:#FFF9F8; color:#5A2B28; }
.note b { font-weight:600; }

/* ---------- barre horizontale de proportion ---------- */
.jauge { height:6px; background:#EAEFF2; border-radius:3px; overflow:hidden; margin-top:.5rem; }
.jauge span { display:block; height:100%; background:var(--bleu); }

/* ---------- chrome Streamlit : on retire ce qui n'appartient pas au produit ---------- */
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display:none !important; }
[data-testid="stHeader"] { background:transparent; height:0; }

/* ---------- barre laterale : la radio devient une vraie navigation ---------- */
[data-testid="stSidebar"] [role="radiogroup"] { gap:.12rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding:.46rem .7rem; border-radius:.35rem; width:100%;
  transition: background .12s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:#18293C; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background:#1D3551;
}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child { display:none; }
[data-testid="stSidebar"] [role="radiogroup"] label p {
  font-size:.92rem; font-weight:500; color:#9FB3C4;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
  color:#FFFFFF;
}

/* ---------- barre laterale ---------- */
[data-testid="stSidebar"] { border-right: 1px solid #22364B; }
.marque { padding: .35rem 0 1.15rem 0; border-bottom:1px solid #22364B; margin-bottom:1.1rem; }
.marque .nom {
  font-family:Archivo, sans-serif; font-weight:700; font-size:1.28rem; color:#FFFFFF;
  letter-spacing:-.02em; line-height:1.1;
}
.marque .sous { font-size:.8rem; color:#7E94A6; margin-top:.28rem; line-height:1.35; }
.pied { margin-top:1.4rem; padding-top:1rem; border-top:1px solid #22364B;
        font-size:.76rem; color:#7E94A6; line-height:1.5; }

/* ---------- tableaux ---------- */
[data-testid="stDataFrame"] { border-radius:.4rem; }

/* ---------- onglets ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom:1px solid var(--trait); }
.stTabs [data-baseweb="tab"] { padding: .4rem 0; font-weight:500; }

/* ---------- accessibilite ---------- */
:focus-visible { outline:2px solid var(--bleu); outline-offset:2px; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
@media (max-width: 900px) {
  .grand-livre { flex-direction:column; }
  .gl-poste { border-left:none; border-top:1px solid var(--trait); }
  .gl-op { width:100%; height:26px; border-left:none; border-top:1px solid var(--trait); }
}
</style>
"""


def appliquer():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- composants
def nb(x, unite=""):
    """Format francais : espace fine comme separateur de milliers."""
    s = f"{x:,.0f}".replace(",", "\u202f")
    return f"{s}\u202f{unite}".strip()


def dec(x, n=2, unite=""):
    """Format francais : virgule decimale."""
    s = f"{x:,.{n}f}".replace(",", "\u202f").replace(".", ",")
    return f"{s}\u202f{unite}".strip()


def pct(x, n=1):
    return dec(x * 100, n) + "\u202f%"


def fcfa(x):
    if abs(x) >= 1e9:
        return dec(x / 1e9, 2, "Md FCFA")
    if abs(x) >= 1e6:
        return dec(x / 1e6, 1, "M FCFA")
    return nb(x, "FCFA")


def entete(titre, description):
    st.markdown(f"<div class='entete'><h1>{titre}</h1><p>{description}</p></div>",
                unsafe_allow_html=True)


def section(titre, compte=""):
    droite = f"<span class='compte'>{compte}</span>" if compte else ""
    st.markdown(f"<div class='section'><h2>{titre}</h2><span class='filet'></span>"
                f"{droite}</div>", unsafe_allow_html=True)


def tuile(etiquette, valeur, sous="", ton="", jauge=None):
    j = (f"<div class='jauge'><span style='width:{max(0, min(100, jauge)):.0f}%'></span></div>"
         if jauge is not None else "")
    st.markdown(
        f"<div class='tuile {ton}'><span class='t-lab'>{etiquette}</span>"
        f"<span class='t-val'>{valeur}</span>"
        f"{f'<span class=t-sub>{sous}</span>' if sous else ''}{j}</div>",
        unsafe_allow_html=True)


def grand_livre(postes):
    """postes : liste de (etiquette, valeur, note, classe, operateur_avant)."""
    html = ["<div class='grand-livre'>"]
    for etiquette, valeur, note, classe, op in postes:
        if op:
            html.append(f"<div class='gl-op'>{op}</div>")
        html.append(
            f"<div class='gl-poste {classe}'><span class='gl-etiquette'>{etiquette}</span>"
            f"<span class='gl-valeur'>{valeur}</span>"
            f"{f'<span class=gl-note>{note}</span>' if note else ''}</div>")
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def note(texte, alerte=False):
    st.markdown(f"<div class='note{' alerte' if alerte else ''}'>{texte}</div>",
                unsafe_allow_html=True)
