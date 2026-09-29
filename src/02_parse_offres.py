"""
02 - Extraction du classeur des offres vers un format long (tidy).
41 feuilles mensuelles, structures heterogenes -> une table (date, offre, activations).
Sorties : outputs/offres_long.csv, outputs/offres_prix.csv, outputs/audit_offres.md
"""
import re
import unicodedata
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# --- mois indiques par le nom de la feuille (l'ordre du classeur est chronologique) ---
MOIS = {
    "Offers April": (2022, 4), "Offers May": (2022, 5), "Offers June": (2022, 6),
    "Offers July": (2022, 7), "Offers August": (2022, 8), "September": (2022, 9),
    "October": (2022, 10), "November": (2022, 11), "December": (2022, 12),
    "Jan23": (2023, 1), "Feb23": (2023, 2), "Mar23": (2023, 3), "Apr23": (2023, 4),
    "May23": (2023, 5), "Jun23": (2023, 6), "Jul23": (2023, 7), "Aug23": (2023, 8),
    "Sept23": (2023, 9), "Oct23": (2023, 10), "Nov23": (2023, 11), "Dec23": (2023, 12),
    "Janv24": (2024, 1), "Feb24": (2024, 2), "Mar24": (2024, 3), "Avr24": (2024, 4),
    "Mai24": (2024, 5), "Jun24": (2024, 6), "Jul24": (2024, 7), "Aou24 ": (2024, 8),
    "Sept24": (2024, 9), "Oct24": (2024, 10), "Nov24": (2024, 11), "Dec24": (2024, 12),
    "JanV25": (2025, 1), "Feb25": (2025, 2), "Mars25": (2025, 3), "Avr25": (2025, 4),
    "Mai25": (2025, 5), "Juin25": (2025, 6), "Juillet25": (2025, 7), "Août25": (2025, 8),
}

# --- table de normalisation des libelles d'offres ---
ALIAS = {
    "spot": "Blue mo S", "spot / blue mo s": "Blue mo S",
    "cool": "Blue mo M", "cool / blue mo m": "Blue mo M",
    "blue booster": "Blue mo L", "blue booster / blue mo l": "Blue mo L",
    "kolo": "Blue mo XL", "kolo / blue mo xl": "Blue mo XL",
    "swim": "Blue mo XXL", "swim / blue mo xxl": "Blue mo XXL",
    "fap": "FAP",
}


def norm_label(s):
    """Normalise un libelle d'offre : casse, accents, espaces, alias historiques."""
    if s is None:
        return None
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s).strip()
    key = s.lower()
    if key in ALIAS:
        return ALIAS[key]
    # "X / Y" : on garde la denomination recente (partie droite)
    if " / " in s:
        return s.split(" / ")[-1].strip()
    return s


def to_num(v):
    """Convertit une cellule en nombre. '-' et '' representent zero/absence."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    if s in {"", "-", "--"}:
        return np.nan
    if s.startswith("="):          # formule non resolue
        return np.nan
    s = s.replace("\u202f", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


wb = load_workbook(ROOT / "data" / "offres.xlsx", read_only=True, data_only=True)

records, prix_records, journal = [], [], []

for sheet in wb.sheetnames:
    if sheet not in MOIS:
        journal.append(f"feuille ignoree (mois inconnu) : {sheet!r}")
        continue
    annee, mois = MOIS[sheet]
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        continue

    header = rows[0]
    # colonne du libelle : celle qui contient 'Number of activations'
    col_lab = 0
    for j, v in enumerate(header[:3]):
        if isinstance(v, str) and "number of activations" in v.lower():
            col_lab = j
            break

    # colonnes de dates : entetes datetime, ou reconstruites a partir de la position
    date_cols = {}
    for j in range(col_lab + 1, len(header)):
        v = header[j]
        if isinstance(v, (dt.datetime, dt.date)):
            d = v.date() if isinstance(v, dt.datetime) else v
            # correction des mois saisis de travers (ex. octobre saisi en aout)
            if (d.year, d.month) != (annee, mois):
                try:
                    d = dt.date(annee, mois, d.day)
                    journal.append(f"{sheet}: entete de date corrigee -> {d}")
                except ValueError:
                    continue
            date_cols[j] = d
        elif isinstance(v, str) and v.strip().upper() == "TOTAL":
            break

    # entetes en formule (=B1+1) : on reconstruit par decalage depuis la premiere date
    if date_cols:
        j0, d0 = min(date_cols.items())
        for j in range(col_lab + 1, len(header)):
            v = header[j]
            if j in date_cols:
                continue
            if isinstance(v, str) and v.strip().startswith("="):
                cand = d0 + dt.timedelta(days=j - j0)
                if cand.month == mois:
                    date_cols[j] = cand
                    journal.append(f"{sheet}: entete formule reconstruite -> {cand}")
            elif isinstance(v, str) and v.strip().upper() == "TOTAL":
                break

    # colonnes prix / montant
    col_prix = None
    for j, v in enumerate(header):
        if isinstance(v, str) and "prix" in v.lower():
            col_prix = j

    bloc = "offres"
    for r in rows[1:]:
        if r is None or len(r) <= col_lab:
            continue
        lab_raw = r[col_lab]
        if lab_raw is None:
            # separateur ou ligne vide ; detecte le passage au bloc SIM
            joined = " ".join(str(x) for x in r[:3] if x is not None).lower()
            if "sim" in joined:
                bloc = "sim"
            continue
        lab = str(lab_raw).strip()
        if lab.upper() == "TOTAL":
            continue
        if "sim card" in lab.lower():
            bloc = "sim"
            continue
        if "number of activations" in lab.lower():
            continue

        offre = norm_label(lab)
        if not offre:
            continue

        for j, d in date_cols.items():
            if j < len(r):
                val = to_num(r[j])
                if not np.isnan(val):
                    records.append((d, offre, bloc, val))

        if col_prix is not None and col_prix < len(r):
            p = to_num(r[col_prix])
            if not np.isnan(p) and p > 0:
                prix_records.append((annee, mois, offre, p))

df = pd.DataFrame(records, columns=["date", "offre", "bloc", "activations"])
df["date"] = pd.to_datetime(df["date"])
df = df.groupby(["date", "offre", "bloc"], as_index=False)["activations"].sum()
df = df.sort_values(["offre", "date"]).reset_index(drop=True)
df.to_csv(OUT / "offres_long.csv", index=False)

prix = pd.DataFrame(prix_records, columns=["annee", "mois", "offre", "prix"])
prix = prix.drop_duplicates().sort_values(["offre", "annee", "mois"])
prix.to_csv(OUT / "offres_prix.csv", index=False)

# ---------------- rapport ----------------
bo = df[df["offre"].isin(["Blue One S", "Blue One M", "Blue One L"])]
lignes = [
    "# Audit du classeur des offres", "",
    f"- Feuilles traitees : {len(MOIS)}",
    f"- Observations (date x offre) : {len(df)}",
    f"- Offres distinctes apres normalisation : {df['offre'].nunique()}",
    f"- Periode : {df['date'].min().date()} -> {df['date'].max().date()}",
    "",
    "## Forfaits Blue One (pont avec le fichier churn)",
    f"- Periode couverte : {bo['date'].min().date()} -> {bo['date'].max().date()}",
    f"- Jours observes : {bo['date'].nunique()}",
]
for o in ["Blue One S", "Blue One M", "Blue One L"]:
    s = bo[bo["offre"] == o]
    if len(s):
        lignes.append(f"- {o} : {len(s)} jours, moyenne {s['activations'].mean():,.0f} act./jour")

lignes += ["", "## Prix releves (FCFA)"]
for o in ["Blue One S", "Blue One M", "Blue One L", "Blue mo S", "Blue mo M", "Blue mo L"]:
    p = prix[prix["offre"] == o]["prix"].unique()
    if len(p):
        lignes.append(f"- {o} : {sorted(p)}")

lignes += ["", "## Corrections appliquees"] + [f"- {x}" for x in journal[:25]]
if len(journal) > 25:
    lignes.append(f"- ... et {len(journal) - 25} autres")

(OUT / "audit_offres.md").write_text("\n".join(lignes), encoding="utf-8")
print("\n".join(lignes))
