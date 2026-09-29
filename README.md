# CAMTEL : prédiction du churn et de la demande en offres

Chaîne complète : audit, prétraitement, modélisation, modèle de flux, tableau de bord.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt   # chaine complete
# pip install -r requirements.txt     # application seule
```

Placer les deux fichiers source dans `data/` :

- `data/churn_database2.csv`
- `data/offres.xlsx` (le classeur `Statistique_offre31082025.xlsx` renommé)

## Exécution

```bash
bash run_all.sh          # ou, sous Windows, les commandes une par une
streamlit run app.py
```

## Organisation

| Script | Rôle |
|---|---|
| `src/01_audit_churn.py` | Audit du fichier abonnés : constantes, fuite, censure, aberrants |
| `src/02_parse_offres.py` | Extraction des 41 feuilles vers une table longue, normalisation des libellés |
| `src/03_preprocess_churn.py` | Nettoyage, retrait de la fuite et de la censure, variables dérivées |
| `src/04_models_churn.py` | Trois classifieurs + ablation + Kaplan-Meier / Cox |
| `src/05_preprocess_offres.py` | Trous, filtre de Hampel, choix du régime stable |
| `src/06_models_demande.py` | SARIMA, Prophet, XGBoost + référence naïve, prévision à 90 jours |
| `src/07_modele_flux.py` | Couplage des deux modèles, scénarios, revenu à risque |
| `src/08_bilan.py` | Tableaux de synthèse prêts à insérer dans le mémoire |
| `app.py` | Tableau de bord Streamlit (7 pages) |

Les sorties intermédiaires vont dans `outputs/`, les figures dans `figures/`.

## Points de méthode

**Fuite d'information.** L'étiquette de churn est définie par une règle d'inactivité de
60 jours calculée à partir de `PURGE_TIME_ATSGSN`. Cette règle reproduit l'étiquette dans
99,96 % des cas. Les variables dérivées de la date de purge sont donc exclues de
l'apprentissage : les conserver donne une AUC proche de 1 sans rien prédire.

**Censure à droite.** Un abonné créé moins de 60 jours avant la fin de la fenêtre ne peut
pas avoir été étiqueté churner. Ses 1 311 observations sont retirées de la classification
et conservées comme observations censurées pour l'analyse de survie.

**Représentativité.** L'extrait contient 44 % de churners, ce qui est incompatible avec la
base observée, stable à +0,26 % par mois. Le modèle fournit donc un classement fiable du
risque, mais son niveau absolu doit être recalé. Le modèle de flux le fait explicitement,
par scénarios.

**Le paramètre alpha.** Le classeur ne distingue pas souscription et renouvellement. Sur
une base stable, l'identité de stock impose alpha ≈ p* + g : alpha est déduit, pas
postulé. À confirmer auprès de CAMTEL.

## Compatibilité scikit-learn

`outputs/modele_churn.joblib` est un pickle scikit-learn : il n'est pas portable entre
versions majeures. Un modèle sérialisé avec scikit-learn 1.8 relu avec une version
antérieure lève `AttributeError: 'LogisticRegression' object has no attribute
'multi_class'`.

L'application gère le cas : elle valide le pickle par une prédiction à blanc et, en cas
d'échec, reconstruit le modèle depuis `outputs/modele_churn.json` et
`outputs/churn_clean.csv`. Le réentraînement prend quelques secondes et donne un modèle
identique (mêmes hyperparamètres, même graine).

Pour éviter le détour, régénérer le modèle dans votre environnement :

```bash
python src/04_models_churn.py
```

## Déploiement

Voir `DEPLOIEMENT.md` : publication sur GitHub, puis mise en ligne sur Streamlit
Community Cloud. Point d'attention principal, le fichier `data/churn_database2.csv`
contient les numéros de téléphone des abonnés et reste exclu du dépôt par `.gitignore`.
