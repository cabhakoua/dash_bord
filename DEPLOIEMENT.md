# Déploiement

## 1. Avant de publier : ce qui ne doit pas partir sur GitHub

`data/churn_database2.csv` contient la colonne `MSISDN`, soit les numéros de téléphone
des abonnés en clair, au format `237XXXXXXXXX`. Ce sont des données personnelles
appartenant à CAMTEL. Elles ne doivent figurer dans aucun dépôt, même privé, sans accord
écrit.

Le `.gitignore` fourni exclut déjà tout le contenu de `data/`.

Le reste est publiable. `MSISDN` est supprimé dès l'étape de prétraitement, et
`CUSTOMER_ID` n'est qu'un numéro de ligne de 1 à 13 937, sans correspondance avec les
identifiants du système de CAMTEL. Vérification faite : aucun fichier de `outputs/` ni de
`figures/` ne contient de numéro d'abonné.

L'application ne lit jamais `data/`. Elle se sert uniquement de `outputs/` et `figures/`,
qui doivent donc être versionnés.

## 2. Dépôt public ou privé ?

| | Public | Privé |
|---|---|---|
| Visible sur votre profil GitHub | oui | non |
| Utilisable comme référence professionnelle | oui | sur invitation |
| Déploiement Streamlit Cloud | oui | oui |
| Permissions demandées par Streamlit | lecture des dépôts publics | portée `repo` complète |

Les deux fonctionnent. Pour un travail de mémoire adossé à des données d'entreprise, le
dépôt privé est le choix par défaut : il évite d'avoir à justifier auprès de CAMTEL ce
qui a été publié. Le dépôt public a l'avantage de servir de référence vérifiable pour une
candidature.

Un compromis souvent retenu : dépôt privé pendant la soutenance, bascule en public
ensuite si CAMTEL ne s'y oppose pas. Le passage de privé à public se fait en deux clics,
l'inverse ne répare pas un historique déjà publié.

## 3. Publier sur GitHub

Installer Git depuis git-scm.com, puis, dans le dossier du projet :

```bash
git init -b main
git add .
git status          # verifier qu'aucun fichier de data/ n'apparait
git commit -m "Prediction du churn et de la demande en offres CAMTEL"
```

Le `git status` est le point de contrôle. Si `data/churn_database2.csv` apparaît dans la
liste, arrêter là et vérifier que `.gitignore` est bien à la racine.

Créer ensuite le dépôt sur github.com (bouton **New repository**), sans cocher
« Add a README file » puisque le projet en a déjà un, puis :

```bash
git remote add origin https://github.com/<votre-compte>/camtel-churn.git
git push -u origin main
```

GitHub demandera un jeton d'accès, pas votre mot de passe : le créer dans
Settings, Developer settings, Personal access tokens, avec la portée `repo`.

## 4. Structure attendue par Streamlit Cloud

Community Cloud initialise l'application depuis la **racine du dépôt**, même si le
fichier d'entrée est ailleurs. Le contenu du projet doit donc se trouver à la racine, pas
dans un sous-dossier `camtel/`. La disposition à obtenir :

```
app.py
theme.py
requirements.txt
.streamlit/config.toml
outputs/
figures/
src/
data/README.md
```

Un seul `.streamlit/config.toml` est reconnu, celui de la racine.

## 5. Deux fichiers de dépendances

`requirements.txt` ne liste que ce dont l'application a besoin : Streamlit, pandas,
numpy, plotly, scikit-learn, xgboost, joblib. C'est ce fichier que Community Cloud
installe à chaque démarrage.

`requirements-dev.txt` ajoute prophet, statsmodels, lifelines, matplotlib et openpyxl,
nécessaires aux scripts d'entraînement mais pas à l'application. Les installer sur le
cloud allongerait le démarrage de plusieurs minutes pour rien : prophet compile cmdstan.

En local, pour exécuter la chaîne complète :

```bash
pip install -r requirements-dev.txt
```

## 6. Déployer

1. Aller sur share.streamlit.io et se connecter avec le compte GitHub.
2. **Create app**, puis **Deploy a public app from GitHub**.
3. Renseigner le dépôt, la branche `main`, et `app.py` comme fichier principal.
4. Dans **Advanced settings**, choisir Python 3.11 ou 3.12.
5. **Deploy**.

Le premier démarrage prend quelques minutes, le temps d'installer xgboost et
scikit-learn. Les suivants sont rapides. Chaque `git push` redéploie automatiquement.

Pour un dépôt privé, Streamlit demandera une autorisation supplémentaire au premier
déploiement : la portée `repo` de GitHub. C'est attendu, l'accès créé est une clé de
déploiement en lecture seule.

## 7. Ce qui peut échouer

**Dépassement de mémoire.** L'offre gratuite est limitée. L'application charge une
quinzaine de CSV, dont `scores_churn.csv` (879 Ko) et `churn_clean.csv` (877 Ko) : cela
reste très en deçà du plafond. Si un message « over its resource limits » apparaît
malgré tout, la cause la plus probable est un cache non libéré. Redémarrer l'application
depuis le tableau de bord Streamlit.

**Incompatibilité de version de scikit-learn.** `outputs/modele_churn.joblib` a été
sérialisé avec une version donnée de scikit-learn et n'est pas portable. Community Cloud
installera probablement une autre version. L'application gère le cas : elle valide le
pickle par une prédiction à blanc et, en cas d'échec, reconstruit le modèle depuis
`outputs/modele_churn.json` et `outputs/churn_clean.csv`. Le réentraînement prend moins
d'une seconde. Un bandeau le signale dans la page de scoring.

**Fichier introuvable.** Si l'application démarre puis affiche une erreur sur un CSV,
c'est que `outputs/` ou `figures/` n'a pas été versionné. Vérifier que `.gitignore`
n'exclut pas ces dossiers, puis `git add outputs figures -f` si nécessaire.

**Accents dans les chemins.** Sous Windows, le dossier de travail actuel contient un
espace (`memoire cedric`). Sans conséquence pour Git, mais éviter les accents dans le nom
du dépôt.

## 8. Après le déploiement

L'adresse obtenue a la forme `https://<nom>.streamlit.app`. Elle se personnalise dans les
paramètres de l'application.

Les applications gratuites se mettent en veille après une période d'inactivité et
redémarrent à la première visite, avec quelques secondes d'attente. Avant une soutenance,
ouvrir l'application une fois pour la réveiller. Et garder une copie locale prête à
lancer : une salle sans réseau fiable est un risque plus concret qu'une panne de
Streamlit.
