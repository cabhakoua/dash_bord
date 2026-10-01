# Donnees sources

Ce dossier est vide dans le depot : les fichiers bruts ne sont pas versionnes.

Le fichier `churn_database2.csv` contient les MSISDN des abonnes, c'est-a-dire des
numeros de telephone en clair. Ces donnees appartiennent a CAMTEL et ne doivent pas
etre publiees.

Pour executer la chaine complete en local, deposer ici :

- `churn_database2.csv`
- `offres.xlsx` (le classeur `Statistique_offre31082025.xlsx`, renomme)

puis lancer `bash run_all.sh`.

L'application `app.py` ne lit jamais ce dossier. Elle se sert uniquement de `outputs/`
et `figures/`, qui ne contiennent aucun identifiant nominatif : `MSISDN` est supprime
des l'etape de pretraitement, et `CUSTOMER_ID` est un simple numero de ligne (1 a 13 937)
sans correspondance avec les identifiants du systeme de CAMTEL.
