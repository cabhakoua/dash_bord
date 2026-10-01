# Bilan general

## 1. Qualite des donnees

- Jeu churn : 13937 lignes, 24 colonnes, aucune valeur manquante, aucun doublon.
- 5 colonnes constantes et 2 quasi-constantes retirees.
- Fuite d'information : la regle « purge anterieure de plus de 60 jours a 2025-07-29 » reproduit l'etiquette dans 99.9641% des cas (5 exceptions). Les variables derivees de la purge sont donc exclues de l'apprentissage.
- Censure a droite : 1311 abonnes crees apres 2025-05-30 ne pouvaient pas cumuler 60 jours d'inactivite (0 churner parmi eux). Ils sont retires de la classification et conserves comme observations censurees pour l'analyse de survie.
- Identite verifiee a 100% : TOTAL_CHARGE = MONTHLY_CHARGE x RECHARGE_FREQUENCY (colonne redondante).
- Jeu offres : 41 feuilles harmonisees, 123 offres apres normalisation des libelles (SPOT/Blue mo S, COOL/Blue mo M, BLUE Booster/Blue mo L, KOLO/Blue mo XL, SWIM/Blue mo XXL fusionnes).

## 2. Modeles de churn

Jeu de test : 25 % des observations, stratifie. Seuil de decision 0,5.

### Pouvoir discriminant

| Modele                | AUC (VC 5 blocs)   |   AUC (test) |   AUC-PR (test) |   Brier |
|:----------------------|:-------------------|-------------:|----------------:|--------:|
| Regression logistique | 0.9003 ± 0.0081    |       0.8914 |          0.8934 |  0.1242 |
| Foret aleatoire       | 0.8924 ± 0.0103    |       0.8896 |          0.8924 |  0.1256 |
| XGBoost               | 0.8887 ± 0.0110    |       0.8875 |          0.8904 |  0.1273 |

### Metriques de classification au seuil 0,5

| Modele                |   Exactitude |   Precision (churn) |   Rappel (churn) |   F1 (churn) |   Precision (fidele) |   Rappel (fidele) | VN / FP / FN / VP       |
|:----------------------|-------------:|--------------------:|-----------------:|-------------:|---------------------:|------------------:|:------------------------|
| Regression logistique |       0.7757 |              0.7528 |           0.8018 |       0.7765 |               0.8004 |            0.7511 | 1219 / 404 / 304 / 1230 |
| Foret aleatoire       |       0.7697 |              0.7565 |           0.7757 |       0.766  |               0.7828 |            0.764  | 1240 / 383 / 344 / 1190 |
| XGBoost               |       0.7669 |              0.7554 |           0.7692 |       0.7623 |               0.7781 |            0.7646 | 1241 / 382 / 354 / 1180 |

(VN / FP / FN / VP : vrais negatifs, faux positifs, faux negatifs, vrais positifs.)

### Au seuil optimisant le F1

| Modele                |   Seuil |   Exactitude |   Precision |   Rappel |     F1 |
|:----------------------|--------:|-------------:|------------:|---------:|-------:|
| Regression logistique |  0.4193 |       0.7802 |      0.7047 |   0.9426 | 0.8065 |
| Foret aleatoire       |  0.3772 |       0.7735 |      0.6938 |   0.9557 | 0.8039 |
| XGBoost               |  0.2487 |       0.7593 |      0.672  |   0.9857 | 0.7992 |

### Etude d'ablation (AUC du meilleur modele, variable retiree)

| Variante                |    AUC |
|:------------------------|-------:|
| sans RECHARGE_FREQUENCY | 0.5686 |
| sans intensite_recharge | 0.8865 |
| sans TENURE_MONTHS      | 0.8862 |
| sans PLAN_NAME          | 0.8859 |

### Importance des variables (permutation, chute d'AUC)

| variable           |   importance |   ecart_type |
|:-------------------|-------------:|-------------:|
| RECHARGE_FREQUENCY |       0.0614 |       0.0031 |
| revenu_cumule      |       0.0481 |       0.0035 |
| intensite_recharge |       0.0439 |       0.0042 |
| MONTHLY_CHARGE     |       0.0379 |       0.0042 |
| jamais_recharge    |       0.0086 |       0.0022 |
| PLAN_NAME          |       0.0079 |       0.002  |
| mois_creation      |       0.0044 |       0.0032 |
| TENURE_MONTHS      |       0.0021 |       0.0025 |

### Analyse de survie

- Indice de concordance du modele de Cox : 0.7543
- Rapports de risque (reference : Blue One L) :

| covariate            |    coef |   exp(coef) |      p |
|:---------------------|--------:|------------:|-------:|
| AGE                  | -0.0139 |      0.9862 | 0.0002 |
| RECHARGE_FREQUENCY   | -0.7017 |      0.4957 | 0      |
| PLAN_NAME_Blue One M |  0.4525 |      1.5723 | 0      |
| PLAN_NAME_Blue One S |  0.0247 |      1.025  | 0.7775 |
| GENDER_Male          | -0.0046 |      0.9954 | 0.8416 |

- Duree mediane de survie : Blue One L = 167 jours, Blue One M = 135 jours, Blue One S = non atteinte sur la fenetre

## 3. Modeles de demande

Test hors echantillon : 60 derniers jours. Validation par origine glissante : 3 blocs de 30 jours.

| Forfait    | Modele                         |    MAE |   RMSE |   MAPE_% |   sMAPE_% |   biais |   MAPE_origine_glissante_% | Retenu   |
|:-----------|:-------------------------------|-------:|-------:|---------:|----------:|--------:|---------------------------:|:---------|
| Blue One S | Naif saisonnier (reference)    |   18.5 |   22.7 |    32.28 |     37.96 |   -14   |                      31.08 |          |
| Blue One S | SARIMA (2,1,2)(1,1,1)7         |   17.4 |   21.7 |    29.71 |     34.73 |   -13.3 |                      21.11 | oui      |
| Blue One S | Prophet                        |   19.3 |   24.6 |    31.25 |     39.54 |   -18.2 |                      23.08 |          |
| Blue One S | XGBoost (retards + calendrier) |   17.1 |   20.9 |    30.17 |     34.41 |   -11.9 |                      23.52 |          |
| Blue One M | Naif saisonnier (reference)    | 1574.9 | 1835.7 |    21.66 |     22.51 |  -684   |                      26.83 |          |
| Blue One M | SARIMA (2,1,2)(1,1,1)7         |  774.8 | 1032.8 |    11.7  |     10.59 |   472.2 |                      15.94 |          |
| Blue One M | Prophet                        |  921.1 | 1206   |    13.81 |     12.43 |   510.3 |                      14.87 |          |
| Blue One M | XGBoost (retards + calendrier) |  565.7 |  807.5 |     7.82 |      7.88 |  -251   |                      13.61 | oui      |
| Blue One L | Naif saisonnier (reference)    |  128.4 |  149.9 |    24.63 |     27.18 |   -69.2 |                      26.62 |          |
| Blue One L | SARIMA (2,1,2)(1,1,1)7         |   89.9 |  110.2 |    19.14 |     16.7  |    72.3 |                      23.57 |          |
| Blue One L | Prophet                        |   81.3 |   95   |    16.11 |     15.39 |     9.9 |                      17.26 |          |
| Blue One L | XGBoost (retards + calendrier) |   52.6 |   67.2 |    10.1  |     10.84 |   -41   |                      14.04 | oui      |

## 4. Modele de flux

Scenario central (taux de depart de reference p* = 5 % par mois, ecarts entre forfaits issus du modele de churn) :

|   alpha | mois    | forfait    |   taux_depart_mensuel |   base_debut |   activations_prevues |   entrees |   departs_attendus |   base_fin |   variation_% |   revenu_a_risque_fcfa |   revenu_base_fcfa |
|--------:|:--------|:-----------|----------------------:|-------------:|----------------------:|----------:|-------------------:|-----------:|--------------:|-----------------------:|-------------------:|
|  0.0526 | 2025-09 | Blue One S |                0.0149 |         1331 |                  1468 |        77 |                 20 |       1389 |          4.31 |                  39715 |            2662667 |
|  0.0526 | 2025-09 | Blue One M |                0.0579 |       231113 |                209370 |     11014 |              13391 |     228736 |         -1.03 |               40174430 |          693339000 |
|  0.0526 | 2025-09 | Blue One L |                0.0771 |        15402 |                 15927 |       838 |               1188 |      15051 |         -2.27 |                5940493 |           77008333 |
|  0.0526 | 2025-10 | Blue One S |                0.0149 |         1389 |                  1484 |        78 |                 21 |       1446 |          4.13 |                  41426 |            2777373 |
|  0.0526 | 2025-10 | Blue One M |                0.0579 |       228736 |                211585 |     11131 |              13254 |     226613 |         -0.93 |               39761195 |          686207308 |
|  0.0526 | 2025-10 | Blue One L |                0.0771 |        15051 |                 16219 |       853 |               1161 |      14744 |         -2.05 |                5805407 |           75257174 |

### Sensibilite

| scenario                      |   alpha |   taux_depart_mensuel | mois    |   base_fin |   variation_% |   revenu_a_risque_fcfa |
|:------------------------------|--------:|----------------------:|:--------|-----------:|--------------:|-----------------------:|
| prudent (p*=2%)               |  0.0226 |                0.0232 | 2025-09 |     230490 |         -0.27 |               16069772 |
| prudent (p*=2%)               |  0.0226 |                0.0232 | 2025-10 |     229931 |         -0.24 |               16026423 |
| central (p*=5%)               |  0.0526 |                0.0579 | 2025-09 |     228736 |         -1.03 |               40174430 |
| central (p*=5%)               |  0.0526 |                0.0579 | 2025-10 |     226613 |         -0.93 |               39761195 |
| haut (p*=10%)                 |  0.1026 |                0.1159 | 2025-09 |     225813 |         -2.29 |               80348860 |
| haut (p*=10%)                 |  0.1026 |                0.1159 | 2025-10 |     221354 |         -1.97 |               78506182 |
| echantillon brut (non recale) |  0.2296 |                0.2631 | 2025-09 |     218389 |         -5.51 |              182383301 |
| echantillon brut (non recale) |  0.2296 |                0.2631 | 2025-10 |     209521 |         -4.06 |              172342134 |

## 5. Limites

- L'echantillon churn (44 % de churners) n'est pas un tirage aleatoire de la base : le modele fournit un classement fiable, mais le niveau absolu du risque doit etre recale sur une reference externe.
- Le mix des forfaits differe fortement entre l'echantillon churn (68 % de M) et les activations reelles de 2025 (93 % de M).
- La part des activations correspondant a de nouvelles souscriptions n'est pas observable dans le classeur ; elle est deduite de l'identite de stock, sous l'hypothese d'un abonne generant une activation par mois. Une confirmation aupres de CAMTEL leverait cette incertitude.
- La fenetre du fichier churn (janvier-juillet 2025) est courte : sept mois d'historique ne permettent pas d'observer de saisonnalite annuelle du churn.