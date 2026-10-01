# Audit du jeu de donnees churn

- Lignes : 13937 | Colonnes : 24
- Doublons complets : 0 | doublons MSISDN : 0
- Valeurs manquantes : 0
- Taux de churn : 44.02%

## Colonnes sans information
Constantes (5) : PURGEDONSGSN, PS_UTRAN, PS_GERAN, PS_E-UTRAN, EPSODBPOS
Quasi-constantes : TGPPAMBRMAXUL, TGPPAMBRMAXDL

## Fuite d'information
Regle 'purge > 60 jours avant 2025-07-29' reproduit l'etiquette dans 99.9641% des cas (5 exceptions).

## Censure a droite
Abonnes crees apres 2025-05-30 : 1311 dont 0 churners.

## Coherence
TOTAL_CHARGE = MONTHLY_CHARGE x RECHARGE_FREQUENCY : 100.00% des lignes
Coordonnees distinctes par region : 1 (=> centroide regional)