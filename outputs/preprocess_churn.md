# Pretraitement du jeu churn

1. Date de reference (derniere purge observee) : 2025-07-29
2. Colonnes constantes supprimees (5) : PURGEDONSGSN, PS_UTRAN, PS_GERAN, PS_E-UTRAN, EPSODBPOS
3. Colonnes quasi-constantes supprimees (2) : TGPPAMBRMAXUL, TGPPAMBRMAXDL
4. Variables de fuite supprimees : PURGE_TIME_ATSGSN, MONTHS_SINCE_PURGE, PURGEDONMME
5. TOTAL_CHARGE supprimee (identite exacte MONTHLY_CHARGE x RECHARGE_FREQUENCY)
6. Latitude/Longitude supprimees (centroide regional, redondant avec REGION)
7. Lignes retirees pour incoherence metier : 0
8. Winsorisation AGE sur [17, 33] : 160 valeurs bornees
9. Winsorisation TENURE_MONTHS sur [2, 8] : 132 valeurs bornees
10. Winsorisation RECHARGE_FREQUENCY sur [0, 3] : 0 valeurs bornees
11. Jeu de survie ecrit : 13937 lignes, 6135 evenements, 7802 censures
12. Abonnes crees apres 2025-05-30 retires (censure a droite) : 1311 lignes
13. Variables derivees : anciennete_jours, intensite_recharge, revenu_cumule, jamais_recharge, mois_creation, tranche_age
14. Jeu final : 12626 lignes (90.6% de l'original), 13 variables explicatives, taux de churn 48.59%