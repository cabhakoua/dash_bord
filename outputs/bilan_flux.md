# Modele de flux : churn et demande

- Churn moyen predit par forfait (fenetre 60 j) : Blue One L 0.578, Blue One M 0.457, Blue One S 0.131
- Equivalent mensuel : Blue One L 0.350, Blue One M 0.263, Blue One S 0.068
- Risque relatif conserve lors du recalage : Blue One L x1.54, Blue One M x1.16, Blue One S x0.30
- Base active estimee (moyenne mai-juillet 2025, un abonne actif genere environ une activation par mois) : Blue One S 1,331, Blue One M 231,113, Blue One L 15,402
- Croissance mensuelle observee (Blue One M, 2024-2025) : +0.26%
- La stabilite de la base impose alpha ~ p* + g : alpha est deduit de l'identite de stock, il n'est pas postule.
- Mois de prevision retenus (complets) : 2025-09, 2025-10
- Activations prevues :
offre    Blue One L  Blue One M  Blue One S
mois                                       
2025-09     15927.0    209370.0      1468.0
2025-10     16219.0    211585.0      1484.0
- Revenu mensuel a risque, scenario central : 2025-09 46.2 M FCFA, 2025-10 45.6 M FCFA
- Taux de churn de l'echantillon : 48.6% => lift maximal atteignable 2.06
- Ciblage : {"top_5%": {"capture_%": 10.3, "lift": 2.06, "lift_max": 2.06}, "top_10%": {"capture_%": 20.6, "lift": 2.06, "lift_max": 2.06}, "top_20%": {"capture_%": 41.2, "lift": 2.06, "lift_max": 2.06}, "top_30%": {"capture_%": 56.8, "lift": 1.89, "lift_max": 2.06}, "top_50%": {"capture_%": 79.3, "lift": 1.59, "lift_max": 2.0}}

## Projection, scenario central

|   alpha | mois    | forfait    |   taux_depart_mensuel |   base_debut |   activations_prevues |   entrees |   departs_attendus |   base_fin |   variation_% |   revenu_a_risque_fcfa |   revenu_base_fcfa |
|--------:|:--------|:-----------|----------------------:|-------------:|----------------------:|----------:|-------------------:|-----------:|--------------:|-----------------------:|-------------------:|
|  0.0526 | 2025-09 | Blue One S |                0.0149 |         1331 |                  1468 |        77 |                 20 |       1389 |          4.31 |                  39715 |            2662667 |
|  0.0526 | 2025-09 | Blue One M |                0.0579 |       231113 |                209370 |     11014 |              13391 |     228736 |         -1.03 |               40174430 |          693339000 |
|  0.0526 | 2025-09 | Blue One L |                0.0771 |        15402 |                 15927 |       838 |               1188 |      15051 |         -2.27 |                5940493 |           77008333 |
|  0.0526 | 2025-10 | Blue One S |                0.0149 |         1389 |                  1484 |        78 |                 21 |       1446 |          4.13 |                  41426 |            2777373 |
|  0.0526 | 2025-10 | Blue One M |                0.0579 |       228736 |                211585 |     11131 |              13254 |     226613 |         -0.93 |               39761195 |          686207308 |
|  0.0526 | 2025-10 | Blue One L |                0.0771 |        15051 |                 16219 |       853 |               1161 |      14744 |         -2.05 |                5805407 |           75257174 |

## Comparaison des scenarios (dernier mois, Blue One M)

| scenario                      |   alpha |   taux_depart_mensuel |   base_fin |   variation_% |   revenu_a_risque_fcfa |
|:------------------------------|--------:|----------------------:|-----------:|--------------:|-----------------------:|
| prudent (p*=2%)               |  0.0226 |                0.0232 |     229931 |         -0.24 |               16026423 |
| central (p*=5%)               |  0.0526 |                0.0579 |     226613 |         -0.93 |               39761195 |
| haut (p*=10%)                 |  0.1026 |                0.1159 |     221354 |         -1.97 |               78506182 |
| echantillon brut (non recale) |  0.2296 |                0.2631 |     209521 |         -4.06 |              172342134 |