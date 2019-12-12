# Modelisation-distribution-de-donnees

librairies : scipy, numpy, pandas, matplotlib, statistics, statsmodel


Data_Analysis.py

Etude statistique sur un set de données :
- Traitement des outliers
- Autocorrélation pour la stationnarité
- Ljungi Box pour le bruit blanc
- Etude de la tendance

Résultat statistique et graphique


Fitting_Distribution.py

Prends le set de donnée et trouve parmi (quasi) toutes les lois, la loi qui fit le plus les données.
Boucle sur toutes les distributions possibles, et prends celle dont l'erreur SSE est la plus faible.

Résultat statistique et graphique
