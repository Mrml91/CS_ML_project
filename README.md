

Répertoire dans le cadre du cours de Machine Learning de 3ème année de CentraleSupélec.
[lien du challenge](https://www.kaggle.com/c/dreem-2-sleep-classification-challenge-2020)

Pour que le notebook soit fonctionnel, il faut mettre les données du challenge dans un dossier `kaggle_data` situé dans le même répertoire que le notebook.
Les modèles finaux sont dans le notebook `final.ipynb`.

<!-- 
### Questions en suspens:
- prendre en compte le temps écoulé entre la première mesure et l'actuelle, dernière mesure et l'actuelle, et diviser par la longueur de la nuit (?)
- traiter le pulse en fréquentiel
- essayer de normaliser par individu
- rescaling par méthode robuste ou virer les outliers (essayé)
- fonction score
- faire des quantile plots avec les quantiles sur les signaux de 30s puis moyenne écart-type classique
- visualisation : fréquence d'apparition des sleep stages, ...
- ajouter moyenne et moments aux features
- peut-on essayer de prédire des fenêtres/séquences (dont la taille sera à déterminer) plutôt que des mini-blocs de 30 secondes ?



### À faire
Lire l'article (en particulier pour détecter les features à ajouter)

### TABLE 

| Random Forest Params | Time Features Quantiles | Time Features Moments | Sleep Features | Pulse Freq (f_max, A_max) | Shifts | Comments | Training Score | Validation Score |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| - | 0.1, 0.5, 0.9 | - | No | - | 0 | - | 1| 0.67|
| - | 0.1, 0.5, 0.9 | - | Yes | - | 0 | - | 1 | 0.69|
| - | 0.1, 0.5, 0.9 | - | Yes | - | -1, 0, 1 | - | 1 | 0.7|
| - | 0.01, 0.1, 0.5, 0.9, 0.99 | - | Yes | - | -1, 0, 1 | - | 1| 0.7  |
| `min_samples_leaf=10` | 0.01, 0.1, 0.5, 0.9, 0.99 | - | Yes | - | -1, 0, 1 | - | 0.89 | 0.69  |
| `min_samples_leaf=10` | 0.01, 0.1, 0.5, 0.9, 0.99 | 1, 2 | Yes | - | -1, 0, 1 |  - | 0.89 | 0.69  |
| - | 0.01, 0.1, 0.5, 0.9, 0.99 | 1, 2 | Yes | - | -1, 0, 1 |  - | 1 | 0.697  |
| - | 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99 | - | Yes | - | -1, 0, 1 |  - | 1 | 0.708  |
| - | 0.01, DECILES, 0.99 | - | Yes | - | -1, 0, 1 |  - | 1 | 0.709  |
| `min_samples_leaf=10` | 0.01, DECILES, 0.99 | - | Yes | - | -1, 0, 1 |  - | 0.89 | 0.697 |
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 |  - | 0.89 | 0.699 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 |  - | 1 | 0.713929 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | Yes | -1, 0, 1 |  - | 1 | 0.7 |
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | Yes | -1, 0, 1 | - | 0.89 | 0.697 |
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | Pulse Only | -1, 0, 1 | - | 0.89 | 0.7 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | Pulse Only | -1, 0, 1 |  - | 1 | 0.7055 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | - |  1 | 0.7055 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX + derivee 0.5 | - | Yes | - | -1, 0, 1 |  - | 1 | 0.708 |
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX + derivee 0.5 | - | Yes | - | -1, 0, 1 |  - | 0.89 | 0.700|
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX + derivee MIN, MAX | - | Yes | - | -1, 0, 1 | - | 0.89| 0.7 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX + derivee MIN, MAX | - | Yes | - | -1, 0, 1 | - | 1| 0.703|
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `bandlog rescaled`| 1| 0.665 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = 10%, 90% for logmod`| 1| 0.7165 |
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = 10%, 90% for logmod`| ? | 0.70 < x < 0.71 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES for logmod`| 1 | 0.737|
| `min_samples_leaf=10` | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES for logmod`| 0.898 | 0.721 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES for logmod; eeg_mean only`| 1 | 0.645 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES and quantiles = 0.1, 0.5, 0.9 for logmod`| 1 | 0.719 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `interquantiles_inv = (0.1, 0.9), (0.3, 0.7) and quantiles_inv = 0.5 for logmod`| 1 | 0.74.. |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES and interquantiles_inv = (0.1, 0.9), (0.3, 0.7) for logmod`| 1 | 0.753 |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES and interquantiles_inv = (0.1, 0.9), (0.3, 0.7) for logmod, interquantiles = (0.1, 0.9), (0.3, 0.7) for time features`| 1 | 0.754 (Alex) - 0.742 (Mrml) |
| - | MIN, 0.01, ODD_DECILES, 0.99, MAX | - | Yes | - | -1, 0, 1 | `quantiles_inv = ODD_DECILES and interquantiles_inv = (0.1, 0.9), (0.3, 0.7) for logmod, interquantiles = (0.1, 0.9), (0.3, 0.7) for time features;  n_estimators = 300`| 1 | 0.756 (Alex)| -->
