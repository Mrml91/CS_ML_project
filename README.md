[lien du challenge kaggle](https://www.kaggle.com/c/dreem-2-sleep-classification-challenge-2020)

Le dossier `kaggle_data` contient les données du challenge.

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
