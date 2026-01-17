Modèle de Probabilité de Défaut (PD)

Ce projet implémente un modèle de Probabilité de Défaut (PD) en risque de crédit
à l’aide d’une régression logistique et de l’encodage Weight of Evidence (WOE).

L’objectif est d’estimer la probabilité qu’un emprunteur fasse défaut
sur un crédit à partir de variables financières et comportementales.

 Objectif du projet

Construire un modèle de PD interprétable

Gérer un jeu de données déséquilibré

Améliorer la capacité de discrimination du modèle (AUC)

Appliquer des pratiques standards utilisées en banque

 Méthodologie

Le projet suit les étapes classiques d’un modèle de risque de crédit :

Analyse et préparation des données

Gestion du déséquilibre des classes

Modèle de base : régression logistique

Feature engineering avec WOE (Weight of Evidence)

Évaluation du modèle avec l’AUC (ROC)

 Évaluation du modèle

Variable cible : not.fully.paid

1 = défaut

0 = non-défaut

Métrique principale : AUC (Area Under the Curve)

AUC ≈ 0.67 pour le modèle de base

Amélioration de la performance après encodage WOE

 Structure du projet
credit-risk-pd/
│
├── project_estimate_pd.py   # Script principal
├── README.md                # Description du projet
├── requirements.txt         # Dépendances Python
 Exécution du projet


Installer les dépendances :

pip install -r requirements.txt

Lancer le script :

python project_estimate_pd.py
