# PricePulse-V1

PricePulse est un modèle d'intelligence artificielle développé pour prédire les mouvements du marché des cryptomonnaies en utilisant des réseaux de neurones récurrents LSTM (Long Short-Term Memory). Le modèle exploite des données historiques, des indicateurs techniques avancés, et des techniques de rééchantillonnage pour effectuer des prédictions fiables sur les tendances du marché.

## Description

Le modèle PricePulse analyse les séries temporelles des prix des cryptomonnaies et utilise des indicateurs techniques pour prédire si le prix d'une cryptomonnaie va augmenter ou diminuer dans un futur proche. Ce projet met en œuvre un réseau de neurones LSTM bidirectionnel pour apprendre les relations complexes et non linéaires entre ces données.

### Objectifs :
- Prédire la direction du prix des cryptomonnaies (hausse ou baisse).
- Utiliser des indicateurs techniques populaires tels que le RSI, MACD, EMA, et d'autres pour enrichir les prédictions.
- Offrir un outil fiable pour les traders souhaitant automatiser et améliorer leurs stratégies d'investissement.

## Fonctionnalités

- **Prédiction des tendances du marché** : Prédire si le prix de la cryptomonnaie va augmenter ou diminuer sur la base des données passées.
- **Utilisation d'indicateurs techniques** : RSI, MACD, SMA, EMA, Bollinger Bands, etc.
- **Entraînement sur données historiques** : Utilisation des données historiques des cryptomonnaies pour entraîner le modèle.
- **Rééchantillonnage pour un jeu de données équilibré** : Utilisation de SMOTE pour résoudre les problèmes de déséquilibre de classe.
- **Évaluation des performances** : Rapport de classification, matrice de confusion, courbe ROC et AUC pour évaluer les résultats du modèle.

## Aperçu

### Architecture du modèle

#### Prétraitement des données :
- Extraction des indicateurs techniques (RSI, MACD, ATR, etc.).
- Nettoyage et normalisation des données.
- Création d'une cible binaire basée sur l'évolution des prix.

#### Modèle LSTM :
- Architecture LSTM bidirectionnelle pour capturer les dépendances temporelles dans les séries de données.
- Dropout pour éviter le sur-apprentissage.

#### Entraînement et Validation :
- Rééchantillonnage des données d'entraînement via SMOTE.
- Entraînement du modèle avec validation croisée et réduction du taux d'apprentissage.

#### Évaluation :
- Prédiction des classes (augmentation ou diminution des prix).
- Calcul des métriques de performance comme l'accuracy, la matrice de confusion, l'AUC.

## Installation

### Prérequis
Avant de commencer, vous devez installer les bibliothèques suivantes :

```bash
pip install -r requirements.txt
```

#### Bibliothèques principales :
- **TensorFlow** : Pour l'entraînement du modèle LSTM.
- **Pandas** : Pour le traitement des données.
- **NumPy** : Pour les opérations numériques.
- **TA-Lib** : Pour calculer des indicateurs techniques.
- **Scikit-learn** : Pour la gestion des données et des outils d'évaluation.
- **Imbalanced-learn** : Pour le rééchantillonnage SMOTE.

### Cloner le dépôt
Clonez ce dépôt pour commencer à travailler avec le projet :

```bash
git clone https://github.com/thomas1908/PricePulse-V1.git
cd PricePulse
```

## Préparation des données

Téléchargez ou générez vos propres données historiques des cryptomonnaies (ex : Bitcoin, Ethereum) en format CSV, incluant des colonnes comme `timestamp`, `open`, `close`, `high`, `low`, et `volume`.

Placez vos données dans le dossier `model/`.

### Format du fichier `historical_data.csv` :
| timestamp            | open  | high  | low   | close | volume |
|----------------------|-------|-------|-------|-------|--------|
| 2025-01-01 00:00:00  | 30000 | 30500 | 29500 | 30050 | 1200   |

## Utilisation

### 1. Préparation des données
Le fichier `historical_data.csv` doit être formaté comme suit :

```csv
timestamp, open, high, low, close, volume
2025-01-01 00:00:00, 30000, 30500, 29500, 30050, 1200
```

### 2. Exécution du modèle
Pour entraîner le modèle, exécutez le script principal `model_training.py` :

```bash
python model/model_training.py
```

Cela exécutera les étapes suivantes :
- Chargement des données.
- Application des indicateurs techniques.
- Entraînement du modèle LSTM.
- Évaluation du modèle avec des métriques comme l'exactitude, la matrice de confusion et l'AUC.

### 3. Sauvegarde et chargement du modèle
Le modèle entraîné est sauvegardé dans le fichier `model/trading_model_nn.h5`. Vous pouvez également charger ce modèle pour effectuer des prédictions sur de nouvelles données :

```python
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model('model/trading_model_nn.h5')

# Faire des prédictions
predictions = model.predict(new_data)
```

## Évaluation du Modèle

Une fois l'entraînement terminé, le modèle est évalué en utilisant les données de test. Les principales métriques incluent :

- **Accuracy** : Taux de prédictions correctes.
- **Classification Report** : Précision, rappel et F-mesure.
- **Confusion Matrix** : Visualisation des prédictions vs valeurs réelles.
- **Courbe ROC et AUC** : Mesure de la qualité du modèle.

## Contribuer

Nous encourageons les contributions à ce projet ! Pour contribuer, veuillez suivre les étapes suivantes :

1. Forkez ce dépôt.
2. Créez une branche pour votre fonctionnalité (`git checkout -b ma-fonctionnalité`).
3. Commitez vos modifications (`git commit -am 'Ajout d’une fonctionnalité'`).
4. Poussez sur la branche (`git push origin ma-fonctionnalité`).
5. Ouvrez une pull request.

## Licence

Ce projet est sous la licence MIT. Voir le fichier `LICENSE` pour plus de détails.
```

Cette version est plus structurée avec des sections bien définies, ce qui facilite la compréhension et l'utilisation du projet. Est-ce que cela correspond à ce que tu attendais ?
