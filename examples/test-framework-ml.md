# Test du Framework Entropic_Measurement avec des Données Réelles

## Vue d'ensemble

Ce guide pratique montre comment tester votre framework `entropic_measurement` avec des datasets réels de machine learning, en particulier pour détecter et corriger les biais algorithmiques.

## Datasets recommandés

### 1. Adult Census Dataset (Recommandé)
- **Description**: Prédiction de revenus >50K$ - très populaire pour fairness ML
- **Source**: UCI ML Repository / Fairlearn  
- **Chargement**: `from fairlearn.datasets import fetch_adult`
- **Attributs de biais**: genre, race, âge
- **Taille**: 48,842 individus
- **Pourquoi c'est bon**: Très étudié pour les biais, attributs protégés bien définis

### 2. COMPAS Recidivism Dataset
- **Description**: Prédiction de récidive criminelle
- **Source**: ProPublica / GitHub
- **Attributs de biais**: race, genre, âge  
- **Taille**: ~7,000 accusés
- **Pourquoi c'est bon**: Cas célèbre de biais algorithmique

### 3. Credit Approval Datasets
- **Description**: Approbation de crédit
- **Source**: UCI ML Repository
- **Attributs de biais**: genre, âge, statut marital
- **Pourquoi c'est bon**: Très pertinent pour l'industrie financière

## Exemple Pratique: Adult Dataset

### Installation des dépendances

```python
# Installation des packages nécessaires
!pip install fairlearn scikit-learn pandas numpy matplotlib seaborn
!pip install ucimlrepo  # package officiel UCI

# Importation de votre framework (après installation locale)
from entropic_measurement.entropy import EntropyEstimator  
from entropic_measurement.correction import BiasCorrector
from entropic_measurement.logger import EntropicLogger
```

### Chargement et exploration des données

```python
import pandas as pd
import numpy as np
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Charger le dataset Adult
data = fetch_adult(as_frame=True)
X, y = data.data, data.target

print(f"Dataset shape: {X.shape}")
print(f"Target distribution: {y.value_counts()}")
print(f"Features: {list(X.columns)}")

# Analyser les distributions par groupes sensibles
protected_attributes = ['sex', 'race']

for attr in protected_attributes:
    print(f"\n=== Distribution par {attr} ===")
    crosstab = pd.crosstab(X[attr], y, normalize='index')
    print(crosstab)
```

### Test des fonctions d'entropie

```python
def test_entropy_framework(X, y, protected_attr='sex'):
    """Test du framework avec calculs d'entropie par groupes"""
    
    results = {}
    
    for group in X[protected_attr].unique():
        # Filtrer les données par groupe
        group_mask = X[protected_attr] == group
        group_target = y[group_mask]
        
        # Calculer la distribution des classes
        class_dist = group_target.value_counts(normalize=True).values
        
        # Utiliser votre framework pour calculer l'entropie
        entropy = EntropyEstimator.shannon_entropy(class_dist)
        
        results[group] = {
            'size': len(group_target),
            'positive_rate': group_target.mean(),
            'entropy': entropy,
            'distribution': class_dist
        }
        
        print(f"Groupe {group}:")
        print(f"  Taille: {results[group]['size']}")
        print(f"  Taux positif: {results[group]['positive_rate']:.3f}")
        print(f"  Entropie: {results[group]['entropy']:.3f}")
        print()
    
    return results

# Exécuter le test
print("=== TEST DU FRAMEWORK ENTROPIC_MEASUREMENT ===")
entropy_results = test_entropy_framework(X, y, 'sex')
```

### Détection des biais avec entropie

```python
def detect_bias_with_entropy(results):
    """Détection des biais en utilisant les mesures d'entropie"""
    
    groups = list(results.keys())
    if len(groups) != 2:
        print("Analyse limitée à 2 groupes pour simplicité")
        return
    
    group1, group2 = groups
    
    # Différences dans les métriques
    entropy_diff = abs(results[group1]['entropy'] - results[group2]['entropy'])
    rate_diff = abs(results[group1]['positive_rate'] - results[group2]['positive_rate'])
    
    print(f"=== DÉTECTION DE BIAIS ===")
    print(f"Différence d'entropie: {entropy_diff:.3f}")
    print(f"Différence de taux positif: {rate_diff:.3f}")
    
    # Seuils pour détection (à ajuster selon votre framework)
    if entropy_diff > 0.1:
        print("⚠️  Biais détecté dans la distribution des classes")
    
    if rate_diff > 0.1:
        print("⚠️  Biais détecté dans les taux de décision")
    
    # Calcul de la divergence KL avec votre framework
    p = results[group1]['distribution']
    q = results[group2]['distribution']
    
    kl_div = EntropyEstimator.kullback_leibler(p, q)
    
    print(f"Divergence KL: {kl_div:.3f}")
    
    return {
        'entropy_diff': entropy_diff,
        'rate_diff': rate_diff,
        'kl_divergence': kl_div
    }

bias_metrics = detect_bias_with_entropy(entropy_results)
```

### Application de corrections avec logging

```python
def apply_bias_correction_with_logging(X, y, protected_attr='sex'):
    """Application des corrections avec traçabilité"""
    
    # Initialiser votre logger
    logger = EntropicLogger()
    
    corrections_applied = []
    
    for group in X[protected_attr].unique():
        group_mask = X[protected_attr] == group
        group_data = X[group_mask]
        group_target = y[group_mask]
        
        # Appliquer une correction avec votre framework
        corrected_predictions = BiasCorrector.apply_correction(
            observed_values=group_target.values,
            method='entropy_regularization'
        )
        
        original_rate = group_target.mean()
        corrected_rate = corrected_predictions.mean()
        
        correction_record = {
            'group': group,
            'original_rate': original_rate,
            'corrected_rate': corrected_rate,
            'entropy_cost': BiasCorrector.get_entropy_cost(),
            'method': 'entropy_regularization',
            'kl_divergence': EntropyEstimator.kullback_leibler(
                group_target.value_counts(normalize=True).values,
                corrected_predictions
            )
        }
        
        corrections_applied.append(correction_record)
        
        # Log avec votre framework
        logger.record(correction_record)
        
        print(f"Correction appliquée pour {group}:")
        print(f"  Taux original: {original_rate:.3f}")
        print(f"  Taux corrigé: {corrected_rate:.3f}")
        print(f"  Coût entropique: {correction_record['entropy_cost']:.3f}")
        print()
    
    # Export des logs
    logger.export('bias_correction_log.csv', format='csv')
    print("✅ Logs exportés vers bias_correction_log.csv")
    
    return corrections_applied

print("=== APPLICATION DES CORRECTIONS ===")
corrections = apply_bias_correction_with_logging(X, y, 'sex')
```

### Visualisation des résultats

```python
def visualize_entropy_analysis(entropy_results, corrections):
    """Visualisation des analyses d'entropie et corrections"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Distribution des entropies par groupe
    groups = list(entropy_results.keys())
    entropies = [entropy_results[g]['entropy'] for g in groups]
    
    axes[0,0].bar(groups, entropies, color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Entropie par Groupe')
    axes[0,0].set_ylabel('Entropie (bits)')
    
    # 2. Taux positifs par groupe
    rates = [entropy_results[g]['positive_rate'] for g in groups]
    axes[0,1].bar(groups, rates, color=['lightgreen', 'gold'])
    axes[0,1].set_title('Taux Positif par Groupe')
    axes[0,1].set_ylabel('Proportion')
    
    # 3. Avant/Après correction
    original_rates = [c['original_rate'] for c in corrections]
    corrected_rates = [c['corrected_rate'] for c in corrections]
    
    x = np.arange(len(groups))
    width = 0.35
    
    axes[1,0].bar(x - width/2, original_rates, width, label='Original', alpha=0.8)
    axes[1,0].bar(x + width/2, corrected_rates, width, label='Corrigé', alpha=0.8)
    axes[1,0].set_title('Taux Avant/Après Correction')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(groups)
    axes[1,0].legend()
    axes[1,0].set_ylabel('Taux')
    
    # 4. Coûts entropiques
    costs = [c['entropy_cost'] for c in corrections]
    axes[1,1].bar(groups, costs, color=['mediumpurple', 'orange'])
    axes[1,1].set_title('Coût Entropique des Corrections')
    axes[1,1].set_ylabel('Coût')
    
    plt.tight_layout()
    plt.show()

# Créer la visualisation
visualize_entropy_analysis(entropy_results, corrections)
```

## Autres datasets à tester

### COMPAS Dataset

```python
# Télécharger depuis GitHub
import requests
import io

url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
compas_data = pd.read_csv(url)

# Tester avec les attributs race et gender
test_entropy_framework(compas_data, compas_data['two_year_recid'], 'race')
```

### Breast Cancer Dataset (pour test d'entropie des features)

```python
from sklearn.datasets import load_breast_cancer

# Charger le dataset
cancer_data = load_breast_cancer(as_frame=True)
X_cancer = cancer_data.data
y_cancer = cancer_data.target

# Test d'entropie pour sélection de features
feature_entropies = {}
for col in X_cancer.columns:
    # Discrétiser la feature continue
    discretized = pd.cut(X_cancer[col], bins=5, labels=False)
    # Calculer l'entropie conditionnelle
    conditional_entropy = 0
    for target_class in [0, 1]:
        class_mask = y_cancer == target_class
        class_dist = discretized[class_mask].value_counts(normalize=True).values
        if len(class_dist) > 0:
            entropy = EntropyEstimator.shannon_entropy(class_dist)
            conditional_entropy += (class_mask.sum() / len(y_cancer)) * entropy
    
    feature_entropies[col] = conditional_entropy

# Sélectionner les features avec la plus faible entropie conditionnelle
top_features = sorted(feature_entropies.items(), key=lambda x: x[1])[:10]
print("Top 10 features (plus faible entropie conditionnelle):")
for feature, entropy in top_features:
    print(f"{feature}: {entropy:.3f}")
```

## Recommandations pour améliorer votre framework

### 1. Intégration avec datasets populaires
```python
# Créer des helpers dans votre package
from entropic_measurement.datasets import load_adult, load_compas, load_credit

# Usage simplifié
adult_data = load_adult()
entropy_analysis = adult_data.analyze_bias(['sex', 'race'])
```

### 2. Détection automatique de biais
```python
# Implémenter des seuils par défaut
class BiasDetector:
    def __init__(self, entropy_threshold=0.1, rate_threshold=0.1):
        self.entropy_threshold = entropy_threshold
        self.rate_threshold = rate_threshold
    
    def detect_bias(self, data, target, protected_attrs):
        # Détection automatique avec alertes
        pass
```

### 3. Rapports automatiques
```python
# Génération de rapports PDF
from entropic_measurement.reporting import BiasReport

report = BiasReport(data, target, protected_attributes=['sex', 'race'])
report.generate_pdf('bias_analysis_report.pdf')
```

### 4. Benchmarking
```python
# Comparaisons avec d'autres librairies
from entropic_measurement.benchmarks import compare_with_scipy, compare_with_fairlearn

results = compare_with_scipy(your_entropy_function, scipy_entropy_function, test_data)
print(f"Différence moyenne: {results['mean_diff']:.6f}")
```

## Métriques de succès

Testez votre framework sur ces critères :

1. **Précision des calculs** : Comparer avec scipy.stats.entropy
2. **Performance** : Temps d'exécution sur de gros datasets
3. **Détection de biais** : Capacité à identifier les biais connus dans Adult/COMPAS
4. **Reproductibilité** : Résultats identiques entre exécutions
5. **Facilité d'usage** : API intuitive pour les data scientists

## Cas d'usage industriels

- **Finance** : Détection de biais dans l'octroi de crédit
- **RH** : Équité dans les processus de recrutement  
- **Justice** : Analyse de biais dans les systèmes d'aide à la décision
- **Santé** : Équité des diagnostics entre groupes démographiques

Ce framework testé sur des données réelles démontre sa valeur pratique et sa robustesse pour des applications industrielles.