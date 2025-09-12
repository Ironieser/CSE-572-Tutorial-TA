#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scikit-Learn Classification Demo
Multi-class classification using Iris dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """Load and explore data"""
    print("🌺 Loading Iris dataset...")
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Feature names: {feature_names}")
    print(f"Target names: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names, target_names

def visualize_data(X, y, feature_names, target_names):
    """Visualize data"""
    print("📊 Visualizing data...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Select two features for visualization
    feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2)]
    
    for i, (feat1, feat2) in enumerate(feature_pairs):
        ax = axes[i]
        
        for class_idx in range(3):
            mask = y == class_idx
            ax.scatter(X[mask, feat1], X[mask, feat2], 
                      label=target_names[class_idx], alpha=0.7)
        
        ax.set_xlabel(feature_names[feat1])
        ax.set_ylabel(feature_names[feat2])
        ax.set_title(f'{feature_names[feat1]} vs {feature_names[feat2]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/iris_data_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    print("🤖 Training multiple classification models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'accuracy': accuracy
        }
        
        print(f"    {name} accuracy: {accuracy:.4f}")
    
    return results

def evaluate_models(results, y_test, target_names):
    """Evaluate model performance"""
    print("📊 Detailed model evaluation...")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    
    # Display detailed reports for all models
    for name, result in results.items():
        print(f"\n📋 {name} detailed report:")
        print(classification_report(y_test, result['y_pred'], 
                                  target_names=target_names))
    
    return best_model_name, results[best_model_name]

def plot_confusion_matrix(y_test, y_pred, target_names, model_name):
    """Plot confusion matrix"""
    print("📈 Plotting confusion matrix...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_prediction(best_model, feature_names, target_names):
    """Demonstrate prediction functionality"""
    print("🔮 Prediction demonstration...")
    
    # Create some new samples for prediction
    new_samples = np.array([
        [5.1, 3.5, 1.4, 0.2],  # Should be setosa
        [6.2, 2.9, 4.3, 1.3],  # Should be versicolor
        [7.3, 2.9, 6.3, 1.8]   # Should be virginica
    ])
    
    predictions = best_model.predict(new_samples)
    probabilities = best_model.predict_proba(new_samples)
    
    print("\nNew sample predictions:")
    for i, (sample, pred, prob) in enumerate(zip(new_samples, predictions, probabilities)):
        print(f"Sample {i+1}: {sample}")
        print(f"  Predicted class: {target_names[pred]}")
        print(f"  Class probabilities: {dict(zip(target_names, prob))}")
        print()

def main():
    """Main function"""
    print("=" * 60)
    print("🌺 Scikit-Learn Classification Demo - Iris Dataset")
    print("=" * 60)
    
    # 1. Load and explore data
    X, y, feature_names, target_names = load_and_explore_data()
    
    # 2. Visualize data
    visualize_data(X, y, feature_names, target_names)
    
    # 3. Split data
    print("\n🔄 Splitting training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 4. Train multiple models
    results = train_multiple_models(X_train, X_test, y_train, y_test)
    
    # 5. Evaluate models
    best_model_name, best_result = evaluate_models(results, y_test, target_names)
    
    # 6. Plot confusion matrix
    plot_confusion_matrix(y_test, best_result['y_pred'], target_names, best_model_name)
    
    # 7. Demonstrate prediction
    demonstrate_prediction(best_result['model'], feature_names, target_names)
    
    print("✅ Complete! Results saved to corresponding image files")
    print("=" * 60)

if __name__ == "__main__":
    main()