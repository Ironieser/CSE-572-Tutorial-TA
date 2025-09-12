#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Scikit-Learn Classification Demo
Easy-to-understand classification example
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_iris_data():
    """Load the famous Iris dataset"""
    print("🌺 Loading Iris dataset...")
    
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    print(f"Dataset has {X.shape[0]} flowers")
    print(f"Each flower has {X.shape[1]} measurements")
    print(f"Flower types: {target_names}")
    
    return X, y, target_names

def simple_visualization(X, y, target_names):
    """Simple 2D visualization"""
    print("📊 Creating simple visualization...")
    
    # Use only 2 features for simplicity
    X_2d = X[:, [0, 2]]  # sepal length and petal length
    feature_names = ['Sepal Length', 'Petal Length']
    
    plt.figure(figsize=(10, 6))
    
    # Plot each flower type with different colors
    colors = ['red', 'green', 'blue']
    for i, (name, color) in enumerate(zip(target_names, colors)):
        mask = y == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=color, label=name, alpha=0.7, s=50)
    
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Iris Flowers - Can you see the groups?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/simple_iris_plot.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_2d, feature_names

def train_simple_model(X, y):
    """Train a simple classification model"""
    print("🤖 Training a simple model...")
    
    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training with {X_train.shape[0]} flowers")
    print(f"Testing with {X_test.shape[0]} flowers")
    
    # Use simple logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2%}")
    
    return model, X_test, y_test, y_pred

def visualize_predictions(X_test, y_test, y_pred, target_names):
    """Show predictions vs actual"""
    print("📈 Showing predictions...")
    
    plt.figure(figsize=(12, 5))
    
    # Left plot: True labels
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue']
    for i, (name, color) in enumerate(zip(target_names, colors)):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=name, alpha=0.7, s=50)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('True Flower Types')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right plot: Predictions
    plt.subplot(1, 2, 2)
    for i, (name, color) in enumerate(zip(target_names, colors)):
        mask = y_pred == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=color, label=name, alpha=0.7, s=50)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/predictions_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def simple_prediction_demo(model, target_names):
    """Simple prediction demonstration"""
    print("🔮 Let's predict some new flowers...")
    
    # Create some new flower measurements (only 2 features: sepal length and petal length)
    new_flowers = np.array([
        [5.1, 1.4],  # Small flower (sepal length, petal length)
        [6.2, 4.3],  # Medium flower
        [7.3, 6.3]   # Large flower
    ])
    
    # Make predictions
    predictions = model.predict(new_flowers)
    probabilities = model.predict_proba(new_flowers)
    
    print("\nNew flower predictions:")
    print("-" * 40)
    
    for i, (flower, pred, prob) in enumerate(zip(new_flowers, predictions, probabilities)):
        print(f"Flower {i+1}:")
        print(f"  Sepal Length: {flower[0]} cm, Petal Length: {flower[1]} cm")
        print(f"  Predicted type: {target_names[pred]}")
        print(f"  Confidence: {prob[pred]:.1%}")
        print()

def main():
    """Main function - simplified version"""
    print("=" * 50)
    print("🌺 Simple Classification Demo")
    print("=" * 50)
    
    # 1. Load data
    X, y, target_names = load_iris_data()
    
    # 2. Simple visualization
    X_2d, feature_names = simple_visualization(X, y, target_names)
    
    # 3. Train model
    model, X_test, y_test, y_pred = train_simple_model(X_2d, y)
    
    # 4. Show predictions
    visualize_predictions(X_test, y_test, y_pred, target_names)
    
    # 5. Predict new flowers
    simple_prediction_demo(model, target_names)
    
    print("✅ Done! Check the saved images.")
    print("=" * 50)

if __name__ == "__main__":
    main()
