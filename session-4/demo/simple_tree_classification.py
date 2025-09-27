"""
Simple Tree Classification with Iris Dataset
============================================

This demo shows decision trees in action with the classic Iris dataset.
Perfect for understanding how trees work with real-world data.

Author: Sixun Dong (CSE 572 TA)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Create results directory if it doesn't exist
os.makedirs('../results', exist_ok=True)

def load_and_explore_iris():
    """
    Load and explore the Iris dataset.
    """
    print("🌸 Loading Iris Dataset")
    print("=" * 30)
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names, target_names

def visualize_iris_data(X, y, feature_names, target_names):
    """
    Visualize the Iris dataset.
    """
    print("\n📊 Visualizing Iris Dataset")
    print("-" * 30)
    
    # Create a 2x2 subplot for pairwise feature comparisons
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Dataset - Pairwise Feature Comparisons', fontsize=16)
    
    # Define colors for each class
    colors = ['red', 'green', 'blue']
    
    # Plot all pairwise combinations
    feature_pairs = [(0, 1), (0, 2), (0, 3), (2, 3)]
    titles = ['Sepal Length vs Width', 'Sepal Length vs Petal Length', 
              'Sepal Length vs Petal Width', 'Petal Length vs Width']
    
    for idx, ((i, j), title) in enumerate(zip(feature_pairs, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for class_idx in range(3):
            mask = y == class_idx
            ax.scatter(X[mask, i], X[mask, j], 
                      c=colors[class_idx], label=target_names[class_idx], 
                      alpha=0.7, s=50)
        
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/iris_data_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_decision_tree(X, y, feature_names, target_names, max_depth=3):
    """
    Train a decision tree classifier.
    """
    print(f"\n🌳 Training Decision Tree (max_depth={max_depth})")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    return clf, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def visualize_tree_structure(clf, feature_names, target_names, max_depth):
    """
    Visualize the decision tree structure.
    """
    print(f"\n🎨 Visualizing Tree Structure")
    print("-" * 30)
    
    # Plot tree
    plt.figure(figsize=(15, 10))
    plot_tree(clf, 
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(f'Decision Tree Structure (max_depth={max_depth})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'../results/tree_structure_depth_{max_depth}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print tree rules
    print("\n📋 Tree Rules (Text Format):")
    print("-" * 30)
    tree_rules = export_text(clf, feature_names=feature_names)
    print(tree_rules)

def analyze_feature_importance(clf, feature_names):
    """
    Analyze and visualize feature importance.
    """
    print(f"\n🔍 Feature Importance Analysis")
    print("-" * 30)
    
    # Get feature importance
    importance = clf.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    print("Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Visualize importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance in Decision Tree')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(y_test, y_test_pred, target_names):
    """
    Evaluate the model performance.
    """
    print(f"\n📊 Model Evaluation")
    print("-" * 20)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_tree_depths(X, y, feature_names, target_names):
    """
    Compare decision trees with different depths.
    """
    print(f"\n🔬 Comparing Tree Depths")
    print("-" * 25)
    
    # Test different depths
    depths = [1, 2, 3, 4, 5, 10]
    train_accuracies = []
    test_accuracies = []
    
    for depth in depths:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train tree
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        
        # Calculate accuracies
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Depth {depth:2d}: Train={train_acc:.3f}, Test={test_acc:.3f}")
    
    # Visualize comparison
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accuracies, 'o-', label='Training Accuracy', linewidth=2)
    plt.plot(depths, test_accuracies, 's-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Tree Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Performance vs Depth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(depths)
    
    # Add value labels
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        plt.annotate(f'{train_acc:.3f}', (depths[i], train_acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{test_acc:.3f}', (depths[i], test_acc), 
                    textcoords="offset points", xytext=(0,-15), ha='center')
    
    plt.tight_layout()
    plt.savefig('../results/tree_depth_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main demonstration function.
    """
    print("🌸 Simple Tree Classification with Iris Dataset")
    print("=" * 50)
    
    # Load and explore data
    X, y, feature_names, target_names = load_and_explore_iris()
    
    # Visualize data
    visualize_iris_data(X, y, feature_names, target_names)
    
    # Train decision tree
    clf, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_decision_tree(
        X, y, feature_names, target_names, max_depth=3
    )
    
    # Visualize tree structure
    visualize_tree_structure(clf, feature_names, target_names, max_depth=3)
    
    # Analyze feature importance
    analyze_feature_importance(clf, feature_names)
    
    # Evaluate model
    evaluate_model(y_test, y_test_pred, target_names)
    
    # Compare different depths
    compare_tree_depths(X, y, feature_names, target_names)
    
    print("\n✅ Demo completed!")
    print("📁 Results saved in ../results/")
    print("   - iris_data_visualization.png")
    print("   - tree_structure_depth_3.png")
    print("   - feature_importance.png")
    print("   - confusion_matrix.png")
    print("   - tree_depth_comparison.png")

if __name__ == "__main__":
    # Import pandas for the demo
    import pandas as pd
    main()
