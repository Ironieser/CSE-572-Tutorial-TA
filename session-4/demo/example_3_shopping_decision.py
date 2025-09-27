"""
Example 3: Shopping Decision
============================

Decide whether to buy based on product price and rating
- Features: Price (yuan), Rating (1-5 stars)
- Target: Buy/Don't Buy

Author: Sixun Dong (CSE 572 TA)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create results directory
os.makedirs('../results', exist_ok=True)

class SimpleDecisionTree:
    """Simple Decision Tree Implementation"""
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
    
    def gini_impurity(self, labels):
        """Calculate Gini Impurity"""
        if len(labels) == 0:
            return 0
        
        # Count each class
        label_counts = Counter(labels)
        total = len(labels)
        
        # Calculate Gini impurity
        gini = 1.0
        for count in label_counts.values():
            probability = count / total
            gini -= probability ** 2
        
        return gini
    
    def find_best_split(self, X, y):
        """Find the best split point"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            # Get all unique values for this feature
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # Split data based on threshold
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                left_gini = self.gini_impurity(y[left_mask])
                right_gini = self.gini_impurity(y[right_mask])
                
                left_weight = np.sum(left_mask) / len(y)
                right_weight = np.sum(right_mask) / len(y)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gini
    
    def build_tree(self, X, y, depth=0):
        """Recursively build decision tree"""
        # Stop conditions
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) <= 1):
            return {'prediction': Counter(y).most_common(1)[0][0]}
        
        # Find best split
        feature, threshold, gini = self.find_best_split(X, y)
        
        if feature is None:
            return {'prediction': Counter(y).most_common(1)[0][0]}
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        
        # Recursively build left and right subtrees
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree,
            'gini': gini
        }
    
    def fit(self, X, y):
        """Train decision tree"""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_single(self, x, tree):
        """Predict single sample"""
        if 'prediction' in tree:
            return tree['prediction']
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])
    
    def predict(self, X):
        """Predict multiple samples"""
        predictions = []
        for x in X:
            predictions.append(self.predict_single(x, self.tree))
        return np.array(predictions)
    
    def print_tree(self, tree=None, depth=0, feature_names=None):
        """Print tree structure"""
        if tree is None:
            tree = self.tree
        
        indent = "  " * depth
        
        if 'prediction' in tree:
            print(f"{indent}Leaf node: {tree['prediction']}")
        else:
            feature_name = feature_names[tree['feature']] if feature_names else f"Feature_{tree['feature']}"
            print(f"{indent}{feature_name} <= {tree['threshold']:.1f}")
            print(f"{indent}├─ Yes:")
            self.print_tree(tree['left'], depth + 1, feature_names)
            print(f"{indent}└─ No:")
            self.print_tree(tree['right'], depth + 1, feature_names)

def main():
    """Main demonstration function"""
    print("🛒 Example 3: Shopping Decision")
    print("=" * 40)
    
    # Create shopping data
    # Features: [Price (yuan), Rating (1-5 stars)]
    # Labels: 0=Don't Buy, 1=Buy
    X = np.array([
        [50, 4],   # 50 yuan, 4 stars → Buy
        [200, 2],  # 200 yuan, 2 stars → Don't Buy
        [80, 5],   # 80 yuan, 5 stars → Buy
        [300, 1],  # 300 yuan, 1 star → Don't Buy
        [120, 4],  # 120 yuan, 4 stars → Buy
        [150, 3],  # 150 yuan, 3 stars → Buy
        [400, 2],  # 400 yuan, 2 stars → Don't Buy
        [60, 5],   # 60 yuan, 5 stars → Buy
        [250, 1],  # 250 yuan, 1 star → Don't Buy
        [100, 4],  # 100 yuan, 4 stars → Buy
    ])
    
    y = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])  # Corresponding labels
    
    feature_names = ['Price (yuan)', 'Rating (stars)']
    class_names = ['Don\'t Buy', 'Buy']
    
    print("Shopping data:")
    for i, (price, rating) in enumerate(X):
        decision = "Buy" if y[i] == 1 else "Don\'t Buy"
        print(f"  {price} yuan, {rating} stars → {decision}")
    
    # Train decision tree
    print("\n🌳 Training decision tree...")
    tree = SimpleDecisionTree(max_depth=3)
    tree.fit(X, y)
    
    # Print tree structure
    print("\n📋 Decision tree structure:")
    print("-" * 30)
    tree.print_tree(feature_names=feature_names)
    
    # Visualization
    plt.figure(figsize=(15, 6))
    
    # Left plot: Data points
    plt.subplot(1, 2, 1)
    colors = ['red', 'green']
    labels = ['Don\'t Buy', 'Buy']
    
    for class_idx in range(2):
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[class_idx], label=labels[class_idx], 
                   s=100, alpha=0.7, edgecolors='black')
    
    plt.xlabel('Price (yuan)')
    plt.ylabel('Rating (stars)')
    plt.title('Shopping Decision Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yticks([1, 2, 3, 4, 5])
    
    # Right plot: Decision boundary
    plt.subplot(1, 2, 2)
    
    # Create grid
    x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20
    y_min, y_max = 0.5, 5.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Predict grid points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = tree.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Draw decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, colors=['#ffcccc', '#ccffcc'])
    plt.contour(xx, yy, Z, colors='black', linewidths=2, alpha=0.8)
    
    # Draw data points
    for class_idx in range(2):
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[class_idx], label=labels[class_idx], 
                   s=120, alpha=0.9, edgecolors='black', linewidth=2)
    
    plt.xlabel('Price (yuan)')
    plt.ylabel('Rating (stars)')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    plt.savefig('../results/shopping_decision.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test predictions
    print("\n🔮 Test predictions:")
    test_cases = [
        [90, 4],   # 90 yuan, 4 stars
        [180, 2],  # 180 yuan, 2 stars
        [70, 5],   # 70 yuan, 5 stars
    ]
    
    for price, rating in test_cases:
        prediction = tree.predict([[price, rating]])[0]
        result = "Buy" if prediction == 1 else "Don\'t Buy"
        print(f"  {price} yuan, {rating} stars → {result}")
    
    print("\n✅ Demo completed!")
    print("📁 Results saved to ../results/shopping_decision.png")

if __name__ == "__main__":
    main()
