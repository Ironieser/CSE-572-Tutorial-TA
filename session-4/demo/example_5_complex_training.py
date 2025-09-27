"""
Example 5: Complex Training Example - Student Grade Prediction
=============================================================

This is a more complex example showing how to use decision trees for multi-class prediction
- Features: Study time, homework completion rate, class participation, exam preparation time
- Target: Predict grade levels (Fail/Pass/Good/Excellent)
- Includes train/test split, model evaluation, visualization

Author: Sixun Dong (CSE 572 TA)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create results directory
os.makedirs('../results', exist_ok=True)

class DecisionTreeClassifier:
    """Decision Tree Classifier"""
    
    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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
                
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
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
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'prediction': Counter(y).most_common(1)[0][0], 'samples': len(y)}
        
        # Find best split
        feature, threshold, gini = self.find_best_split(X, y)
        
        if feature is None:
            return {'prediction': Counter(y).most_common(1)[0][0], 'samples': len(y)}
        
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
            'gini': gini,
            'samples': len(y)
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
            print(f"{indent}Leaf node: {tree['prediction']} (samples: {tree['samples']})")
        else:
            feature_name = feature_names[tree['feature']] if feature_names else f"Feature_{tree['feature']}"
            print(f"{indent}{feature_name} <= {tree['threshold']:.1f} (Gini: {tree['gini']:.3f}, samples: {tree['samples']})")
            print(f"{indent}├─ Yes:")
            self.print_tree(tree['left'], depth + 1, feature_names)
            print(f"{indent}└─ No:")
            self.print_tree(tree['right'], depth + 1, feature_names)

def generate_student_data(n_samples=300):
    """Generate student grade data"""
    np.random.seed(42)
    
    # Generate features
    study_time = np.random.uniform(5, 25, n_samples)      # Study time (hours/week)
    homework_rate = np.random.uniform(40, 100, n_samples) # Homework completion rate (%)
    participation = np.random.uniform(1, 10, n_samples)   # Class participation (1-10)
    exam_prep = np.random.uniform(10, 50, n_samples)      # Exam preparation time (hours)
    
    # Generate target variable (grade levels: 0=Fail, 1=Pass, 2=Good, 3=Excellent)
    # Grade = study time influence + homework influence + participation influence + prep time influence + noise
    score = (study_time * 0.4 + 
             homework_rate * 0.3 + 
             participation * 3 + 
             exam_prep * 0.2 + 
             np.random.normal(0, 5, n_samples))
    
    # Convert to levels - adjust thresholds to ensure multiple classes
    y = np.zeros(n_samples, dtype=int)
    y[score >= 40] = 1  # Pass
    y[score >= 55] = 2  # Good
    y[score >= 70] = 3  # Excellent
    
    X = np.column_stack([study_time, homework_rate, participation, exam_prep])
    
    return X, y

def main():
    """Main demonstration function"""
    print("📚 Example 5: Complex Training Example - Student Grade Prediction")
    print("=" * 60)
    
    # Generate data
    print("📊 Generating student grade data...")
    X, y = generate_student_data(n_samples=300)
    
    feature_names = ['Study Time (hours/week)', 'Homework Completion (%)', 'Class Participation', 'Exam Prep Time (hours)']
    class_names = ['Fail', 'Pass', 'Good', 'Excellent']
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Grade distribution: {Counter(y)}")
    for i, name in enumerate(class_names):
        count = np.sum(y == i)
        print(f"  {name}: {count} students ({count/len(y)*100:.1f}%)")
    
    # Split training and test sets
    print("\n🔄 Splitting training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train decision tree
    print("\n🌳 Training decision tree classifier...")
    tree = DecisionTreeClassifier(max_depth=4, min_samples_split=10)
    tree.fit(X_train, y_train)
    
    # Print tree structure
    print("\n📋 Decision tree structure:")
    print("-" * 40)
    tree.print_tree(feature_names=feature_names)
    
    # Predict
    print("\n🔮 Making predictions...")
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    
    # Evaluate model
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📈 Model evaluation:")
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    print(f"\n📊 Classification report:")
    # Only show existing classes
    unique_classes = np.unique(np.concatenate([y_test, y_test_pred]))
    existing_class_names = [class_names[i] for i in unique_classes]
    print(classification_report(y_test, y_test_pred, labels=unique_classes, target_names=existing_class_names))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Student Grade Prediction Results Analysis', fontsize=16)
    
    # 1. Confusion matrix (test set)
    cm = confusion_matrix(y_test, y_test_pred)
    im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix (Test Set)', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    
    # Add numerical labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0, 0].text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=12)
    
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_yticks(range(len(class_names)))
    axes[0, 0].set_xticklabels(class_names, rotation=45, fontsize=11)
    axes[0, 0].set_yticklabels(class_names, fontsize=11)
    
    # 2. Feature importance (based on split count)
    feature_importance = np.zeros(len(feature_names))
    def count_splits(tree):
        if 'prediction' in tree:
            return 0
        feature_importance[tree['feature']] += 1
        return 1 + count_splits(tree['left']) + count_splits(tree['right'])
    
    count_splits(tree.tree)
    feature_importance = feature_importance / np.sum(feature_importance)
    
    bars = axes[0, 1].bar(range(len(feature_names)), feature_importance, color='skyblue', alpha=0.8)
    axes[0, 1].set_title('Feature Importance', fontsize=14)
    axes[0, 1].set_xlabel('Feature', fontsize=12)
    axes[0, 1].set_ylabel('Importance', fontsize=12)
    axes[0, 1].set_xticks(range(len(feature_names)))
    axes[0, 1].set_xticklabels([name.split('(')[0] for name in feature_names], rotation=45, fontsize=11)
    
    # Add numerical labels
    for bar, importance in zip(bars, feature_importance):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{importance:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Study time vs Homework completion rate (training set)
    colors = ['red', 'orange', 'lightgreen', 'green']
    for class_idx in range(4):
        mask = y_train == class_idx
        axes[1, 0].scatter(X_train[mask, 0], X_train[mask, 1], 
                          c=colors[class_idx], label=class_names[class_idx], 
                          alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
    
    axes[1, 0].set_xlabel('Study Time (hours/week)', fontsize=12)
    axes[1, 0].set_ylabel('Homework Completion (%)', fontsize=12)
    axes[1, 0].set_title('Training Set Data Distribution', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # 4. Prediction accuracy comparison
    categories = ['Training Set', 'Test Set']
    accuracies = [train_accuracy, test_accuracy]
    bars = axes[1, 1].bar(categories, accuracies, color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[1, 1].set_title('Model Accuracy Comparison', fontsize=14)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=11)
    
    # Add numerical labels
    for bar, acc in zip(bars, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/student_grade_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test predictions
    print("\n🔮 Test predictions:")
    test_cases = [
        [20, 95, 9, 40],  # Study 20 hours/week, homework 95%, participation 9, prep 40 hours
        [8, 60, 4, 15],   # Study 8 hours/week, homework 60%, participation 4, prep 15 hours
        [15, 85, 7, 30],  # Study 15 hours/week, homework 85%, participation 7, prep 30 hours
    ]
    
    for i, (study_time, homework_rate, participation, exam_prep) in enumerate(test_cases):
        prediction = tree.predict([[study_time, homework_rate, participation, exam_prep]])[0]
        result = class_names[prediction]
        print(f"  Student{i+1}: Study {study_time} hours/week, homework {homework_rate}%, participation {participation}, prep {exam_prep} hours → Predicted grade: {result}")
    
    print("\n✅ Demo completed!")
    print("📁 Results saved to ../results/student_grade_prediction.png")
    
    print("\n🎓 Learning points:")
    print("1. Decision trees can be used for multi-class problems")
    print("2. Use Gini impurity as the splitting criterion")
    print("3. Leaf node prediction is the mode of samples in that node")
    print("4. Evaluate model performance through train/test split")
    print("5. Confusion matrix and classification report help understand model performance")
    print("6. Feature importance analysis helps understand which features are most important")

if __name__ == "__main__":
    main()
