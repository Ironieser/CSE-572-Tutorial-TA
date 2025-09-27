# Decision Tree Theory

## 🌳 What is a Decision Tree?

A decision tree is a flowchart-like structure where each internal node represents a test on an attribute, each branch represents the outcome of the test, and each leaf node represents a class label or decision.

### Key Characteristics
- **Interpretable**: Easy to understand and explain
- **Non-parametric**: No assumptions about data distribution
- **Handles both numerical and categorical data**
- **Can model non-linear relationships**

## 🧠 How Decision Trees Work

### Basic Concept
Decision trees work by recursively partitioning the feature space into regions that are as "pure" as possible with respect to the target variable.

### Tree Construction Process
1. **Start with all data at the root**
2. **Find the best split** (feature + threshold)
3. **Create left and right child nodes**
4. **Repeat recursively** until stopping criteria met
5. **Assign class labels** to leaf nodes

## 📊 Splitting Criteria

### Gini Impurity
Measures how "mixed" a node is. Lower Gini = more pure.

```
Gini = 1 - Σ(p_i)²
```

Where p_i is the proportion of samples belonging to class i.

**Example:**
- Node with [5 Fail, 5 Pass]: Gini = 1 - (0.5² + 0.5²) = 0.5
- Node with [9 Excellent, 1 Fail]: Gini = 1 - (0.9² + 0.1²) = 0.18

### Entropy
Measures the amount of information in a node. Lower entropy = more pure.

```
Entropy = -Σ(p_i * log₂(p_i))
```

**Example:**
- Node with [5 Fail, 5 Pass]: Entropy = -(0.5*log₂(0.5) + 0.5*log₂(0.5)) = 1.0
- Node with [9 Excellent, 1 Fail]: Entropy = -(0.9*log₂(0.9) + 0.1*log₂(0.1)) = 0.47

### Information Gain
Measures how much information a split provides.

```
Information Gain = Entropy(parent) - Weighted Average Entropy(children)
```

## 🎯 Best Split Selection

### Algorithm
1. **For each feature:**
   - Find all possible split points
   - Calculate impurity for each split
   - Choose the split with lowest impurity

2. **Compare across features:**
   - Select the feature with the best split
   - Create the split

### Example: Student Grade Classification
```
Feature: Study Time
Threshold: 10.0

Left child:  [5 samples, 3 Fail, 2 Pass]  → Gini = 0.48
Right child: [5 samples, 0 Fail, 5 Excellent]  → Gini = 0.0
Weighted Gini = 0.5 * 0.48 + 0.5 * 0.0 = 0.24
```

## 🛑 Stopping Criteria

### When to Stop Splitting
1. **Maximum depth reached**
2. **Minimum samples per leaf**
3. **Minimum samples to split**
4. **All samples in node have same class**
5. **No improvement in impurity**

### Example Stopping Rules
```python
# Stop if depth > 3
if depth >= max_depth:
    return leaf_node

# Stop if less than 10 samples
if len(samples) < min_samples_split:
    return leaf_node

# Stop if all samples same class
if len(unique_classes) == 1:
    return leaf_node
```

## 🌿 Tree Structure

### Node Types
- **Internal Node**: Contains split condition
- **Leaf Node**: Contains prediction

### Node Information
```python
node = {
    'feature': 0,           # Which feature to split on (0=Study Time)
    'threshold': 10.0,      # Split threshold
    'left': left_subtree,   # Left child
    'right': right_subtree, # Right child
    'samples': 10,          # Number of samples
    'gini': 0.3             # Gini impurity at this node
}
```

### Leaf Node
```python
leaf = {
    'prediction': 'Excellent',   # Predicted class
    'samples': 5                 # Number of samples
}
```

## 🔮 Prediction Process

### How to Make Predictions
1. **Start at root node**
2. **Check split condition**
3. **Move to appropriate child**
4. **Repeat until leaf node**
5. **Return prediction**

### Example Prediction
```
Sample: [study_time=15, homework_rate=85]

Root: study_time <= 10.0? → No → Go right
Right: Leaf → Prediction: "Excellent"
```

## ⚖️ Advantages and Disadvantages

### Advantages
- **Easy to understand and interpret**
- **Requires little data preparation**
- **Handles both numerical and categorical data**
- **Can model non-linear relationships**
- **No feature scaling needed**

### Disadvantages
- **Prone to overfitting**
- **Unstable (small data changes → different tree)**
- **Biased towards features with many levels**
- **Can create overly complex trees**

## 🎛️ Hyperparameters

### Key Parameters
- **max_depth**: Maximum tree depth
- **min_samples_split**: Minimum samples to split
- **min_samples_leaf**: Minimum samples per leaf
- **criterion**: 'gini' or 'entropy'
- **max_features**: Number of features to consider

### Parameter Effects
- **Higher max_depth**: More complex, risk of overfitting
- **Higher min_samples_split**: Simpler trees, less overfitting
- **Higher min_samples_leaf**: Smoother decision boundaries

## 🔧 Implementation Details

### Recursive Tree Building
```python
def build_tree(X, y, depth=0):
    # Stopping criteria
    if depth >= max_depth or len(y) < min_samples_split:
        return create_leaf(y)
    
    # Find best split
    feature, threshold = find_best_split(X, y)
    
    # Split data
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold
    
    # Recursively build subtrees
    left_tree = build_tree(X[left_mask], y[left_mask], depth + 1)
    right_tree = build_tree(X[right_mask], y[right_mask], depth + 1)
    
    return {
        'feature': feature,
        'threshold': threshold,
        'left': left_tree,
        'right': right_tree
    }
```

### Finding Best Split
```python
def find_best_split(X, y):
    best_impurity = float('inf')
    best_feature = None
    best_threshold = None
    
    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            # Calculate weighted impurity
            left_impurity = gini_impurity(y[left_mask])
            right_impurity = gini_impurity(y[right_mask])
            
            left_weight = np.sum(left_mask) / len(y)
            right_weight = np.sum(right_mask) / len(y)
            
            weighted_impurity = left_weight * left_impurity + right_weight * right_impurity
            
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold
```

## 📈 Tree Visualization

### Text Representation
```
Study Time (hours/week) <= 10.0
├─ Yes: Study Time (hours/week) <= 6.0
│  ├─ Yes: Leaf node: Fail
│  └─ No: Leaf node: Pass
└─ No: Leaf node: Excellent
```

### Graphical Representation
- **Nodes**: Show split conditions
- **Edges**: Show split outcomes
- **Leaves**: Show predictions
- **Colors**: Show class distributions

## 🎓 Teaching Tips

### For Beginners
1. **Start with simple examples** (2 features, 2-3 classes)
2. **Use visualizations** to show decision boundaries
3. **Explain the intuition** behind each split
4. **Show how trees make decisions** step by step
5. **Use relatable examples** like student grades, movie recommendations

### Key Questions to Ask
- "What question should we ask first?"
- "How do we measure the quality of a split?"
- "When should we stop splitting?"
- "How can we interpret the final tree?"

### Common Misconceptions
- **Trees always find the best solution** → No, they're greedy
- **More splits always better** → No, risk of overfitting
- **Trees work well with any data** → No, they have limitations

## 🔬 Advanced Topics

### Ensemble Methods
- **Random Forest**: Multiple trees with random features
- **Gradient Boosting**: Trees that learn from previous mistakes
- **Bagging**: Bootstrap aggregating of multiple trees

### Tree Pruning
- **Pre-pruning**: Stop splitting early
- **Post-pruning**: Remove branches after building
- **Cost-complexity pruning**: Balance accuracy vs complexity

### Handling Missing Values
- **Surrogate splits**: Use other features when primary is missing
- **Default direction**: Go to most common child
- **Imputation**: Fill missing values before training

## 🎯 Practice Examples

### Recommended Learning Path
1. **Start with Example 1**: Student Grade Prediction (3-class, 2 features)
2. **Try Example 2**: Movie Recommendation (2-class, categorical data)
3. **Explore Example 3**: Shopping Decision (2-class, continuous data)
4. **Study Example 4**: Restaurant Choice (2-class, mixed data)
5. **Master Example 5**: Complex Training (4-class, 4 features, full ML workflow)

### Key Concepts to Practice
- **Gini Impurity**: How to calculate and interpret
- **Tree Construction**: Step-by-step splitting process
- **Decision Boundaries**: How trees partition feature space
- **Overfitting**: When to stop splitting
- **Model Evaluation**: Train/test split, accuracy, confusion matrix

---

*This theory guide provides the foundation for understanding decision trees. Practice with the demo files to see these concepts in action!*
