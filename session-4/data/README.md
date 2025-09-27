# Session 4 Data Files

## 📊 Available Datasets

### Built-in Datasets
All demo files use built-in datasets from scikit-learn and generate synthetic data as needed:

- **Iris Dataset**: Classic 3-class flower classification
- **Wine Dataset**: 3-class wine classification with 13 features
- **Synthetic Datasets**: Generated for specific demonstrations

### No External Files Required
- All demos are self-contained
- Data is generated or loaded programmatically
- No need to download additional files

## 🎯 Dataset Usage

### Simple Examples
- `simple_decision_tree.py` → Generates 2D synthetic data
- `simple_tree_classification.py` → Uses Iris dataset

### Advanced Examples
- `decision_tree_scratch.py` → Generates complex 3D synthetic data
- `tree_comparison.py` → Uses both synthetic and wine datasets

## 📈 Data Characteristics

### Iris Dataset
- **Samples**: 150
- **Features**: 4 (sepal length, sepal width, petal length, petal width)
- **Classes**: 3 (setosa, versicolor, virginica)
- **Perfect for**: Understanding basic tree concepts

### Wine Dataset
- **Samples**: 178
- **Features**: 13 (alcohol, malic acid, ash, etc.)
- **Classes**: 3 (wine types)
- **Perfect for**: Comparing different algorithms

### Synthetic Datasets
- **Customizable**: Size, features, classes, noise
- **Perfect for**: Testing specific scenarios

## 🔧 Data Loading Examples

### Load Iris Dataset
```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names
```

### Load Wine Dataset
```python
from sklearn.datasets import load_wine
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names
```

### Generate Synthetic Data
```python
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=3,
    random_state=42
)
```

## 📚 Data Preprocessing

### Common Preprocessing Steps
1. **Train-Test Split**: Separate training and testing data
2. **Feature Scaling**: Normalize features (if needed)
3. **Stratified Sampling**: Maintain class distribution

### Example Preprocessing
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (optional for trees)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🎓 Teaching Notes

### For Beginners
- Start with Iris dataset (simple, well-known)
- Use 2D synthetic data for visualization
- Focus on understanding the data structure

### For Advanced Students
- Experiment with wine dataset (more features)
- Generate custom synthetic datasets
- Compare different data characteristics

## 🔍 Data Exploration

### Quick Data Analysis
```python
import pandas as pd
import numpy as np

# Convert to DataFrame for easy analysis
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Basic statistics
print(df.describe())

# Class distribution
print(df['target'].value_counts())

# Correlation analysis
print(df.corr())
```

### Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Pairwise feature plots
sns.pairplot(df, hue='target')
plt.show()

# Feature distributions
df.hist(figsize=(12, 8))
plt.show()
```

---

*All datasets are ready to use. No additional setup required!*
