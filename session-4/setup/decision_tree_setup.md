# Decision Tree Setup Guide

## 📦 Required Packages

### Core Dependencies
```bash
pip install scikit-learn matplotlib seaborn pandas numpy
```

### Optional Dependencies
```bash
pip install graphviz  # For tree visualization
pip install jupyter   # For interactive notebooks
```

## 🚀 Quick Installation

### All-in-One Command
```bash
pip install scikit-learn matplotlib seaborn pandas numpy graphviz
```

### Verify Installation
```python
import sklearn
import matplotlib
import seaborn
import pandas
import numpy
print("All packages installed successfully!")
```


## 🔧 Environment Setup

### Using Conda
```bash
# Create new environment
conda create -n cse572 python=3.9

# Activate environment
conda activate cse572

# Install packages
conda install scikit-learn matplotlib seaborn pandas numpy
pip install graphviz
```

### Using Virtual Environment
```bash
# Create virtual environment
python -m venv cse572_env

# Activate environment
# On Windows:
cse572_env\Scripts\activate
# On macOS/Linux:
source cse572_env/bin/activate

# Install packages
pip install scikit-learn matplotlib seaborn pandas numpy graphviz
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# If you get import errors, try:
pip install --upgrade scikit-learn matplotlib seaborn pandas numpy
```

#### 2. Graphviz Issues
```bash
# On macOS:
brew install graphviz

# On Ubuntu/Debian:
sudo apt-get install graphviz

# On Windows:
# Download from: https://graphviz.org/download/
```

#### 3. Matplotlib Backend Issues
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg'
import matplotlib.pyplot as plt
```

### Performance Issues

#### 1. Slow Tree Building
```python
# For large datasets, consider:
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(
    max_depth=10,        # Limit depth
    min_samples_split=20, # Increase minimum samples
    random_state=42
)
```

#### 2. Memory Issues
```python
# For memory-constrained environments:
import numpy as np
np.random.seed(42)  # For reproducible results
```

## 🧪 Test Your Setup

### Run This Test Script
```python
# test_setup.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate test data
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                   random_state=42)

# Train decision tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Make predictions
accuracy = tree.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

# Create simple plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('Test Data Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()

print("✅ Setup test completed successfully!")
```

## 📚 Additional Resources

### Documentation
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### Tutorials
- [Decision Tree Tutorial](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)
- [Tree Visualization](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html)

## 🎯 Next Steps

1. **Run the demo files** in order:
   ```bash
   python demo/example_1_student_grades.py
   python demo/example_2_movie_recommendation.py
   python demo/example_3_shopping_decision.py
   python demo/example_4_restaurant_choice.py
   python demo/example_5_complex_training.py
   ```

2. **Read the theory guide**: `THEORY.md`

3. **Explore the data**: `data/README.md`

## 💡 Tips for Success

- **Start simple**: Begin with the basic examples
- **Experiment**: Try different parameters and datasets
- **Visualize**: Always plot your data and results
- **Understand**: Focus on the concepts, not just the code

---

*If you encounter any issues, check the troubleshooting section or contact the TA.*
