# Scikit-Learn Installation Guide

## 📦 Installation Methods

### Method 1: Using pip
```bash
pip install scikit-learn
```

### Method 2: Using conda
```bash
conda install scikit-learn
```

### Method 3: Install complete data science environment
```bash
# Using pip
pip install scikit-learn matplotlib seaborn pandas numpy

# Or using conda
conda install scikit-learn matplotlib seaborn pandas numpy
```

## 🔍 Verify Installation

```python
import sklearn
print(f"Scikit-Learn version: {sklearn.__version__}")

# Test basic functionality
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification
print("✅ Installation successful!")
```

## 📚 Dependencies

- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **Matplotlib**: Data visualization
- **Pandas**: Data manipulation (optional but recommended)

## 🚀 Quick Start

```python
# Import common modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Create simple data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
predictions = model.predict([[6]])
print(f"Prediction result: {predictions[0]}")
```

## 🆘 Common Issues

### Q: Installation failed?
A: Try updating pip: `pip install --upgrade pip`

### Q: Version conflicts?
A: Use virtual environment:
```bash
python -m venv sklearn_env
source sklearn_env/bin/activate  # Linux/Mac
# or
sklearn_env\Scripts\activate  # Windows
pip install scikit-learn
```

### Q: Need GPU acceleration?
A: Scikit-Learn mainly uses CPU. For GPU acceleration, consider other libraries like XGBoost or CuML
