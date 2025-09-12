# Session 3: Scikit-Learn Tutorial

Week 3 - Introduction to Scikit-Learn Machine Learning Library

## 🎯 Learning Objectives
- Understand basic concepts and usage of Scikit-Learn
- Master supervised learning: regression and classification
- Learn how to visualize and interpret machine learning results
- Practice with real-world examples and datasets

## 📂 Course Content

### 🔧 Demo Examples

#### 🚀 Simple Versions (Start Here!)
- **House Price Prediction** (`simple_regression_demo.py`) 
  - Real-world example: predict house prices from size
  - Complex non-linear relationship: `Price = 0.8 × Size^1.2 + 0.3 × Size + 20`
  - Shows how linear models approximate complex reality

- **Flower Classification** (`simple_classification_demo.py`)
  - Classic Iris dataset: identify flower types from measurements
  - Uses only 2 features for easy visualization
  - Perfect for understanding classification concepts

#### 🔬 Advanced Versions
- **Mathematical Regression** (`linear_regression_demo.py`) - Function fitting with noise
- **Multi-Algorithm Classification** (`classification_demo.py`) - Compare different algorithms

### 📊 Data Generation
- All demos generate their own sample data
- No external datasets required - everything runs out of the box!

### 📈 Results
- `results/` folder contains all generated visualizations
- Images are automatically saved when you run the demos
- Perfect for presentations and reports

### ⚙️ Setup Guide
- `setup/scikit_learn_setup.md` - Complete installation instructions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install scikit-learn matplotlib seaborn pandas numpy
```

### 2. Run Examples (Recommended Order)

#### Start with Simple Versions
```bash
# Learn regression with house prices
python demo/simple_regression_demo.py

# Learn classification with flowers
python demo/simple_classification_demo.py
```

#### Then Try Advanced Versions
```bash
# Mathematical function fitting
python demo/linear_regression_demo.py

# Multiple algorithms comparison
python demo/classification_demo.py
```

## 📚 Key Concepts You'll Learn

### 🏠 Regression (Predicting Numbers)
- **What**: Predict continuous values (prices, temperatures, scores)
- **Example**: "How much will this house cost?"
- **Challenge**: Real relationships are often complex and non-linear

### 🌺 Classification (Predicting Categories)
- **What**: Predict categories or labels (types, classes, groups)
- **Example**: "What type of flower is this?"
- **Challenge**: Find patterns that separate different groups

### 📊 Visualization & Interpretation
- **Why it matters**: Pictures help us understand what the model learned
- **What you'll see**: Data points, fitted lines, prediction accuracy
- **Key insight**: Good models find patterns that humans can understand

## 🎓 Teaching Approach

### For Beginners
1. **Start with simple examples** - House prices and flower types
2. **Focus on concepts** - What is the model trying to learn?
3. **Use visualizations** - See the data and predictions
4. **Real-world context** - Why does this matter?

### Key Questions to Ask
- "Can you see the pattern in the data?"
- "How well did the model learn?"
- "What would happen with new data?"
- "Why might this be useful in real life?"

## 🗓 Session Info
- **Date:** Week 3
- **Time:** TBD
- **Recording:** Available on Canvas

## 🔗 Contact
**Sixun Dong** (CSE 572 TA)  
Homepage: [https://cv.ironieser.cc](https://cv.ironieser.cc)

---

*All materials are ready. Start with the simple demos and work your way up!*
