# Session 4: Decision Tree Classification

Week 5 - Mining Labeled Data: Decision Tree Classification (Coding DT by TA)

## 🎯 Learning Objectives
- Understand how decision trees work conceptually
- Learn tree construction algorithms: information gain and Gini impurity
- Implement decision trees from scratch
- Use Scikit-Learn's DecisionTreeClassifier
- Compare decision trees with other classification methods

## 📂 Course Content

### 🔧 Demo Examples

#### 🚀 Simple Examples (Start Here!)
- **Example 1: Student Grade Prediction** (`example_1_student_grades.py`)
  - 3-class classification: Fail/Pass/Excellent
  - Features: Study time, homework completion rate
  - Perfect for understanding basic tree concepts

- **Example 2: Movie Recommendation** (`example_2_movie_recommendation.py`)
  - 2-class classification: Recommend/Not Recommend
  - Features: Genre, rating
  - Shows how trees handle categorical data

- **Example 3: Shopping Decision** (`example_3_shopping_decision.py`)
  - 2-class classification: Buy/Don't Buy
  - Features: Price, rating
  - Demonstrates decision boundaries

- **Example 4: Restaurant Choice** (`example_4_restaurant_choice.py`)
  - 2-class classification: Go/Don't Go
  - Features: Distance, rating
  - Real-world decision making

#### 🔬 Complex Example
- **Example 5: Complex Training** (`example_5_complex_training.py`)
  - 4-class classification: Fail/Pass/Good/Excellent
  - Features: Study time, homework rate, participation, exam prep
  - Includes train/test split, model evaluation, visualization
  - Shows complete machine learning workflow

### 📊 Data Generation
- All demos include sample datasets
- Real-world examples: Iris flowers, wine classification
- No external data files required

### 📈 Results
- `results/` folder contains tree visualizations
- Decision boundaries and tree structure plots
- Performance comparison charts

### ⚙️ Setup Guide
- `setup/decision_tree_setup.md` - Complete installation instructions
- `THEORY.md` - Comprehensive decision tree theory
- `data/README.md` - Dataset information and usage

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install scikit-learn matplotlib seaborn pandas numpy graphviz
```

### 2. Run Examples (Recommended Order)

#### Start with Simple Examples
```bash
# Learn basic decision tree concepts
python demo/example_1_student_grades.py

# See trees with categorical data
python demo/example_2_movie_recommendation.py

# Understand decision boundaries
python demo/example_3_shopping_decision.py

# Real-world decision making
python demo/example_4_restaurant_choice.py
```

#### Then Try Complex Example
```bash
# Complete machine learning workflow
python demo/example_5_complex_training.py
```

## 📚 Key Concepts You'll Learn

### 🌳 Decision Trees (Tree-based Classification)
- **What**: Make decisions by asking yes/no questions
- **Example**: "Is petal length > 2.5cm?" → "Is sepal width < 3.0cm?" → "Setosa"
- **Advantage**: Easy to understand and interpret

### 🧠 Tree Construction
- **Information Gain**: How much information each split provides
- **Gini Impurity**: Measure of how "mixed" a node is
- **Stopping Criteria**: When to stop splitting

### 📊 Tree Visualization
- **Why it matters**: See exactly how decisions are made
- **What you'll see**: Tree structure, decision boundaries, feature importance
- **Key insight**: Trees create interpretable decision rules

## 🎓 Teaching Approach

### For Beginners
1. **Start with simple examples** - Basic tree construction
2. **Focus on intuition** - How do trees make decisions?
3. **Use visualizations** - See the tree structure and boundaries
4. **Real-world context** - Why are trees useful?

### Key Questions to Ask
- "What question should we ask first?"
- "How do we measure the quality of a split?"
- "When should we stop splitting?"
- "How can we interpret the final tree?"

## 🗓 Session Info
- **Date:** Week 5
- **Time:** Friday 4:50pm - 5:20pm
- **Recording:** Available on Canvas

## 🔗 Contact
**Sixun Dong** (CSE 572 TA)  
Homepage: [https://cv.ironieser.cc](https://cv.ironieser.cc)

---

*All materials are ready. Start with the simple demos to understand tree concepts!*
