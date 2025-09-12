#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scikit-Learn Linear Regression Demo
Simple function fitting: y = 2x + 1 + noise
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def generate_simple_data():
    """Generate simple linear data"""
    print("🔧 Generating data...")
    np.random.seed(42)  # Set random seed for reproducibility
    
    # Generate x data: from 0 to 10, 100 points
    x = np.linspace(0, 10, 100).reshape(-1, 1)
    
    # True function: y = 2x + 1
    y_true = 2 * x.flatten() + 1
    
    # Add noise
    noise = np.random.normal(0, 1, 100)
    y = y_true + noise
    
    return x, y, y_true

def train_linear_model(x, y):
    """Train linear regression model"""
    print("🤖 Training linear regression model...")
    
    # Split training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    
    # Predict
    y_pred = model.predict(x_test)
    
    return model, x_train, x_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2

def plot_results(x, y, y_true, model, x_test, y_test, y_pred):
    """Plot results"""
    print("📈 Plotting results...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Original data and true function
    ax1.scatter(x, y, alpha=0.6, label='Data points', color='blue')
    ax1.plot(x, y_true, 'r-', linewidth=2, label='True function: y = 2x + 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Original Data and True Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Prediction results
    ax2.scatter(x_test, y_test, alpha=0.6, label='Test data', color='blue')
    ax2.scatter(x_test, y_pred, alpha=0.6, label='Predictions', color='red')
    
    # Plot fitted line
    x_line = np.linspace(0, 10, 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax2.plot(x_line, y_line, 'g-', linewidth=2, label=f'Fitted line: y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Model Prediction Results')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/linear_regression_result.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=" * 50)
    print("🎯 Scikit-Learn Linear Regression Demo")
    print("=" * 50)
    
    # 1. Generate data
    x, y, y_true = generate_simple_data()
    print(f"Data shape: X={x.shape}, y={y.shape}")
    
    # 2. Train model
    model, x_train, x_test, y_train, y_test, y_pred = train_linear_model(x, y)
    
    # 3. Display model parameters
    print(f"\n📋 Model Parameters:")
    print(f"Slope (coef_): {model.coef_[0]:.4f}")
    print(f"Intercept (intercept_): {model.intercept_:.4f}")
    print(f"True parameters: slope=2.0, intercept=1.0")
    
    # 4. Evaluate model
    print(f"\n📊 Model Performance:")
    mse, r2 = evaluate_model(y_test, y_pred)
    
    # 5. Plot results
    plot_results(x, y, y_true, model, x_test, y_test, y_pred)
    
    print(f"\n✅ Complete! Results saved to linear_regression_result.png")
    print("=" * 50)

if __name__ == "__main__":
    main()
