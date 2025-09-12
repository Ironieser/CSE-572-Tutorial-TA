#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Scikit-Learn Regression Demo
Easy-to-understand regression example
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set font for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_simple_data():
    """Create simple data that's easy to understand"""
    print("🔧 Creating simple data...")
    
    # Create data: house size vs price
    np.random.seed(42)
    
    # House sizes (in square meters)
    house_sizes = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    
    # House prices (in thousands of dollars)
    # Real relationship: price = 0.8 * size^1.2 + 0.3 * size + 20 + noise
    true_prices = 0.8 * (house_sizes ** 1.2) + 0.3 * house_sizes + 20
    noise = np.random.normal(0, 15, len(house_sizes))
    house_prices = true_prices + noise
    
    print(f"Created {len(house_sizes)} houses")
    print(f"Size range: {house_sizes.min()}-{house_sizes.max()} sqm")
    print(f"Price range: ${house_prices.min():.0f}k-${house_prices.max():.0f}k")
    
    return house_sizes, house_prices, true_prices

def visualize_data(sizes, prices, true_prices):
    """Show the data in a simple plot"""
    print("📊 Visualizing data...")
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(sizes, prices, color='blue', alpha=0.7, s=60, label='Houses')
    
    # Plot true relationship (what we want to find)
    plt.plot(sizes, true_prices, 'r--', linewidth=2, label='True relationship')
    
    plt.xlabel('House Size (square meters)')
    plt.ylabel('Price (thousands of dollars)')
    plt.title('House Size vs Price - Complex Real Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/simple_house_data.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def train_simple_model(sizes, prices):
    """Train a simple linear regression model"""
    print("🤖 Training model to find the pattern...")
    
    # Reshape data for sklearn
    X = sizes.reshape(-1, 1)  # Features (house sizes)
    y = prices                # Target (house prices)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate error
    mse = mean_squared_error(y, y_pred)
    
    print(f"Model learned: Price = {model.coef_[0]:.1f} × Size + {model.intercept_:.1f}")
    print(f"True relationship: Price = 0.8 × Size^1.2 + 0.3 × Size + 20")
    print(f"Average error: ${np.sqrt(mse):.1f}k")
    print("Note: Linear model tries to approximate the complex real relationship!")
    
    return model, y_pred

def visualize_predictions(sizes, prices, true_prices, model, y_pred):
    """Show how well the model learned"""
    print("📈 Showing model predictions...")
    
    plt.figure(figsize=(12, 5))
    
    # Left plot: Original data
    plt.subplot(1, 2, 1)
    plt.scatter(sizes, prices, color='blue', alpha=0.7, s=60, label='Houses')
    plt.plot(sizes, true_prices, 'r--', linewidth=2, label='True relationship')
    plt.xlabel('House Size (sqm)')
    plt.ylabel('Price (thousands $)')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right plot: Model predictions
    plt.subplot(1, 2, 2)
    plt.scatter(sizes, prices, color='blue', alpha=0.7, s=60, label='Houses')
    plt.plot(sizes, y_pred, 'g-', linewidth=2, label='Model prediction')
    plt.xlabel('House Size (sqm)')
    plt.ylabel('Price (thousands $)')
    plt.title('Model Learned the Pattern!')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/ironieser/project/CSE-572-Tutorial-TA/session-3/results/model_learning.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def predict_new_houses(model):
    """Predict prices for new houses"""
    print("🔮 Predicting prices for new houses...")
    
    # New house sizes
    new_sizes = np.array([75, 125, 175])
    new_prices = model.predict(new_sizes.reshape(-1, 1))
    
    print("\nNew house predictions:")
    print("-" * 30)
    
    for size, price in zip(new_sizes, new_prices):
        print(f"House size: {size} sqm")
        print(f"Predicted price: ${price:.0f}k")
        print()

def main():
    """Main function - simplified version"""
    print("=" * 50)
    print("🏠 Simple Regression Demo - House Prices")
    print("=" * 50)
    
    # 1. Create simple data
    sizes, prices, true_prices = create_simple_data()
    
    # 2. Show the data
    visualize_data(sizes, prices, true_prices)
    
    # 3. Train model
    model, y_pred = train_simple_model(sizes, prices)
    
    # 4. Show how well model learned
    visualize_predictions(sizes, prices, true_prices, model, y_pred)
    
    # 5. Predict new houses
    predict_new_houses(model)
    
    print("✅ Done! The model learned to predict house prices!")
    print("=" * 50)

if __name__ == "__main__":
    main()
