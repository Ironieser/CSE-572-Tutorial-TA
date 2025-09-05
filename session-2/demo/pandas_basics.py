#!/usr/bin/env python3
"""
Pandas Basics Demo
CSE-572 Session 2: Common Pandas operations
"""

import pandas as pd
import os

def demo_1_import_and_create():
    """Demo 1: Import pandas and create DataFrame"""
    print("=" * 50)
    print("📊 Demo 1: Import pandas and create DataFrame")
    print("=" * 50)
    
    # Create DataFrame from dictionary
    data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [20, 22, 21],
        'City': ['New York', 'London', 'Tokyo']
    }
    
    df = pd.DataFrame(data)
    print("Created DataFrame:")
    print(df)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def demo_2_add_delete_rows():
    """Demo 2: Add and delete rows"""
    print("\n" + "=" * 50)
    print("➕ Demo 2: Add and delete rows")
    print("=" * 50)
    
    # Start with basic data
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [20, 22],
        'City': ['New York', 'London']
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Add new row
    new_row = {'Name': 'Charlie', 'Age': 21, 'City': 'Tokyo'}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print("\nAfter adding Charlie:")
    print(df)
    
    # Delete row by index
    df = df.drop(1)  # Delete Bob (index 1)
    print("\nAfter deleting Bob:")
    print(df)
    
    return df

def demo_3_add_delete_columns():
    """Demo 3: Add and delete columns"""
    print("\n" + "=" * 50)
    print("📋 Demo 3: Add and delete columns")
    print("=" * 50)
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [20, 22, 21]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Add new column
    df['City'] = ['New York', 'London', 'Tokyo']
    print("\nAfter adding City column:")
    print(df)
    
    # Add calculated column
    df['Age_Next_Year'] = df['Age'] + 1
    print("\nAfter adding Age_Next_Year column:")
    print(df)
    
    # Delete column
    df = df.drop('Age_Next_Year', axis=1)
    print("\nAfter deleting Age_Next_Year column:")
    print(df)
    
    return df

def demo_4_read_csv():
    """Demo 4: Read CSV file"""
    print("\n" + "=" * 50)
    print("📁 Demo 4: Read CSV file")
    print("=" * 50)
    
    # Read students data
    csv_path = os.path.join('..', 'data', 'students.csv')
    df = pd.read_csv(csv_path)
    
    print("Students data from CSV:")
    print(df)
    print(f"\nData info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def demo_5_modify_data():
    """Demo 5: Modify data values"""
    print("\n" + "=" * 50)
    print("✏️ Demo 5: Modify data values")
    print("=" * 50)
    
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 78, 92]
    })
    
    print("Original DataFrame:")
    print(df)
    
    # Modify specific value
    df.loc[1, 'Score'] = 88  # Change Bob's score
    print("\nAfter changing Bob's score to 88:")
    print(df)
    
    # Modify multiple values
    df.loc[df['Score'] > 85, 'Grade'] = 'A'
    df.loc[df['Score'] <= 85, 'Grade'] = 'B'
    print("\nAfter adding Grade column:")
    print(df)
    
    return df

def demo_6_filter_data():
    """Demo 6: Filter data"""
    print("\n" + "=" * 50)
    print("🔍 Demo 6: Filter data")
    print("=" * 50)
    
    # Read students data
    csv_path = os.path.join('..', 'data', 'students.csv')
    df = pd.read_csv(csv_path)
    
    print("All students:")
    print(df)
    
    # Filter by condition
    high_scores = df[df['Score'] > 80]
    print("\nStudents with score > 80:")
    print(high_scores)
    
    # Filter by multiple conditions
    grade_a = df[df['Grade'] == 'A']
    print("\nStudents with Grade A:")
    print(grade_a)
    
    return df

def demo_7_sort_data():
    """Demo 7: Sort data"""
    print("\n" + "=" * 50)
    print("📊 Demo 7: Sort data")
    print("=" * 50)
    
    csv_path = os.path.join('..', 'data', 'students.csv')
    df = pd.read_csv(csv_path)
    
    print("Original data:")
    print(df)
    
    # Sort by score (descending)
    df_sorted = df.sort_values('Score', ascending=False)
    print("\nSorted by Score (highest first):")
    print(df_sorted)
    
    return df_sorted

def demo_8_save_data():
    """Demo 8: Save data to CSV"""
    print("\n" + "=" * 50)
    print("💾 Demo 8: Save data to CSV")
    print("=" * 50)
    
    # Create sample data
    df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Score': [85, 88, 92],
        'Grade': ['B', 'B', 'A']
    })
    
    print("Data to save:")
    print(df)
    
    # Save to CSV
    output_file = 'output_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to: {output_file}")
    
    # Verify by reading back
    df_read = pd.read_csv(output_file)
    print("\nData read back from file:")
    print(df_read)
    
    return df

def main():
    """Run all demos"""
    print("🎓 CSE-572 Session 2: Pandas Basics Demo")
    print("=" * 60)
    
    # Run all demos
    demo_1_import_and_create()
    demo_2_add_delete_rows()
    demo_3_add_delete_columns()
    demo_4_read_csv()
    demo_5_modify_data()
    demo_6_filter_data()
    demo_7_sort_data()
    demo_8_save_data()
    
    print("\n" + "=" * 60)
    print("🎉 All demos completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
