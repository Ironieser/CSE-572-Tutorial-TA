#!/usr/bin/env python3
"""
Pandas Exercises
CSE-572 Session 2: Practice exercises for students
"""

import pandas as pd
import os

def exercise_1_create_dataframe():
    """Exercise 1: Create your own DataFrame"""
    print("📝 Exercise 1: Create DataFrame")
    print("-" * 40)
    
    # TODO: Create a DataFrame with your favorite movies
    # Columns: Title, Year, Rating, Genre
    # Add at least 3 movies
    
    # Your code here:
    movies = {
        'Title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight'],
        'Year': [1999, 2010, 2014, 2016],
        'Rating': [8.7, 8.8, 8.6, 9.0],
        'Genre': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Sci-Fi']
    }
    
    df = pd.DataFrame(movies)
    print("Your movies DataFrame:")
    print(df)
    
    return df

def exercise_2_add_data():
    """Exercise 2: Add new data"""
    print("\n📝 Exercise 2: Add new data")
    print("-" * 40)
    
    # Start with the movies DataFrame
    df = exercise_1_create_dataframe()
    
    # TODO: Add a new movie to the DataFrame
    # Hint: Use pd.concat() or df.loc[]
    
    # Your code here:
    new_movie = {'Title': 'The Dark Knight', 'Year': 2008, 'Rating': 9.0, 'Genre': 'Action'}
    df = pd.concat([df, pd.DataFrame([new_movie])], ignore_index=True)
    
    print("After adding new movie:")
    print(df)
    
    return df

def exercise_3_filter_data():
    """Exercise 3: Filter data"""
    print("\n📝 Exercise 3: Filter data")
    print("-" * 40)
    
    # Read students data
    csv_path = os.path.join('..', 'data', 'students.csv')
    df = pd.read_csv(csv_path)
    
    print("All students:")
    print(df)
    
    # TODO: Filter students with score > 80
    # TODO: Filter students in Grade A
    
    # Your code here:
    high_scores = df[df['Score'] > 80]
    print("\nStudents with score > 80:")
    print(high_scores)
    
    grade_a = df[df['Grade'] == 'A']
    print("\nStudents with Grade A:")
    print(grade_a)
    
    return df

def exercise_4_calculate_stats():
    """Exercise 4: Calculate statistics"""
    print("\n📝 Exercise 4: Calculate statistics")
    print("-" * 40)
    
    csv_path = os.path.join('..', 'data', 'students.csv')
    df = pd.read_csv(csv_path)
    
    # TODO: Calculate average score
    # TODO: Find highest and lowest scores
    # TODO: Count students by grade
    
    # Your code here:
    avg_score = df['Score'].mean()
    max_score = df['Score'].max()
    min_score = df['Score'].min()
    grade_counts = df['Grade'].value_counts()
    
    print(f"Average score: {avg_score:.1f}")
    print(f"Highest score: {max_score}")
    print(f"Lowest score: {min_score}")
    print(f"\nStudents by grade:")
    print(grade_counts)
    
    return df

def exercise_5_save_results():
    """Exercise 5: Save your results"""
    print("\n📝 Exercise 5: Save results")
    print("-" * 40)
    
    # Create some results
    results = pd.DataFrame({
        'Operation': ['Average Score', 'Max Score', 'Min Score'],
        'Value': [85.0, 95, 65]
    })
    
    print("Results to save:")
    print(results)
    
    # TODO: Save results to CSV file
    # Your code here:
    results.to_csv('my_results.csv', index=False)
    print("\n✅ Results saved to 'my_results.csv'")
    
    return results

def main():
    """Run all exercises"""
    print("🎓 CSE-572 Session 2: Pandas Exercises")
    print("=" * 50)
    print("Complete each exercise by modifying the code!")
    print("=" * 50)
    
    # Run exercises
    exercise_1_create_dataframe()
    exercise_2_add_data()
    exercise_3_filter_data()
    exercise_4_calculate_stats()
    exercise_5_save_results()
    
    print("\n" + "=" * 50)
    print("🎉 All exercises completed!")
    print("Check your 'my_results.csv' file!")
    print("=" * 50)

if __name__ == "__main__":
    main()
