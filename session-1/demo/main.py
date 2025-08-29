#!/usr/bin/env python3
"""
CSE-572 Tutorial Demo
Simple Python example
"""

def main():
    """Main function"""
    print("=" * 40)
    print("🎓 CSE-572 Tutorial Demo")
    print("=" * 40)
    
    # 1. Basic math
    print("\n📊 1. Basic Math")
    numbers = [1, 2, 3, 4, 5]
    print(f"Numbers: {numbers}")
    print(f"Sum: {sum(numbers)}")
    print(f"Average: {sum(numbers)/len(numbers):.1f}")
    
    # 2. Working with data
    print("\n📈 2. Student Data")
    students = [
        {"name": "Alice", "age": 20, "score": 85},
        {"name": "Bob", "age": 22, "score": 92},
        {"name": "Charlie", "age": 21, "score": 78}
    ]
    
    for student in students:
        print(f"{student['name']}: Age {student['age']}, Score {student['score']}")
    
    # 3. Calculate average score
    scores = [student['score'] for student in students]
    avg_score = sum(scores) / len(scores)
    print(f"\nAverage score: {avg_score:.1f}")
    
    # 4. Find best student
    best_student = max(students, key=lambda x: x['score'])
    print(f"Best student: {best_student['name']} with score {best_student['score']}")
    
    # 5. Save results
    print("\n📁 3. Save Results")
    with open("results.txt", "w") as f:
        f.write("CSE-572 Demo Results\n")
        f.write("=" * 20 + "\n")
        f.write(f"Number of students: {len(students)}\n")
        f.write(f"Average score: {avg_score:.1f}\n")
        f.write(f"Best student: {best_student['name']}\n")
    
    print("✅ Results saved to results.txt")
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed!")
    print("=" * 40)

if __name__ == "__main__":
    main()
