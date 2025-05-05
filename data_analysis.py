# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
def load_and_explore_data():
    try:
        # Load the Iris dataset
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['species'] = iris.target
        iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        
        # Display first few rows
        print("First 5 rows of the dataset:")
        print(iris_df.head())
        
        # Explore structure
        print("\nDataset info:")
        print(iris_df.info())
        
        # Check for missing values
        print("\nMissing values:")
        print(iris_df.isnull().sum())
        
        # Since there are no missing values in this dataset, we don't need to clean
        # But here's how we would handle missing values if they existed:
        # iris_df = iris_df.dropna()  # or iris_df.fillna(value)
        
        return iris_df
    
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Task 2: Basic Data Analysis
def perform_data_analysis(df):
    if df is None:
        return
    
    print("\nBasic statistics of numerical columns:")
    print(df.describe())
    
    # Group by species and compute mean for each feature
    print("\nMean values by species:")
    print(df.groupby('species').mean())
    
    # Additional interesting findings
    print("\nInteresting findings:")
    print("- Setosa has significantly smaller petal dimensions compared to other species")
    print("- Virginica has the largest sepal length on average")
    print("- Versicolor is intermediate in most measurements")

# Task 3: Data Visualization
def create_visualizations(df):
    if df is None:
        return
    
    # Set style for better looking plots
    sns.set(style="whitegrid")
    
    # 1. Line chart (simulating trends over time - since Iris isn't time series, we'll use index as x-axis)
    plt.figure(figsize=(10, 6))
    df['sepal length (cm)'].plot(kind='line', title='Sepal Length Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.show()
    
    # 2. Bar chart - average petal length per species
    plt.figure(figsize=(8, 5))
    df.groupby('species')['petal length (cm)'].mean().plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.xticks(rotation=0)
    plt.show()
    
    # 3. Histogram - distribution of sepal width
    plt.figure(figsize=(8, 5))
    df['sepal width (cm)'].plot(kind='hist', bins=15, color='purple', alpha=0.7)
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.show()
    
    # 4. Scatter plot - sepal length vs petal length
    plt.figure(figsize=(8, 6))
    colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
    for species, group in df.groupby('species'):
        plt.scatter(group['sepal length (cm)'], group['petal length (cm)'], 
                   label=species, color=colors[species], alpha=0.7)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.show()
    
    # Bonus: Pairplot to show all relationships
    sns.pairplot(df, hue='species', height=2.5)
    plt.suptitle('Pairwise Relationships in Iris Dataset', y=1.02)
    plt.show()

# Main execution
if __name__ == "__main__":
    print("=== Task 1: Load and Explore the Dataset ===")
    iris_df = load_and_explore_data()
    
    print("\n=== Task 2: Basic Data Analysis ===")
    perform_data_analysis(iris_df)
    
    print("\n=== Task 3: Data Visualization ===")
    create_visualizations(iris_df)