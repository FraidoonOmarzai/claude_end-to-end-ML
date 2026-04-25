"""
Exploratory Data Analysis (EDA) for Iris Dataset
=================================================

This script demonstrates how to explore and understand your data
before building ML models.

Key Learning Points:
- Loading datasets
- Understanding data structure
- Statistical analysis
- Visualization
- Identifying patterns

Run this script: python src/ml/eda.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

# Create output directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('notebooks/figures', exist_ok=True)


def load_data():
    """
    Load the Iris dataset from scikit-learn.

    STUDY NOTE:
    -----------
    scikit-learn comes with several built-in datasets for learning.
    load_iris() returns a Bunch object (similar to a dictionary) with:
    - data: Feature values (numpy array)
    - target: Target labels (numpy array)
    - feature_names: Names of features
    - target_names: Names of classes
    """
    print("=" * 60)
    print("STEP 1: Loading the Iris Dataset")
    print("=" * 60)

    # Load dataset
    iris = load_iris()

    # Convert to pandas DataFrame for easier manipulation
    # STUDY NOTE: DataFrames are the standard data structure for tabular data
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )

    # Add target column
    df['target'] = iris.target

    # Map numeric targets to actual flower names
    # STUDY NOTE: This makes the data more interpretable
    target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(target_map)

    print(f"\n✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


def basic_info(df):
    """
    Display basic information about the dataset.

    STUDY NOTE:
    -----------
    Always start EDA by understanding:
    1. How many samples do we have?
    2. What are the features?
    3. What are the data types?
    4. Are there missing values?
    """
    print("\n" + "=" * 60)
    print("STEP 2: Basic Dataset Information")
    print("=" * 60)

    # First few rows
    print("\n📋 First 5 rows of the dataset:")
    print(df.head().to_string())

    # Data types and non-null counts
    print("\n📋 Data Types and Non-Null Counts:")
    print(df.dtypes)

    # Check for missing values
    # STUDY NOTE: Missing values can significantly impact model performance
    print("\n📋 Missing Values per Column:")
    missing = df.isnull().sum()
    print(missing)
    print(f"\n✓ Total missing values: {missing.sum()}")

    # Shape
    print(f"\n📋 Dataset Shape: {df.shape}")
    print(f"  - Rows (samples): {df.shape[0]}")
    print(f"  - Columns (features + target): {df.shape[1]}")


def statistical_analysis(df):
    """
    Perform statistical analysis on the features.

    STUDY NOTE:
    -----------
    Statistical measures help understand:
    - Central tendency (mean, median)
    - Spread (std, min, max)
    - Distribution shape
    """
    print("\n" + "=" * 60)
    print("STEP 3: Statistical Analysis")
    print("=" * 60)

    # Descriptive statistics
    # STUDY NOTE: describe() gives count, mean, std, min, 25%, 50%, 75%, max
    print("\n📊 Descriptive Statistics:")
    print(df.describe().round(2).to_string())

    # Class distribution
    # STUDY NOTE: Imbalanced classes can cause biased models
    print("\n📊 Class Distribution:")
    class_dist = df['species'].value_counts()
    print(class_dist)
    print(f"\n✓ Classes are balanced! (50 samples each)")

    # Feature correlation
    # STUDY NOTE: High correlation between features might indicate redundancy
    print("\n📊 Feature Correlations:")
    numeric_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    corr_matrix = df[numeric_cols].corr().round(2)
    print(corr_matrix.to_string())


def create_visualizations(df):
    """
    Create visualizations to understand data patterns.

    STUDY NOTE:
    -----------
    Visualizations help identify:
    - Feature distributions
    - Class separability
    - Outliers
    - Relationships between features
    """
    print("\n" + "=" * 60)
    print("STEP 4: Creating Visualizations")
    print("=" * 60)

    # Set style
    sns.set_style("whitegrid")

    # 1. Pairplot - shows relationships between all feature pairs
    # STUDY NOTE: Pairplot is excellent for understanding feature relationships
    print("\n📈 Creating pairplot (relationships between features)...")
    fig = sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
    fig.savefig('notebooks/figures/01_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: notebooks/figures/01_pairplot.png")

    # 2. Correlation heatmap
    print("\n📈 Creating correlation heatmap...")
    plt.figure(figsize=(8, 6))
    numeric_cols = ['sepal length (cm)', 'sepal width (cm)',
                    'petal length (cm)', 'petal width (cm)']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm',
                center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('notebooks/figures/02_correlation_heatmap.png', dpi=150)
    plt.close()
    print("  ✓ Saved: notebooks/figures/02_correlation_heatmap.png")

    # 3. Box plots for each feature by species
    print("\n📈 Creating box plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(x='species', y=col, data=df, ax=ax, palette='Set2')
        ax.set_title(f'{col} by Species')

    plt.tight_layout()
    plt.savefig('notebooks/figures/03_boxplots.png', dpi=150)
    plt.close()
    print("  ✓ Saved: notebooks/figures/03_boxplots.png")

    # 4. Distribution plots
    print("\n📈 Creating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // 2, idx % 2]
        for species in df['species'].unique():
            subset = df[df['species'] == species]
            sns.kdeplot(data=subset[col], ax=ax, label=species, fill=True, alpha=0.3)
        ax.set_title(f'{col} Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig('notebooks/figures/04_distributions.png', dpi=150)
    plt.close()
    print("  ✓ Saved: notebooks/figures/04_distributions.png")

    # 5. Class distribution bar chart
    print("\n📈 Creating class distribution chart...")
    plt.figure(figsize=(8, 5))
    df['species'].value_counts().plot(kind='bar', color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Class Distribution')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('notebooks/figures/05_class_distribution.png', dpi=150)
    plt.close()
    print("  ✓ Saved: notebooks/figures/05_class_distribution.png")


def key_insights(df):
    """
    Summarize key insights from EDA.

    STUDY NOTE:
    -----------
    After EDA, document what you learned. This guides:
    - Feature selection
    - Model choice
    - Preprocessing decisions
    """
    print("\n" + "=" * 60)
    print("STEP 5: Key Insights")
    print("=" * 60)

    insights = """
    🔍 KEY INSIGHTS FROM EDA:

    1. DATA QUALITY
       ✓ No missing values - data is clean
       ✓ No obvious outliers
       ✓ All features are numeric (continuous)

    2. CLASS BALANCE
       ✓ Perfectly balanced: 50 samples per class
       ✓ No need for oversampling/undersampling techniques

    3. FEATURE OBSERVATIONS
       • Petal features (length & width) show better class separation
       • Setosa is linearly separable from other classes
       • Versicolor and Virginica have some overlap
       • Sepal width shows most overlap between classes

    4. CORRELATIONS
       • Petal length & petal width: highly correlated (0.96)
       • Sepal length & petal length: correlated (0.87)
       • This suggests we might not need all features

    5. MODEL RECOMMENDATIONS
       • Simple algorithms should work well (Logistic Regression, Decision Tree)
       • Petal features alone might give good accuracy
       • Multi-class classification needed (3 classes)
    """
    print(insights)


def save_data(df):
    """Save the processed data for later use."""
    print("\n" + "=" * 60)
    print("STEP 6: Saving Data")
    print("=" * 60)

    # Save to CSV
    df.to_csv('data/raw/iris.csv', index=False)
    print("\n✓ Data saved to: data/raw/iris.csv")


def main():
    """Run the complete EDA pipeline."""
    print("\n" + "=" * 60)
    print("   EXPLORATORY DATA ANALYSIS (EDA) - IRIS DATASET")
    print("=" * 60)

    # Run all EDA steps
    df = load_data()
    basic_info(df)
    statistical_analysis(df)
    create_visualizations(df)
    key_insights(df)
    save_data(df)

    print("\n" + "=" * 60)
    print("   EDA COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review the generated visualizations in notebooks/figures/")
    print("  2. Proceed to Step 2: Model Training")
    print("=" * 60 + "\n")

    return df


if __name__ == "__main__":
    main()
