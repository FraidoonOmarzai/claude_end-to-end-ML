"""
ML Model Training Script
========================

This script demonstrates the complete ML training workflow:
1. Load and preprocess data
2. Split into train/test sets
3. Train multiple models
4. Evaluate and compare models
5. Save the best model

Key Learning Points:
- Train/test splitting
- Model training with scikit-learn
- Evaluation metrics
- Model persistence with joblib

Run this script: python src/ml/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Create output directory for models
os.makedirs('models', exist_ok=True)


def load_data():
    """
    Load and prepare the Iris dataset.

    STUDY NOTE:
    -----------
    In ML, we separate:
    - X (features): The input variables used to make predictions
    - y (target): The output variable we want to predict

    This is the standard convention in scikit-learn.
    """
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    iris = load_iris()

    # X = features (150 samples, 4 features)
    X = iris.data

    # y = target labels (150 samples)
    y = iris.target

    # Store feature and target names for later use
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\n✓ Data loaded successfully!")
    print(f"  - Features shape: {X.shape}")
    print(f"  - Target shape: {y.shape}")
    print(f"  - Feature names: {feature_names}")
    print(f"  - Target classes: {list(target_names)}")

    return X, y, feature_names, target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.

    STUDY NOTE:
    -----------
    Why split data?
    - Training set: Used to train the model (learn patterns)
    - Test set: Used to evaluate how well model generalizes

    Parameters:
    - test_size=0.2: 20% for testing, 80% for training
    - random_state=42: Ensures reproducibility (same split every time)

    IMPORTANT: Never use test data during training!
    This would cause "data leakage" and give overly optimistic results.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Data (Train/Test)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class distribution in both sets
    )

    print(f"\n✓ Data split complete!")
    print(f"  - Training set: {X_train.shape[0]} samples ({100-test_size*100:.0f}%)")
    print(f"  - Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")

    # Verify stratification (class balance maintained)
    print(f"\n  Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"    - Class {cls}: {cnt} samples")

    return X_train, X_test, y_train, y_test


def create_models():
    """
    Create a dictionary of models to train and compare.

    STUDY NOTE:
    -----------
    We use scikit-learn Pipelines to combine:
    1. Preprocessing (StandardScaler)
    2. Model (Classifier)

    Why Pipelines?
    - Ensures preprocessing is applied consistently
    - Prevents data leakage (scaler fitted only on training data)
    - Makes code cleaner and more maintainable

    StandardScaler:
    - Transforms features to have mean=0 and std=1
    - Important for algorithms sensitive to feature scales
    - Formula: z = (x - mean) / std
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating Models")
    print("=" * 60)

    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,      # Max iterations for convergence
                random_state=42,
                multi_class='multinomial'  # For multi-class classification
            ))
        ]),

        'Decision Tree': Pipeline([
            ('scaler', StandardScaler()),  # Not required for trees, but consistent
            ('classifier', DecisionTreeClassifier(
                max_depth=5,        # Limit depth to prevent overfitting
                random_state=42
            ))
        ]),

        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,   # Number of trees
                max_depth=5,
                random_state=42,
                n_jobs=-1           # Use all CPU cores
            ))
        ])
    }

    print("\n✓ Models created:")
    for name in models.keys():
        print(f"  - {name}")

    print("\n📚 Model Descriptions:")
    print("""
    1. LOGISTIC REGRESSION
       - Linear model for classification
       - Fast, interpretable, good baseline
       - Works well when classes are linearly separable

    2. DECISION TREE
       - Tree-based model with if-else rules
       - Easy to visualize and interpret
       - Can overfit if not regularized (max_depth)

    3. RANDOM FOREST
       - Ensemble of many decision trees
       - More robust, less prone to overfitting
       - Generally better performance
    """)

    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test, target_names):
    """
    Train all models and evaluate their performance.

    STUDY NOTE:
    -----------
    Evaluation Metrics Explained:

    1. ACCURACY = (TP + TN) / Total
       - Percentage of correct predictions
       - Good for balanced classes

    2. PRECISION = TP / (TP + FP)
       - Of all positive predictions, how many are correct?
       - Important when false positives are costly

    3. RECALL = TP / (TP + FN)
       - Of all actual positives, how many did we find?
       - Important when false negatives are costly

    4. F1-SCORE = 2 * (Precision * Recall) / (Precision + Recall)
       - Harmonic mean of precision and recall
       - Good single metric when you need balance

    5. CONFUSION MATRIX
       - Shows all predictions vs actual values
       - Diagonal = correct, off-diagonal = errors
    """
    print("\n" + "=" * 60)
    print("STEP 4: Training and Evaluating Models")
    print("=" * 60)

    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print('─' * 50)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Cross-validation score (more robust estimate)
        # STUDY NOTE: Cross-validation trains on different folds
        # and gives a more reliable performance estimate
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }

        # Print results
        print(f"\n📊 Results for {name}:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   {target_names}")
        for i, row in enumerate(cm):
            print(f"   {target_names[i]:12} {row}")

        # Classification Report
        print(f"\n   Classification Report:")
        report = classification_report(y_test, y_pred, target_names=target_names)
        for line in report.split('\n'):
            print(f"   {line}")

    return results


def compare_models(results):
    """
    Compare all models and select the best one.

    STUDY NOTE:
    -----------
    Model selection criteria:
    - Usually pick the model with highest F1 or accuracy
    - Consider CV score for more robust selection
    - Also consider model complexity and interpretability
    """
    print("\n" + "=" * 60)
    print("STEP 5: Model Comparison")
    print("=" * 60)

    # Create comparison table
    print("\n📊 Model Comparison Summary:")
    print(f"\n{'Model':<25} {'Accuracy':<12} {'F1-Score':<12} {'CV Score':<15}")
    print("─" * 65)

    best_model_name = None
    best_f1 = 0

    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:.4f}       {metrics['f1']:.4f}       {metrics['cv_mean']:.4f} +/- {metrics['cv_std']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model_name = name

    print("─" * 65)
    print(f"\n✓ Best Model: {best_model_name} (F1-Score: {best_f1:.4f})")

    return best_model_name, results[best_model_name]['model']


def save_model(model, model_name, feature_names, target_names):
    """
    Save the trained model for later use.

    STUDY NOTE:
    -----------
    Model Persistence:
    - joblib is preferred for scikit-learn models
    - Saves the entire pipeline (scaler + model)
    - Can be loaded later for predictions

    We also save metadata for reference:
    - Feature names (what inputs the model expects)
    - Target names (what outputs mean)
    - Training date
    """
    print("\n" + "=" * 60)
    print("STEP 6: Saving Model")
    print("=" * 60)

    # Create model artifact dictionary
    model_artifact = {
        'model': model,
        'feature_names': feature_names,
        'target_names': list(target_names),
        'model_name': model_name,
        'training_date': datetime.now().isoformat(),
        'version': '1.0.0'
    }

    # Save with joblib
    model_path = 'models/iris_classifier.joblib'
    joblib.dump(model_artifact, model_path)

    print(f"\n✓ Model saved successfully!")
    print(f"  - Path: {model_path}")
    print(f"  - Model: {model_name}")
    print(f"  - Version: 1.0.0")

    # Also save a simple version (just the model)
    joblib.dump(model, 'models/model_only.joblib')
    print(f"  - Simple model: models/model_only.joblib")

    return model_path


def demonstrate_prediction(model, feature_names, target_names):
    """
    Demonstrate how to use the saved model for predictions.

    STUDY NOTE:
    -----------
    This shows the prediction workflow:
    1. Prepare input data (same format as training)
    2. Call model.predict() for class labels
    3. Call model.predict_proba() for probabilities
    """
    print("\n" + "=" * 60)
    print("STEP 7: Prediction Demo")
    print("=" * 60)

    # Example: A new flower measurement
    new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Likely Setosa

    print(f"\n🌸 New flower measurements:")
    for name, value in zip(feature_names, new_sample[0]):
        print(f"   - {name}: {value}")

    # Predict class
    prediction = model.predict(new_sample)
    predicted_class = target_names[prediction[0]]

    # Predict probabilities
    probabilities = model.predict_proba(new_sample)[0]

    print(f"\n📊 Prediction Results:")
    print(f"   - Predicted Class: {predicted_class}")
    print(f"\n   - Class Probabilities:")
    for name, prob in zip(target_names, probabilities):
        bar = "█" * int(prob * 20)
        print(f"     {name:12}: {prob:.4f} {bar}")


def main():
    """Run the complete training pipeline."""
    print("\n" + "=" * 60)
    print("   ML MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    X, y, feature_names, target_names = load_data()

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 3: Create models
    models = create_models()

    # Step 4: Train and evaluate
    results = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, target_names
    )

    # Step 5: Compare and select best
    best_model_name, best_model = compare_models(results)

    # Step 6: Save the best model
    model_path = save_model(best_model, best_model_name, feature_names, target_names)

    # Step 7: Demo prediction
    demonstrate_prediction(best_model, feature_names, target_names)

    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Model saved at: {model_path}")
    print(f"  2. Proceed to Step 3: FastAPI Backend")
    print("=" * 60 + "\n")

    return best_model, results


if __name__ == "__main__":
    main()
