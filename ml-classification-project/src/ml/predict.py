"""
Prediction Module
=================

This module provides functions for loading saved models
and making predictions.

This will be used by the FastAPI backend in Step 3.

Usage:
    from src.ml.predict import IrisPredictor

    predictor = IrisPredictor()
    result = predictor.predict(5.1, 3.5, 1.4, 0.2)
"""

import joblib
import numpy as np
from pathlib import Path


class IrisPredictor:
    """
    A class to handle Iris flower predictions.

    STUDY NOTE:
    -----------
    Why use a class?
    - Encapsulates model loading and prediction logic
    - Loads model once, reuses for multiple predictions
    - Easy to integrate with APIs
    - Provides clean interface for predictions
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading the trained model.

        Parameters:
        -----------
        model_path : str, optional
            Path to the saved model. If None, uses default path.
        """
        if model_path is None:
            # Default path (relative to project root)
            model_path = Path(__file__).parent.parent.parent / 'models' / 'iris_classifier.joblib'

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """
        Load the model and metadata from disk.

        STUDY NOTE:
        -----------
        joblib.load() reconstructs the entire model artifact:
        - The trained model (including preprocessing pipeline)
        - Feature names
        - Target names
        - Metadata (version, training date, etc.)
        """
        try:
            artifact = joblib.load(self.model_path)

            self.model = artifact['model']
            self.feature_names = artifact['feature_names']
            self.target_names = artifact['target_names']
            self.model_name = artifact['model_name']
            self.version = artifact['version']

            print(f"✓ Model loaded: {self.model_name} v{self.version}")

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please run train.py first to train and save a model."
            )

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> dict:
        """
        Make a prediction for a single flower.

        Parameters:
        -----------
        sepal_length : float
            Length of sepal in cm
        sepal_width : float
            Width of sepal in cm
        petal_length : float
            Length of petal in cm
        petal_width : float
            Width of petal in cm

        Returns:
        --------
        dict : Prediction results including:
            - predicted_class: The predicted species name
            - predicted_label: Numeric label (0, 1, or 2)
            - probabilities: Dict of class probabilities
            - confidence: Highest probability (confidence score)

        STUDY NOTE:
        -----------
        predict() returns class labels
        predict_proba() returns probability for each class
        Confidence is the max probability - indicates model certainty
        """
        # Prepare input as 2D array (model expects this shape)
        X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Get prediction and probabilities
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Build result dictionary
        result = {
            'predicted_class': self.target_names[prediction],
            'predicted_label': int(prediction),
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.target_names, probabilities)
            },
            'confidence': float(max(probabilities)),
            'input_features': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            }
        }

        return result

    def predict_batch(self, samples: list) -> list:
        """
        Make predictions for multiple flowers.

        Parameters:
        -----------
        samples : list of dict
            Each dict should have keys: sepal_length, sepal_width,
            petal_length, petal_width

        Returns:
        --------
        list : List of prediction results

        STUDY NOTE:
        -----------
        Batch prediction is more efficient than calling
        predict() multiple times because:
        - Single model call
        - Vectorized operations
        - Less Python overhead
        """
        results = []
        for sample in samples:
            result = self.predict(
                sample['sepal_length'],
                sample['sepal_width'],
                sample['petal_length'],
                sample['petal_width']
            )
            results.append(result)
        return results

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
        --------
        dict : Model metadata
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_path': str(self.model_path)
        }


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   PREDICTION MODULE DEMO")
    print("=" * 60)

    # Initialize predictor
    predictor = IrisPredictor()

    # Print model info
    print("\n📋 Model Info:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")

    # Single prediction example
    print("\n" + "─" * 60)
    print("Single Prediction Example:")
    print("─" * 60)

    result = predictor.predict(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )

    print(f"\n🌸 Input Features:")
    for name, value in result['input_features'].items():
        print(f"   {name}: {value}")

    print(f"\n📊 Prediction:")
    print(f"   Class: {result['predicted_class']}")
    print(f"   Confidence: {result['confidence']:.2%}")

    print(f"\n   Probabilities:")
    for cls, prob in result['probabilities'].items():
        bar = "█" * int(prob * 20)
        print(f"   {cls:12}: {prob:.4f} {bar}")

    # Batch prediction example
    print("\n" + "─" * 60)
    print("Batch Prediction Example:")
    print("─" * 60)

    samples = [
        {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2},
        {'sepal_length': 6.7, 'sepal_width': 3.0, 'petal_length': 5.2, 'petal_width': 2.3},
        {'sepal_length': 5.9, 'sepal_width': 3.0, 'petal_length': 4.2, 'petal_width': 1.5},
    ]

    results = predictor.predict_batch(samples)

    print("\n📊 Batch Results:")
    for i, res in enumerate(results, 1):
        print(f"   Sample {i}: {res['predicted_class']} ({res['confidence']:.2%})")

    print("\n" + "=" * 60)
