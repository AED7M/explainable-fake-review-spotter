#!/usr/bin/env python
"""
Prediction script for Fake Review Detection.

This script loads trained models and generates predictions on new review text.

Usage:
    python scripts/predict.py
    python scripts/predict.py --text "This product is amazing!"
    python scripts/predict.py --model stacking --text "Great product, highly recommend!"
"""
import argparse
import sys
from pathlib import Path
from typing import Union, List, Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    TEXT_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    STACKING_MODEL_PATH,
    VOTING_MODEL_PATH,
    XGBOOST_MODEL_PATH,
    SVM_MODEL_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
)
from src.preprocessing import (
    clean_text,
    remove_stopwords,
    lemmatize_text,
    extract_numeric_features,
    download_nltk_resources,
)


# Ensure NLTK resources are available
download_nltk_resources()


class FakeReviewPredictor:
    """
    Predictor class for fake review detection.
    
    Loads a trained model and provides methods for making predictions
    on new review text.
    """
    
    # Map model names to file paths
    MODEL_PATHS = {
        "stacking": STACKING_MODEL_PATH,
        "voting": VOTING_MODEL_PATH,
        "xgboost": XGBOOST_MODEL_PATH,
        "svm": SVM_MODEL_PATH,
        "logistic_regression": LOGISTIC_REGRESSION_MODEL_PATH,
        "lr": LOGISTIC_REGRESSION_MODEL_PATH,
    }
    
    # Class labels
    LABELS = {0: "Real (Human-Written)", 1: "Fake (Computer-Generated)"}
    
    def __init__(self, model_name: str = "stacking"):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_name: Name of the model to load. Options:
                        'stacking', 'voting', 'xgboost', 'svm', 'logistic_regression', 'lr'
        """
        self.model_name = model_name.lower()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        if self.model_name not in self.MODEL_PATHS:
            available = ", ".join(self.MODEL_PATHS.keys())
            raise ValueError(f"Unknown model '{self.model_name}'. Available: {available}")
        
        model_path = self.MODEL_PATHS[self.model_name]
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please run 'python scripts/train.py' first to train and save models."
            )
        
        print(f"📂 Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        print(f"✅ Model '{self.model_name}' loaded successfully!")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Apply the same preprocessing pipeline used during training.
        
        Args:
            text: Raw review text
            
        Returns:
            Preprocessed text
        """
        text = clean_text(text)
        text = remove_stopwords(text)
        text = lemmatize_text(text)
        return text
    
    def _prepare_input(self, texts: List[str]) -> pd.DataFrame:
        """
        Prepare input texts for prediction.
        
        Applies feature extraction and preprocessing to match training format.
        
        Args:
            texts: List of raw review texts
            
        Returns:
            DataFrame ready for model prediction
        """
        # Create DataFrame with raw text
        df = pd.DataFrame({TEXT_COLUMN: texts})
        
        # Extract numeric features from RAW text (before cleaning)
        df = extract_numeric_features(df, text_column=TEXT_COLUMN)
        
        # Apply text preprocessing (after feature extraction)
        df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(self._preprocess_text)
        
        return df
    
    def predict(self, text_input: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Generate predictions for new review text(s).
        
        Args:
            text_input: A single review string or list of review strings
            
        Returns:
            List of prediction dictionaries containing:
                - 'text': Original input text (truncated)
                - 'prediction': Predicted class (0=Real, 1=Fake)
                - 'label': Human-readable label
                - 'confidence': Confidence score (if available)
        """
        # Handle single string input
        if isinstance(text_input, str):
            texts = [text_input]
        else:
            texts = list(text_input)
        
        # Prepare input DataFrame
        df = self._prepare_input(texts)
        
        # Get predictions
        predictions = self.model.predict(df)
        
        # Build results
        results = []
        for text, pred in zip(texts, predictions):
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,
                "prediction": int(pred),
                "label": self.LABELS[int(pred)],
            }
            results.append(result)
        
        return results
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        Convenience method for single text prediction.
        
        Args:
            text: A single review string
            
        Returns:
            Prediction dictionary
        """
        return self.predict(text)[0]


def predict(text_input: Union[str, List[str]], model_name: str = "stacking") -> List[Dict[str, Any]]:
    """
    Convenience function for making predictions.
    
    Args:
        text_input: A single review string or list of review strings
        model_name: Name of the model to use (default: 'stacking')
        
    Returns:
        List of prediction dictionaries
    """
    predictor = FakeReviewPredictor(model_name=model_name)
    return predictor.predict(text_input)


def print_prediction(result: Dict[str, Any]):
    """Pretty print a single prediction result."""
    print("\n" + "=" * 60)
    print(f"📝 Text: {result['text']}")
    print("-" * 60)
    print(f"🎯 Prediction: {result['label']}")
    
    # Visual indicator
    if result["prediction"] == 1:
        print("⚠️  This review appears to be FAKE (Computer-Generated)")
    else:
        print("✅ This review appears to be REAL (Human-Written)")
    print("=" * 60)


def main(args):
    """Main function for command-line usage."""
    print("\n" + "=" * 60)
    print("🔍 FAKE REVIEW DETECTION - PREDICTION")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = FakeReviewPredictor(model_name=args.model)
        
        # Get text input
        if args.text:
            texts = [args.text]
        elif args.file:
            # Read texts from file (one per line)
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            print(f"📂 Loaded {len(texts)} reviews from {args.file}")
        else:
            # Use sample texts for demonstration
            texts = [
                "This product is absolutely amazing! Best purchase I've ever made. "
                "The quality is outstanding and it arrived super fast. Highly recommend!",
                
                "I bought this expecting great things based on the reviews but was "
                "disappointed. The material feels cheap and it broke after a week. "
                "Customer service was unhelpful. Would not buy again.",
                
                "Perfect perfect perfect! Five stars! Amazing product amazing seller "
                "amazing shipping! Best best best! Everyone should buy this now!!!",
            ]
            print("ℹ️  Using sample texts for demonstration...")
            print("   Use --text 'your review' to analyze specific text")
        
        # Make predictions
        print(f"\n🔄 Processing {len(texts)} review(s)...")
        results = predictor.predict(texts)
        
        # Print results
        for result in results:
            print_prediction(result)
        
        # Summary
        if len(results) > 1:
            fake_count = sum(1 for r in results if r["prediction"] == 1)
            real_count = len(results) - fake_count
            print(f"\n📊 Summary: {fake_count} Fake, {real_count} Real out of {len(results)} reviews")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: Run 'python scripts/train.py' first to train and save models.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict if reviews are fake or real",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Analyze a single review:
        python scripts/predict.py --text "This product is amazing!"
    
    Use a specific model:
        python scripts/predict.py --model xgboost --text "Great product!"
    
    Analyze reviews from a file:
        python scripts/predict.py --file reviews.txt
    
    Run with sample texts:
        python scripts/predict.py

Available models: stacking, voting, xgboost, svm, logistic_regression (or lr)
        """,
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Review text to analyze",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="stacking",
        help="Model to use for prediction (default: stacking)",
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file with reviews (one per line)",
    )
    
    args = parser.parse_args()
    main(args)
