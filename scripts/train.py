#!/usr/bin/env python
"""
Training script for Fake Review Detection models.

This script trains and saves all models for the fake review detection pipeline:
- Logistic Regression
- SVM (LinearSVC)
- XGBoost
- Voting Classifier (ensemble)
- Stacking Classifier (ensemble)

Usage:
    python scripts/train.py
    python scripts/train.py --model stacking
    python scripts/train.py --data-path /path/to/data.csv
"""
import argparse
import sys
import os
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TARGET_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    TFIDF_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    SVM_PARAMS,
    XGBOOST_PARAMS,
    CV_FOLDS,
)
from src.preprocessing import (
    load_and_prepare_data,
    prepare_features,
    create_preprocessor,
)


def evaluate_model(model, X, y, dataset_name: str = "Dataset") -> dict:
    """
    Evaluate model and print metrics.
    
    Args:
        model: Trained model or pipeline
        X: Feature matrix
        y: True labels
        dataset_name: Name for display
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} Set Evaluation")
    print(f"{'='*60}")
    
    y_pred = model.predict(X)
    
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n📈 Metrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=["Real", "Fake"]))
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_logistic_regression(preprocessor, X_train, y_train) -> Pipeline:
    """Train Logistic Regression pipeline with best hyperparameters."""
    print("\n" + "="*60)
    print("🔧 Training Logistic Regression...")
    print("="*60)
    
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("clf", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
    ])
    
    pipeline.fit(X_train, y_train)
    print("✅ Logistic Regression training complete!")
    
    return pipeline


def train_svm(preprocessor, X_train, y_train) -> Pipeline:
    """Train SVM pipeline with best hyperparameters."""
    print("\n" + "="*60)
    print("🔧 Training SVM (LinearSVC)...")
    print("="*60)
    
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("clf", LinearSVC(**SVM_PARAMS)),
    ])
    
    pipeline.fit(X_train, y_train)
    print("✅ SVM training complete!")
    
    return pipeline


def train_xgboost(preprocessor, X_train, y_train) -> Pipeline:
    """Train XGBoost pipeline with best hyperparameters."""
    print("\n" + "="*60)
    print("🔧 Training XGBoost...")
    print("="*60)
    
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("clf", XGBClassifier(**XGBOOST_PARAMS)),
    ])
    
    pipeline.fit(X_train, y_train)
    print("✅ XGBoost training complete!")
    
    return pipeline


def train_voting_classifier(lr_model, svm_model, xgb_model, X_train, y_train) -> VotingClassifier:
    """Train Voting Classifier using pre-trained base models."""
    print("\n" + "="*60)
    print("🔧 Training Voting Classifier...")
    print("="*60)
    
    estimators = [
        ("lr", lr_model),
        ("svm", svm_model),
        ("xgb", xgb_model),
    ]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting="hard",  # Hard voting because LinearSVC doesn't output probabilities
        n_jobs=-1,
    )
    
    voting_clf.fit(X_train, y_train)
    print("✅ Voting Classifier training complete!")
    
    return voting_clf


def train_stacking_classifier(lr_model, svm_model, xgb_model, X_train, y_train) -> StackingClassifier:
    """Train Stacking Classifier using pre-trained base models."""
    print("\n" + "="*60)
    print("🔧 Training Stacking Classifier...")
    print("="*60)
    
    estimators = [
        ("lr", lr_model),
        ("svm", svm_model),
        ("xgb", xgb_model),
    ]
    
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=skf,
        n_jobs=-1,
    )
    
    stacking_clf.fit(X_train, y_train)
    print("✅ Stacking Classifier training complete!")
    
    return stacking_clf


def save_model(model, filepath: Path, model_name: str):
    """Save model to disk using joblib."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"💾 {model_name} saved to: {filepath}")


def main(args):
    """Main training function."""
    print("="*60)
    print("🚀 FAKE REVIEW DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Determine data path
    data_path = Path(args.data_path) if args.data_path else DATA_FILE
    
    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print(f"   Please ensure the data file exists or provide correct path with --data-path")
        sys.exit(1)
    
    # Load and prepare data
    print("\n📂 STEP 1: Loading and preparing data...")
    reviews = load_and_prepare_data(str(data_path))
    
    # Prepare features
    print("\n📊 STEP 2: Extracting features...")
    reviews = prepare_features(reviews)
    
    # Split features and target
    X = reviews.drop(columns=[TARGET_COLUMN])
    y = reviews[TARGET_COLUMN]
    
    # Train-test split
    print("\n✂️ STEP 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"   Training set size: {len(X_train)}")
    print(f"   Test set size: {len(X_test)}")
    
    # Create preprocessor
    print("\n🔧 STEP 4: Creating preprocessor...")
    preprocessor = create_preprocessor(
        text_column=TEXT_COLUMN,
        numeric_columns=NUMERIC_FEATURE_COLUMNS,
        tfidf_params=TFIDF_PARAMS,
    )
    
    # Determine which models to train
    models_to_train = args.model.split(",") if args.model else ["all"]
    train_all = "all" in models_to_train
    
    trained_models = {}
    
    # Train individual models
    print("\n🎯 STEP 5: Training models...")
    
    if train_all or "lr" in models_to_train or "logistic" in models_to_train:
        lr_model = train_logistic_regression(preprocessor, X_train, y_train)
        trained_models["logistic_regression"] = lr_model
        evaluate_model(lr_model, X_train, y_train, "Train (LR)")
        evaluate_model(lr_model, X_test, y_test, "Test (LR)")
    
    if train_all or "svm" in models_to_train:
        svm_model = train_svm(preprocessor, X_train, y_train)
        trained_models["svm"] = svm_model
        evaluate_model(svm_model, X_train, y_train, "Train (SVM)")
        evaluate_model(svm_model, X_test, y_test, "Test (SVM)")
    
    if train_all or "xgb" in models_to_train or "xgboost" in models_to_train:
        xgb_model = train_xgboost(preprocessor, X_train, y_train)
        trained_models["xgboost"] = xgb_model
        evaluate_model(xgb_model, X_train, y_train, "Train (XGBoost)")
        evaluate_model(xgb_model, X_test, y_test, "Test (XGBoost)")
    
    # Train ensemble models (require all base models)
    if train_all or "voting" in models_to_train or "stacking" in models_to_train:
        # Ensure base models are trained
        if "logistic_regression" not in trained_models:
            lr_model = train_logistic_regression(preprocessor, X_train, y_train)
            trained_models["logistic_regression"] = lr_model
        if "svm" not in trained_models:
            svm_model = train_svm(preprocessor, X_train, y_train)
            trained_models["svm"] = svm_model
        if "xgboost" not in trained_models:
            xgb_model = train_xgboost(preprocessor, X_train, y_train)
            trained_models["xgboost"] = xgb_model
        
        lr_model = trained_models["logistic_regression"]
        svm_model = trained_models["svm"]
        xgb_model = trained_models["xgboost"]
        
        if train_all or "voting" in models_to_train:
            voting_model = train_voting_classifier(lr_model, svm_model, xgb_model, X_train, y_train)
            trained_models["voting"] = voting_model
            evaluate_model(voting_model, X_train, y_train, "Train (Voting)")
            evaluate_model(voting_model, X_test, y_test, "Test (Voting)")
        
        if train_all or "stacking" in models_to_train:
            stacking_model = train_stacking_classifier(lr_model, svm_model, xgb_model, X_train, y_train)
            trained_models["stacking"] = stacking_model
            evaluate_model(stacking_model, X_train, y_train, "Train (Stacking)")
            evaluate_model(stacking_model, X_test, y_test, "Test (Stacking)")
    
    # Save models
    print("\n💾 STEP 6: Saving models...")
    
    model_paths = {
        "logistic_regression": MODELS_DIR / "logistic_regression_model.pkl",
        "svm": MODELS_DIR / "svm_model.pkl",
        "xgboost": MODELS_DIR / "xgboost_model.pkl",
        "voting": MODELS_DIR / "voting_classifier.pkl",
        "stacking": MODELS_DIR / "stacking_classifier.pkl",
    }
    
    for model_name, model in trained_models.items():
        save_model(model, model_paths[model_name], model_name)
    
    # Save the preprocessor separately for inference
    preprocessor_fitted = clone(preprocessor)
    preprocessor_fitted.fit(X_train)
    save_model(preprocessor_fitted, MODELS_DIR / "preprocessor.pkl", "Preprocessor")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📁 Models saved to: {MODELS_DIR}")
    print("\nSaved artifacts:")
    for model_name in trained_models:
        print(f"   ✅ {model_paths[model_name].name}")
    print(f"   ✅ preprocessor.pkl")
    
    return trained_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Fake Review Detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    Train all models:
        python scripts/train.py
    
    Train specific model:
        python scripts/train.py --model stacking
        python scripts/train.py --model lr,svm
    
    Use custom data path:
        python scripts/train.py --data-path data/raw/reviews.csv
        
Available models: lr, svm, xgb, voting, stacking, all
        """,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model(s) to train (comma-separated). Options: lr, svm, xgb, voting, stacking, all",
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the training data CSV file",
    )
    
    args = parser.parse_args()
    main(args)
