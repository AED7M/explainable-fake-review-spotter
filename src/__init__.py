"""
Fake Review Detection - Source Package

This package contains the core modules for the fake review detection ML pipeline.
"""

from src.config import (
    DATA_FILE,
    MODELS_DIR,
    RANDOM_STATE,
    TFIDF_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    SVM_PARAMS,
    XGBOOST_PARAMS,
)

from src.preprocessing import (
    load_and_prepare_data,
    prepare_features,
    create_preprocessor,
    preprocess_text,
    clean_text,
    remove_stopwords,
    lemmatize_text,
)

__all__ = [
    # Config
    "DATA_FILE",
    "MODELS_DIR",
    "RANDOM_STATE",
    "TFIDF_PARAMS",
    "LOGISTIC_REGRESSION_PARAMS",
    "SVM_PARAMS",
    "XGBOOST_PARAMS",
    # Preprocessing
    "load_and_prepare_data",
    "prepare_features",
    "create_preprocessor",
    "preprocess_text",
    "clean_text",
    "remove_stopwords",
    "lemmatize_text",
]
