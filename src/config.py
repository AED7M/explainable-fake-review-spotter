"""
Configuration file for Fake Review Detection ML Pipeline.
Contains all hyperparameters, file paths, and constants.
"""
import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models" / "baseline"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data file path
DATA_FILE = DATA_DIR / "fake reviews dataset.csv"

# Model artifact paths
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
VOTING_MODEL_PATH = MODELS_DIR / "voting_classifier.pkl"
STACKING_MODEL_PATH = MODELS_DIR / "stacking_classifier.pkl"

# =============================================================================
# RANDOM STATE (for reproducibility)
# =============================================================================
RANDOM_STATE = 42

# =============================================================================
# DATA SPLIT CONFIGURATION
# =============================================================================
TEST_SIZE = 0.2

# =============================================================================
# FEATURE COLUMNS
# =============================================================================
TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
LABEL_COLUMN = "label"
COLUMNS_TO_DROP = ["category", "rating"]

NUMERIC_FEATURE_COLUMNS = [
    "word_count",
    "avg_word_length",
    "punctuation_count",
    "stopword_count",
    "num_sentences",
    "uppercase_word_count",
    "special_char_count",
    "unique_word_ratio",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]

# =============================================================================
# SLANG DICTIONARY (for text normalization)
# =============================================================================
SLANG_DICT = {
    "urself": "yourself",
    "rly": "really",
    "rlly": "really",
    "srsly": "seriously",
    "afaik": "as far as i know",
    "luv": "love",
    "gr8": "great",
    "gud": "good",
    "b4": "before",
    "yea": "yeah",
    "ya": "you",
    "tho": "though",
    "btw": "by the way",
    "bcz": "because",
    "bc": "because",
    "cuz": "because",
    "coz": "because",
    "ima": "i am",
    "imma": "i am",
    "idk": "i do not know",
    "idc": "i do not care",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "thanx": "thanks",
    "u": "you",
    "ur": "your",
    "4u": "for you",
    "4me": "for me",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "ppl": "people",
    "pkg": "package",
    "perf": "perfect",
    "rec": "recommend",
    "recd": "received",
    "def": "definitely",
    "prob": "problem",
    "info": "information",
    "cust": "customer",
    "qty": "quantity",
    "amz": "amazing",
    "amazin": "amazing",
    "fave": "favorite",
    "fav": "favorite",
    "awsm": "awesome",
    "fab": "fantastic",
    "lit": "excellent",
    "legit": "genuine",
    "omg": "oh my god",
    "dam": "damn",
    "meh": "not good",
    "ugh": "frustrating",
    "gah": "annoying",
    "didnt": "did not",
    "doesnt": "does not",
    "dont": "do not",
    "cant": "cannot",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "wasnt": "was not",
    "werent": "were not",
    "isnt": "is not",
    "arent": "are not",
    "ive": "i have",
    "ill": "i will",
    "its": "it is",
    "aint": "is not",
    "wont": "will not",
    "btwn": "between",
    "thru": "through",
    "atm": "at the moment",
    "tbh": "to be honest",
    "rn": "right now",
    "ty": "thank you",
    "np": "no problem",
    "smh": "shaking my head",
    "asap": "as soon as possible",
}

# =============================================================================
# TF-IDF VECTORIZER HYPERPARAMETERS (Best from GridSearch)
# =============================================================================
TFIDF_PARAMS = {
    "lowercase": False,  # Already lowercased during preprocessing
    "stop_words": None,  # Already removed stopwords
    "ngram_range": (1, 2),
    "max_df": 0.7,
    "min_df": 0.0002,
    "max_features": 4000,
    "sublinear_tf": True,
    "norm": "l2",
    "smooth_idf": True,
}

# =============================================================================
# LOGISTIC REGRESSION HYPERPARAMETERS (Best from GridSearch)
# =============================================================================
LOGISTIC_REGRESSION_PARAMS = {
    "C": 0.75,
    "penalty": "l2",
    "solver": "liblinear",
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

# =============================================================================
# SVM (LinearSVC) HYPERPARAMETERS (Best from GridSearch)
# =============================================================================
SVM_PARAMS = {
    "C": 0.1,
    "loss": "squared_hinge",
    "max_iter": 4000,
    "tol": 1e-4,
    "dual": False,
    "random_state": RANDOM_STATE,
}

# =============================================================================
# XGBOOST HYPERPARAMETERS (Best from RandomizedSearchCV + Early Stopping)
# =============================================================================
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 350,  # Determined via early stopping
    "max_depth": 6,
    "min_child_weight": 5,
    "learning_rate": 0.08,
    "gamma": 1.5,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "colsample_bylevel": 0.7,
    "colsample_bynode": 0.7,
    "reg_lambda": 3.0,
    "reg_alpha": 1.0,
    "scale_pos_weight": 1.0,
    "max_delta_step": 1,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================
CV_FOLDS = 5
