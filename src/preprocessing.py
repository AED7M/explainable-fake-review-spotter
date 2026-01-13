"""
Preprocessing module for Fake Review Detection.
Contains all text cleaning, feature extraction, and transformation functions.
"""
import re
import string
import numpy as np
import pandas as pd
import nltk
import textstat
import contractions
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    SLANG_DICT,
    TFIDF_PARAMS,
    NUMERIC_FEATURE_COLUMNS,
    TEXT_COLUMN,
)


def download_nltk_resources():
    """Download required NLTK resources for text preprocessing."""
    resources = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "punkt_tab",
    ]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {resource}: {e}")


# Initialize NLTK resources
download_nltk_resources()

# Initialize stopwords set
STOP_WORDS = set(stopwords.words("english"))

# Initialize lemmatizer
LEMMATIZER = WordNetLemmatizer()


# =============================================================================
# TEXT CLEANING FUNCTIONS
# =============================================================================


def expand_slang(text: str) -> str:
    """
    Expand slang and abbreviations to their full forms.
    
    Args:
        text: Input text string
        
    Returns:
        Text with slang expanded
    """
    words = text.split()
    expanded = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(expanded)


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing noise and standardizing format.
    
    Operations performed:
    1. Remove HTML tags
    2. Remove URLs
    3. Remove email addresses
    4. Expand contractions
    5. Convert to lowercase
    6. Remove punctuation and numbers
    7. Normalize whitespace
    8. Expand slang
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and normalized text
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # Remove URLs
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        text,
    )
    # Remove email addresses
    text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text
    )
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase
    text = text.lower()
    # Remove punctuation & numbers, keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    # Normalize multiple spaces
    text = " ".join(text.split())
    # Expand slang/abbreviations
    text = expand_slang(text)
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords from text.
    
    Args:
        text: Input text string
        
    Returns:
        Text with stopwords removed
    """
    return " ".join([word for word in text.split() if word not in STOP_WORDS])


def get_wordnet_pos(tag: str):
    """
    Map POS tag to WordNet POS tag for lemmatization.
    
    Args:
        tag: NLTK POS tag
        
    Returns:
        WordNet POS constant
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text using POS-aware lemmatization.
    
    Args:
        text: Input text string
        
    Returns:
        Lemmatized text
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return " ".join(
        [LEMMATIZER.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    )


def preprocess_text(text: str) -> str:
    """
    Apply full text preprocessing pipeline.
    
    Pipeline:
    1. Clean text (remove noise, normalize)
    2. Remove stopwords
    3. Lemmatize
    
    Args:
        text: Raw input text
        
    Returns:
        Fully preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================


def extract_word_count(text: str) -> int:
    """Extract total number of words."""
    return len(text.split())


def extract_avg_word_length(text: str) -> float:
    """Extract average word length."""
    words = text.split()
    return np.mean([len(w) for w in words]) if words else 0


def extract_punctuation_count(text: str) -> int:
    """Extract total punctuation count."""
    return len([c for c in text if c in string.punctuation])


def extract_stopword_count(text: str) -> int:
    """Extract stopword count."""
    return len([w for w in text.split() if w.lower() in STOP_WORDS])


def extract_num_sentences(text: str) -> int:
    """Extract number of sentences."""
    return len(nltk.sent_tokenize(text))


def extract_uppercase_word_count(text: str) -> int:
    """Extract count of uppercase words (length > 1)."""
    return len([w for w in text.split() if w.isupper() and len(w) > 1])


def extract_special_char_count(text: str) -> int:
    """Extract count of special characters."""
    special_chars = "!@#$%^&*()_+=<>?{}[]|"
    return len([c for c in text if c in special_chars])


def extract_unique_word_ratio(text: str) -> float:
    """Extract lexical diversity (unique words / total words)."""
    words = text.split()
    return len(set(words)) / len(words) if words else 0


def extract_flesch_reading_ease(text: str) -> float:
    """Extract Flesch Reading Ease score."""
    return textstat.flesch_reading_ease(text) if len(text) > 0 else 0


def extract_flesch_kincaid_grade(text: str) -> float:
    """Extract Flesch-Kincaid Grade Level."""
    return textstat.flesch_kincaid_grade(text) if len(text) > 0 else 0


def extract_numeric_features(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """
    Extract all numeric features from raw text before cleaning.
    
    Args:
        df: DataFrame containing the text column
        text_column: Name of the text column
        
    Returns:
        DataFrame with added numeric feature columns
    """
    df = df.copy()
    
    df["word_count"] = df[text_column].apply(extract_word_count)
    df["avg_word_length"] = df[text_column].apply(extract_avg_word_length)
    df["punctuation_count"] = df[text_column].apply(extract_punctuation_count)
    df["stopword_count"] = df[text_column].apply(extract_stopword_count)
    df["num_sentences"] = df[text_column].apply(extract_num_sentences)
    df["uppercase_word_count"] = df[text_column].apply(extract_uppercase_word_count)
    df["special_char_count"] = df[text_column].apply(extract_special_char_count)
    df["unique_word_ratio"] = df[text_column].apply(extract_unique_word_ratio)
    df["flesch_reading_ease"] = df[text_column].apply(extract_flesch_reading_ease)
    df["flesch_kincaid_grade"] = df[text_column].apply(extract_flesch_kincaid_grade)
    
    return df


# =============================================================================
# SKLEARN-COMPATIBLE PREPROCESSOR
# =============================================================================


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible text preprocessing transformer.
    Applies the full text preprocessing pipeline to a text column.
    """
    
    def __init__(self, text_column: str = "text"):
        self.text_column = text_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        print("🧹 Cleaning text...")
        X[self.text_column] = X[self.text_column].apply(clean_text)
        print("🔍 Removing stopwords...")
        X[self.text_column] = X[self.text_column].apply(remove_stopwords)
        print("🔄 Lemmatizing text...")
        X[self.text_column] = X[self.text_column].apply(lemmatize_text)
        print("✅ Text preprocessing complete!")
        return X


def create_preprocessor(
    text_column: str = TEXT_COLUMN,
    numeric_columns: list = NUMERIC_FEATURE_COLUMNS,
    tfidf_params: dict = None,
) -> ColumnTransformer:
    """
    Create the sklearn ColumnTransformer preprocessor.
    
    Combines:
    - TF-IDF vectorization for text
    - Power transformation for numeric features
    
    Args:
        text_column: Name of the text column
        numeric_columns: List of numeric feature column names
        tfidf_params: Dictionary of TF-IDF parameters (uses defaults if None)
        
    Returns:
        Configured ColumnTransformer
    """
    if tfidf_params is None:
        tfidf_params = TFIDF_PARAMS
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(**tfidf_params), text_column),
            ("num", PowerTransformer(), numeric_columns),
        ]
    )
    
    return preprocessor


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Load raw data and perform initial preparation.
    
    Operations:
    1. Load CSV file
    2. Drop irrelevant columns (category, rating)
    3. Rename text_ to text
    4. Remove duplicates
    5. Create binary target variable
    
    Args:
        file_path: Path to the CSV data file
        
    Returns:
        Prepared DataFrame
    """
    print(f"📂 Loading data from {file_path}...")
    reviews = pd.read_csv(file_path)
    
    print(f"📌 Initial dataset shape: {reviews.shape}")
    
    # Drop irrelevant columns
    reviews.drop(columns=["category", "rating"], inplace=True, errors="ignore")
    
    # Rename text column
    if "text_" in reviews.columns:
        reviews.rename(columns={"text_": "text"}, inplace=True)
    
    # Remove duplicates
    initial_count = len(reviews)
    reviews = reviews.drop_duplicates(subset="text", keep="first")
    print(f"🔍 Duplicates removed: {initial_count - len(reviews)}")
    
    # Create binary target variable (CG = 1, OR = 0)
    reviews["target"] = np.where(reviews["label"] == "CG", 1, 0)
    reviews.drop(columns=["label"], inplace=True)
    
    print(f"📊 Final dataset shape: {reviews.shape}")
    print(f"📊 Target distribution:\n{reviews['target'].value_counts()}")
    
    return reviews


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare all features from raw data.
    
    1. Extract numeric features from raw text
    2. Apply text preprocessing pipeline
    
    Args:
        df: Raw DataFrame with text column
        
    Returns:
        DataFrame with all features prepared
    """
    # Extract numeric features BEFORE text cleaning
    print("\n📊 Extracting numeric features...")
    df = extract_numeric_features(df, text_column="text")
    
    # Apply text preprocessing
    print("\n🔄 Applying text preprocessing pipeline...")
    df["text"] = df["text"].apply(clean_text)
    df["text"] = df["text"].apply(remove_stopwords)
    df["text"] = df["text"].apply(lemmatize_text)
    
    print("✅ Feature preparation complete!")
    return df
