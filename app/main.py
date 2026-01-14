"""
FastAPI Application for Fake Review Detection.

This module provides a REST API for detecting fake reviews using
trained machine learning models.

Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from src.config import (
    MODELS_DIR,
    XGBOOST_MODEL_PATH,
    TEXT_COLUMN,
)
from src.preprocessing import (
    clean_text,
    remove_stopwords,
    lemmatize_text,
    extract_numeric_features,
    download_nltk_resources,
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    text: str = Field(
        ...,
        min_length=1,
        description="The review text to analyze",
        json_schema_extra={"example": "This product is absolutely amazing! Best purchase ever!"}
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    prediction: int = Field(
        ...,
        description="Predicted class (0=Real, 1=Fake)"
    )
    label: str = Field(
        ...,
        description="Human-readable prediction label"
    )
    text_preview: str = Field(
        ...,
        description="Preview of the analyzed text"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    model_name: str = Field(default="")


class WelcomeResponse(BaseModel):
    """Response model for root endpoint."""
    
    message: str
    version: str
    docs_url: str


# =============================================================================
# GLOBAL STATE
# =============================================================================


class ModelState:
    """Container for loaded model artifacts."""
    
    model: Optional[object] = None
    model_name: str = ""
    labels: dict = {0: "Real (Human-Written)", 1: "Fake (Computer-Generated)"}


model_state = ModelState()


# =============================================================================
# LIFESPAN CONTEXT MANAGER
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    
    Loads the model once at startup and cleans up at shutdown.
    """
    # Startup: Load model and resources
    print("🚀 Starting Fake Review Detection API...")
    
    # Download NLTK resources
    print("📥 Downloading NLTK resources...")
    download_nltk_resources()
    
    # Load the trained model
    model_path = XGBOOST_MODEL_PATH
    
    if not model_path.exists():
        print(f"⚠️  Warning: Model not found at {model_path}")
        print("   Run 'python scripts/train.py' to train and save models.")
    else:
        print(f"📂 Loading model from: {model_path}")
        model_state.model = joblib.load(model_path)
        model_state.model_name = "xgboost"
        print("✅ Model loaded successfully!")
    
    print("🟢 API is ready to serve requests!\n")
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    print("\n🔴 Shutting down API...")
    model_state.model = None
    print("✅ Cleanup complete!")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================


app = FastAPI(
    title="Fake Review Detection API",
    description=(
        "A REST API for detecting fake (computer-generated) reviews using "
        "machine learning. The model analyzes text patterns and linguistic "
        "features to classify reviews as Real or Fake."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def preprocess_text(text: str) -> str:
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


def prepare_input(text: str) -> pd.DataFrame:
    """
    Prepare input text for model prediction.
    
    Args:
        text: Raw review text
        
    Returns:
        DataFrame ready for model prediction
    """
    # Create DataFrame with raw text
    df = pd.DataFrame({TEXT_COLUMN: [text]})
    
    # Extract numeric features from RAW text (before cleaning)
    df = extract_numeric_features(df, text_column=TEXT_COLUMN)
    
    # Apply text preprocessing (after feature extraction)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(preprocess_text)
    
    return df


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get(
    "/",
    response_model=WelcomeResponse,
    summary="Root endpoint",
    tags=["General"],
)
async def root():
    """
    Welcome endpoint with API information.
    
    Returns basic information about the API and links to documentation.
    """
    return WelcomeResponse(
        message="Welcome to the Fake Review Detection API!",
        version="1.0.0",
        docs_url="/docs",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["General"],
)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and model loading state.
    Useful for container orchestration and load balancers.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=model_state.model is not None,
        model_name=model_state.model_name,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict if a review is fake",
    tags=["Prediction"],
)
async def predict(request: PredictionRequest):
    """
    Analyze a review and predict if it's fake or real.
    
    This endpoint takes a review text, processes it through the same
    preprocessing pipeline used during training, and returns a prediction.
    
    - **text**: The review text to analyze (required, non-empty string)
    
    Returns:
    - **prediction**: 0 for Real, 1 for Fake
    - **label**: Human-readable label
    - **text_preview**: Truncated preview of the input text
    """
    # Check if model is loaded
    if model_state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model file exists and restart the server.",
        )
    
    try:
        # Validate input
        text = request.text.strip()
        
        if not text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review text cannot be empty.",
            )
        
        # Prepare input for model
        df = prepare_input(text)
        
        # Make prediction
        prediction = model_state.model.predict(df)[0]
        prediction = int(prediction)
        
        # Create response
        return PredictionResponse(
            prediction=prediction,
            label=model_state.labels[prediction],
            text_preview=text[:100] + "..." if len(text) > 100 else text,
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
