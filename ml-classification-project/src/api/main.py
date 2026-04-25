"""
FastAPI Backend for Iris Classification
========================================

This module creates a REST API to serve ML predictions.

Key Learning Points:
- FastAPI basics (routes, request/response)
- Pydantic for data validation
- Dependency injection
- API documentation (automatic with FastAPI)
- Error handling

Run this server:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

API Docs available at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ml.predict import IrisPredictor


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================
"""
STUDY NOTE: Pydantic Models
---------------------------
Pydantic provides data validation using Python type hints.

Benefits:
1. Automatic request validation
2. Clear API documentation
3. Type safety
4. Serialization/deserialization

The Field() function adds:
- Descriptions (shown in docs)
- Constraints (min, max, etc.)
- Examples
"""


class IrisFeatures(BaseModel):
    """
    Input features for iris classification.

    STUDY NOTE:
    -----------
    This model defines what the API expects in the request body.
    FastAPI automatically:
    - Validates incoming JSON against this schema
    - Returns 422 error if validation fails
    - Generates OpenAPI documentation
    """
    sepal_length: float = Field(
        ...,  # ... means required
        description="Length of the sepal in centimeters",
        ge=0.0,  # greater than or equal to 0
        le=10.0,  # less than or equal to 10
        examples=[5.1]
    )
    sepal_width: float = Field(
        ...,
        description="Width of the sepal in centimeters",
        ge=0.0,
        le=10.0,
        examples=[3.5]
    )
    petal_length: float = Field(
        ...,
        description="Length of the petal in centimeters",
        ge=0.0,
        le=10.0,
        examples=[1.4]
    )
    petal_width: float = Field(
        ...,
        description="Width of the petal in centimeters",
        ge=0.0,
        le=10.0,
        examples=[0.2]
    )

    # Custom validator example
    @field_validator('sepal_length', 'sepal_width', 'petal_length', 'petal_width')
    @classmethod
    def check_positive(cls, v: float) -> float:
        """Ensure all measurements are positive."""
        if v < 0:
            raise ValueError('Measurements must be positive')
        return v

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """
    Response model for predictions.

    STUDY NOTE:
    -----------
    Defining response models:
    - Documents what the API returns
    - Ensures consistent response format
    - Enables response validation
    """
    predicted_class: str = Field(
        ...,
        description="Predicted iris species",
        examples=["setosa"]
    )
    predicted_label: int = Field(
        ...,
        description="Numeric label (0=setosa, 1=versicolor, 2=virginica)",
        examples=[0]
    )
    confidence: float = Field(
        ...,
        description="Prediction confidence (0-1)",
        ge=0.0,
        le=1.0,
        examples=[0.97]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability for each class"
    )
    input_features: Dict[str, float] = Field(
        ...,
        description="Input features used for prediction"
    )


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    samples: List[IrisFeatures] = Field(
        ...,
        description="List of samples to predict",
        min_length=1,
        max_length=100  # Limit batch size
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    count: int = Field(..., description="Number of predictions made")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    model_loaded: bool
    model_name: Optional[str] = None
    version: Optional[str] = None


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    version: str
    feature_names: List[str]
    target_names: List[str]


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
"""
STUDY NOTE: FastAPI Application
-------------------------------
FastAPI is a modern, fast web framework for building APIs.

Key features:
- Automatic OpenAPI documentation
- Type hints for validation
- Async support
- Dependency injection
- High performance (based on Starlette and Pydantic)
"""

# Create FastAPI app instance
app = FastAPI(
    title="Iris Classification API",
    description="""
    ## ML-Powered Iris Flower Classification

    This API provides predictions for iris flower species based on
    sepal and petal measurements.

    ### Features
    - Single prediction endpoint
    - Batch prediction endpoint
    - Model information
    - Health checks

    ### Models
    The API uses a trained scikit-learn model (Logistic Regression or Random Forest)
    to classify iris flowers into three species:
    - **Setosa**
    - **Versicolor**
    - **Virginica**
    """,
    version="1.0.0",
    contact={
        "name": "ML Project",
        "email": "ml@example.com"
    },
    license_info={
        "name": "MIT"
    }
)


# =============================================================================
# CORS MIDDLEWARE
# =============================================================================
"""
STUDY NOTE: CORS (Cross-Origin Resource Sharing)
------------------------------------------------
CORS is a security feature that controls which domains can access your API.

Without CORS:
- Browser blocks requests from different domains
- Your Streamlit frontend couldn't call the API

With CORS configured:
- Specified origins can make requests
- Browser allows the communication
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================
"""
STUDY NOTE: Dependency Injection
--------------------------------
Instead of loading the model in every request, we load it once
and inject it where needed.

Benefits:
- Model loaded only once (at startup)
- Faster request handling
- Easier testing (can mock dependencies)
"""

# Global predictor instance (loaded once)
predictor: Optional[IrisPredictor] = None


def get_predictor() -> IrisPredictor:
    """
    Dependency that provides the predictor instance.

    STUDY NOTE:
    -----------
    This function is called by FastAPI when an endpoint
    needs the predictor. We use a global instance to avoid
    reloading the model on every request.
    """
    global predictor
    if predictor is None:
        try:
            predictor = IrisPredictor()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model not loaded: {str(e)}"
            )
    return predictor


# =============================================================================
# STARTUP EVENT
# =============================================================================
"""
STUDY NOTE: Startup Events
--------------------------
FastAPI allows running code when the application starts.
We use this to:
- Pre-load the model (warm start)
- Validate model exists
- Log startup information
"""


@app.on_event("startup")
async def startup_event():
    """Load model on startup for faster first request."""
    global predictor
    print("\n" + "=" * 50)
    print("Starting Iris Classification API...")
    print("=" * 50)

    try:
        predictor = IrisPredictor()
        info = predictor.get_model_info()
        print(f"✓ Model loaded: {info['model_name']} v{info['version']}")
        print(f"✓ API ready at http://localhost:8000")
        print(f"✓ Docs at http://localhost:8000/docs")
        print("=" * 50 + "\n")
    except FileNotFoundError:
        print("⚠ Warning: Model not found. Run train.py first.")
        print("  API will return 503 until model is available.")
        print("=" * 50 + "\n")


# =============================================================================
# API ENDPOINTS
# =============================================================================
"""
STUDY NOTE: HTTP Methods
------------------------
- GET: Retrieve data (no body, idempotent)
- POST: Create/submit data (has body)
- PUT: Update data (replace entirely)
- PATCH: Update data (partial)
- DELETE: Remove data

For ML predictions, we use POST because:
- We're sending data in the request body
- The operation may not be idempotent (stateful models)
"""


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API welcome message.

    STUDY NOTE:
    -----------
    Good practice to have a root endpoint that confirms
    the API is running and provides basic info.
    """
    return {
        "message": "Welcome to Iris Classification API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    STUDY NOTE:
    -----------
    Health endpoints are essential for:
    - Kubernetes liveness/readiness probes
    - Load balancer health checks
    - Monitoring systems

    Returns:
    - status: "healthy" or "unhealthy"
    - model_loaded: whether the model is ready
    - timestamp: current server time
    """
    global predictor

    model_loaded = predictor is not None
    model_info = predictor.get_model_info() if model_loaded else {}

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_name=model_info.get('model_name'),
        version=model_info.get('version')
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.

    Returns model metadata including:
    - Model name and version
    - Expected feature names
    - Target class names
    """
    pred = get_predictor()
    info = pred.get_model_info()

    return ModelInfoResponse(
        model_name=info['model_name'],
        version=info['version'],
        feature_names=info['feature_names'],
        target_names=info['target_names']
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(features: IrisFeatures):
    """
    Make a single prediction.

    STUDY NOTE:
    -----------
    This is the main prediction endpoint.

    Flow:
    1. FastAPI validates request body against IrisFeatures schema
    2. If valid, our code runs
    3. We call the predictor
    4. FastAPI validates response against PredictionResponse
    5. Response sent to client

    If validation fails at any step, FastAPI returns appropriate error.

    **Example Request:**
    ```json
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    ```

    **Example Response:**
    ```json
    {
        "predicted_class": "setosa",
        "predicted_label": 0,
        "confidence": 0.97,
        "probabilities": {
            "setosa": 0.97,
            "versicolor": 0.02,
            "virginica": 0.01
        }
    }
    ```
    """
    pred = get_predictor()

    try:
        result = pred.predict(
            sepal_length=features.sepal_length,
            sepal_width=features.sepal_width,
            petal_length=features.petal_length,
            petal_width=features.petal_width
        )
        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make predictions for multiple samples.

    STUDY NOTE:
    -----------
    Batch endpoints are useful for:
    - Processing multiple items efficiently
    - Reducing HTTP overhead
    - Bulk operations

    Limits:
    - Max 100 samples per request (prevents abuse)

    **Example Request:**
    ```json
    {
        "samples": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
        ]
    }
    ```
    """
    pred = get_predictor()

    try:
        predictions = []
        for sample in request.samples:
            result = pred.predict(
                sepal_length=sample.sepal_length,
                sepal_width=sample.sepal_width,
                petal_length=sample.petal_length,
                petal_width=sample.petal_width
            )
            predictions.append(PredictionResponse(**result))

        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# =============================================================================
# ERROR HANDLERS
# =============================================================================
"""
STUDY NOTE: Error Handling
--------------------------
Good APIs provide clear error messages.
FastAPI handles most errors automatically, but we can customize.
"""


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.

    STUDY NOTE:
    -----------
    Catches any exception not handled elsewhere.
    In production, you might want to:
    - Log the error
    - Send alerts
    - Return generic message (hide internal details)
    """
    return {
        "error": "Internal server error",
        "detail": str(exc) if app.debug else "An unexpected error occurred"
    }


# =============================================================================
# MAIN (for direct execution)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 50)
    print("Running Iris Classification API")
    print("=" * 50)
    print("\nEndpoints:")
    print("  GET  /         - Welcome message")
    print("  GET  /health   - Health check")
    print("  GET  /model/info - Model information")
    print("  POST /predict  - Single prediction")
    print("  POST /predict/batch - Batch predictions")
    print("\nDocs: http://localhost:8000/docs")
    print("=" * 50 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
