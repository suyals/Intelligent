import os
import datetime
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    ComplaintRequest,
    BatchComplaintRequest,
    PredictionResult,
    BatchPredictionResponse,
    HealthResponse,
    AnalyticsResponse,
    SampleQuery
)
from backend.model import ModelManager

# Initialize FastAPI App
app = FastAPI(
    title="Intelligent Student Complaint Classification & Priority Prediction API",
    description="Automated NLP and ML system for student complaint categorization and urgency routing.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Manager immediately at startup
try:
    model_manager = ModelManager(models_dir="models")
    print("Backend: ML models, vectorizer, and metadata loaded successfully.")
except Exception as e:
    model_manager = None
    print(f"Backend Warning: Failed to load models on startup: {e}")

# Predefined representative sample queries from the dataset
SAMPLE_QUERIES: List[SampleQuery] = [
    SampleQuery(
        id=1,
        query="Internship deadline today, need NOC approval.",
        expected_category="Registrar & Student Records",
        expected_priority="High",
        description="Critical time-sensitive academic clearance query"
    ),
    SampleQuery(
        id=2,
        query="My fee payment failed but amount deducted.",
        expected_category="Finance & Accounts",
        expected_priority="High",
        description="High urgency banking & fee transaction grievance"
    ),
    SampleQuery(
        id=3,
        query="I cannot submit my exam form and today is the last date.",
        expected_category="Academics & Examination",
        expected_priority="High",
        description="Critical examination deadline issue"
    ),
    SampleQuery(
        id=4,
        query="How to apply for revaluation?",
        expected_category="Academics & Examination",
        expected_priority="Medium",
        description="Post-result evaluation procedure query"
    ),
    SampleQuery(
        id=5,
        query="How can I apply for hostel change?",
        expected_category="Hostel & Housing",
        expected_priority="Medium",
        description="Residential room transfer request"
    ),
    SampleQuery(
        id=6,
        query="Please update my contact number.",
        expected_category="Registrar & Student Records",
        expected_priority="Medium",
        description="Student record information update"
    ),
    SampleQuery(
        id=7,
        query="WiFi password for campus?",
        expected_category="IT & Technical Support",
        expected_priority="Low",
        description="Standard campus IT credential lookup"
    ),
    SampleQuery(
        id=8,
        query="What are library timings?",
        expected_category="Campus Facilities & Transport",
        expected_priority="Low",
        description="Campus facility informational inquiry"
    ),
    SampleQuery(
        id=9,
        query="How to join sports club?",
        expected_category="Student Affairs & Clubs",
        expected_priority="Low",
        description="Extracurricular student activity query"
    ),
]

# Health check endpoints
@app.get("/health", response_model=HealthResponse)
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    global model_manager
    is_loaded = model_manager is not None and model_manager.loaded
    vocab_len = len(model_manager.vectorizer.vocabulary_) if is_loaded and model_manager.vectorizer else 0
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        version="1.0.0",
        models_loaded=is_loaded,
        vocabulary_size=vocab_len,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    )

# Prediction endpoints
@app.post("/predict", response_model=PredictionResult)
@app.post("/api/predict", response_model=PredictionResult)
def predict_complaint(payload: ComplaintRequest):
    global model_manager
    if model_manager is None or not model_manager.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning models are not yet loaded. Please run 'python scripts/train.py'."
        )
    
    text = payload.complaint.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Complaint text cannot be empty."
        )

    try:
        result = model_manager.predict(text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict-batch", response_model=BatchPredictionResponse)
@app.post("/api/predict-batch", response_model=BatchPredictionResponse)
def predict_batch_complaints(payload: BatchComplaintRequest):
    global model_manager
    if model_manager is None or not model_manager.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning models are not yet loaded."
        )

    predictions = []
    for text in payload.complaints:
        if text and text.strip():
            predictions.append(model_manager.predict(text.strip()))
        
    return BatchPredictionResponse(
        total=len(predictions),
        predictions=predictions
    )

# Analytics endpoint
@app.get("/analytics", response_model=AnalyticsResponse)
@app.get("/api/analytics", response_model=AnalyticsResponse)
def get_analytics():
    global model_manager
    if model_manager is None or model_manager.model_metrics is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model analytics not found. Ensure the model has been trained."
        )
    return model_manager.model_metrics

# Sample queries endpoint
@app.get("/sample-queries", response_model=List[SampleQuery])
@app.get("/api/sample-queries", response_model=List[SampleQuery])
def get_sample_queries():
    return SAMPLE_QUERIES

# Serve Frontend static files if available
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend not yet built. Visit /docs for Swagger API documentation."})
