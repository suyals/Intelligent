from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class ComplaintRequest(BaseModel):
    complaint: str = Field(..., min_length=2, max_length=2000, description="The student's complaint or query text")

class BatchComplaintRequest(BaseModel):
    complaints: List[str] = Field(..., min_items=1, max_items=100, description="List of complaint texts")

class PredictionResult(BaseModel):
    complaint: str
    cleaned_complaint: str
    predicted_category: str
    category_confidence: float
    category_probabilities: Dict[str, float]
    predicted_priority: str
    priority_confidence: float
    priority_probabilities: Dict[str, float]
    contributing_keywords: List[str]
    explanation: str
    category_explanation: str
    priority_explanation: str
    suggested_action: str
    resolution_sla: str

class BatchPredictionResponse(BaseModel):
    total: int
    predictions: List[PredictionResult]

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    vocabulary_size: int
    timestamp: str

class SampleQuery(BaseModel):
    id: int
    query: str
    expected_category: str
    expected_priority: str
    description: str

class AnalyticsResponse(BaseModel):
    dataset_summary: Dict[str, Any]
    model_comparison: Dict[str, Any]
    final_evaluations: Dict[str, Any]
    confusion_matrices: Dict[str, Any]

