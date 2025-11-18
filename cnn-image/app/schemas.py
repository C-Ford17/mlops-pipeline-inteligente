"""
Pydantic Schemas for CNN Service
Purpose: Define request/response models
"""

from pydantic import BaseModel
from typing import Dict, Any, List

class PredictionResponse(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: List[float]
    limitations: Dict[str, Any]
    filters_applied: List[Dict[str, Any]] = None  # Información adicional
    
class HealthResponse(BaseModel):
    status: str
    service: str
    
class InfoResponse(BaseModel):
    service: str
    version: str
    input_shape: tuple
    num_classes: int
    class_names: List[str]
    filters_applied: List[str]
