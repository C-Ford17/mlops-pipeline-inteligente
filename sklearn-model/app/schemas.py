"""
Pydantic Schemas para ML Service
Purpose: Define modelos de request/response
Author: Christian
Date: 2025-01-17
"""

from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List

# Config para evitar warnings de pydantic
class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    prediction: int
    probabilities: List[float]
    confidence: float
    model_version: str

class TrainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    metrics: Dict[str, float]
    message: str

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    service: str

class InfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    service: str
    version: str
    model_type: str
    dataset: str
    status: str
