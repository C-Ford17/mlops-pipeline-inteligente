"""
ML Service API
Purpose: FastAPI endpoints para entrenamiento y predicción
Author: Christian Gomez
Date: 2025-01-17
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import (
    PredictionRequest, PredictionResponse, TrainResponse,
    HealthResponse, InfoResponse
)
from pipeline.trainer import MLTrainer
from pipeline.predictor import MLPredictor
from pydantic import BaseModel
import sys
import pydantic
import mlflow

# FIX para Pydantic v2
pydantic.BaseModel.model_config = pydantic.ConfigDict(protected_namespaces=())

# Configurar MLflow ANTES que cualquier otra cosa
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)
logger.info(f"MLflow tracking: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

# Inicializar componentes ANTES de lifespan
trainer = MLTrainer()
predictor = MLPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager para FastAPI (reemplaza on_event)
    """
    logger.info("🚀 Sklearn Model Service iniciando...")
    try:
        predictor.load_model()
        logger.info("✅ Modelo cargado exitosamente")
    except FileNotFoundError as e:
        logger.warning(f"⚠️ {e}")
    except Exception as e:
        logger.error(f"❌ Error crítico al cargar modelo: {e}")
        # NO hacer sys.exit() - permite que el contenedor siga corriendo
    
    yield  # Servicio está corriendo
    
    logger.info("🛑 Sklearn Model Service apagando...")

# Crear app con lifespan
app = FastAPI(title="Sklearn ML Service", version="1.0.0", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Endpoint de predicción
    """
    logger.info(f"Recibida predicción con features: {list(request.features.keys())}")
    
    try:
        result = predictor.predict(request.features)
        return result
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainResponse)
async def train():
    """
    Endpoint de entrenamiento
    """
    logger.info("Iniciando entrenamiento")
    
    try:
        metrics = trainer.train()
        # Recargar modelo después de entrenar
        predictor.load_model()
        return {
            "status": "success",
            "metrics": metrics,
            "message": "Modelo entrenado y registrado en MLflow"
        }
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # CORREGIDO: Usar predictor.pipeline, NO predictor.model
    is_model_loaded = predictor.pipeline is not None
    return {
        "status": "healthy" if is_model_loaded else "unhealthy", 
        "service": "sklearn-model",
        "model_loaded": is_model_loaded
    }

@app.get("/info", response_model=InfoResponse)
async def info():
    # CORREGIDO: Usar predictor.pipeline, NO predictor.model
    is_model_loaded = predictor.pipeline is not None
    return {
        "service": "sklearn-model",
        "version": "1.0.0",
        "model_type": "RandomForestClassifier",
        "dataset": "Titanic",
        "status": "operational" if is_model_loaded else "model_not_loaded"
    }
