"""
CNN Service API
Purpose: FastAPI endpoints for image classification
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
import logging
from filters.custom_filters import ConvolutionFilters
from app.model import CNNImageClassifier
from app.schemas import PredictionResponse, HealthResponse, InfoResponse

# Configurar logging AL INICIO
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CNN Service", version="1.0.0")
classifier = CNNImageClassifier()
filters = ConvolutionFilters()

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Validación y procesamiento de imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Archivo debe ser una imagen")
        
        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Aplicar filtros
        filtered_image = filters.apply_filters(image_array)
        
        # Predicción
        result = classifier.predict(filtered_image)
        
        # Agregar información de filtros
        result["filters_applied"] = filters.get_filter_info()["filters_applied"]
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {"status": "healthy", "service": "cnn-service"}

@app.post("/train")
async def train(epochs: int = 10):
    """
    Entrena modelo con CIFAR-10
    """
    logger.info(f"Entrenando CNN con {epochs} epochs")
    try:
        metrics = classifier.train(epochs=epochs)
        return {
            "status": "success",
            "metrics": metrics,
            "message": "CNN entrenado con CIFAR-10"
        }
    except Exception as e:
        logger.error(f"Error en entrenamiento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Agregar información de dataset a /info
@app.get("/info", response_model=InfoResponse)
async def info():
    return {
        "service": "cnn-image",
        "version": "1.0.0",
        "input_shape": classifier.input_shape,
        "num_classes": classifier.num_classes,
        "class_names": classifier.class_names,
        "filters_applied": ["smoothing", "edge_detection", "sharpness"],
        "dataset": "CIFAR-10 (5 clases)",
        "status": "operational"
    }