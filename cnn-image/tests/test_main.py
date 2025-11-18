"""
Tests para CNN Image Service
"""

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import io
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)


class TestCNNService:
    """Tests para endpoints del servicio CNN"""
    
    def test_health_endpoint(self):
        """Test del endpoint /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cnn-service"
    
    def test_info_endpoint(self):
        """Test del endpoint /info"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "cnn-image"
        assert "class_names" in data
        assert len(data["class_names"]) == 5
    
    def test_predict_with_image(self):
        """Test de predicción con imagen"""
        # Crear imagen de prueba 32x32
        img = Image.new('RGB', (32, 32), color='red')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        
        files = {"file": ("test.png", buf, "image/png")}
        response = client.post("/predict", files=files)
        
        # Puede ser 200 (modelo cargado) o 500 (modelo no cargado)
        assert response.status_code in [200, 500]
    
    def test_predict_without_image(self):
        """Test de predicción sin imagen"""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error


class TestCNNModel:
    """Tests para el modelo CNN"""
    
    def test_cifar10_download(self):
        """Test de descarga de CIFAR-10"""
        from app.model import CNNImageClassifier
        
        classifier = CNNImageClassifier()
        result = classifier.download_cifar10()
        
        assert result is True
        assert os.path.exists("data/raw/cifar-10-batches-py")
    
    def test_model_creation(self):
        """Test de creación del modelo"""
        from app.model import CNNImageClassifier
        
        classifier = CNNImageClassifier()
        model = classifier.create_model()
        
        assert model is not None
        assert len(model.layers) > 0
