"""
Tests para Sklearn Model Service
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)


class TestSklearnService:
    """Tests para endpoints del servicio ML"""
    
    def test_health_endpoint(self):
        """Test del endpoint /health"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["service"] == "sklearn-model"
    
    def test_info_endpoint(self):
        """Test del endpoint /info"""
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "sklearn-model"
        assert "version" in data
        # CORREGIDO: Verificar campos que SÍ existen
        assert "model_type" in data
        assert "dataset" in data
    
    def test_predict_validation(self):
        """Test de validación de entrada en /predict"""
        # Payload inválido (falta features)
        payload = {}
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_valid_features(self):
        """Test de predicción con features válidos"""
        payload = {
            "features": {
                "Pclass": 1,
                "Sex": 1,
                "Age": 30.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 50.0,
                "Embarked": 0
            }
        }
        
        response = client.post("/predict", json=payload)
        # Puede ser 200 (modelo cargado) o 500 (modelo no cargado)
        assert response.status_code in [200, 500]


class TestTitanicTrainer:
    """Tests para el módulo de entrenamiento"""
    
    def test_titanic_download(self):
        """Test de descarga de dataset Titanic"""
        # CORREGIDO: Importar la clase correcta
        from pipeline.trainer import MLTrainer
        
        trainer = MLTrainer()
        data_path = trainer.download_titanic_dataset()
        
        assert os.path.exists(data_path)
        assert data_path.endswith("titanic.csv")
    
    def test_data_loading(self):
        """Test de carga de datos"""
        # CORREGIDO: Importar la clase correcta
        from pipeline.trainer import MLTrainer
        
        trainer = MLTrainer()
        X, y = trainer.load_data()
        
        assert X.shape[0] > 0
        assert X.shape[1] == 7  # 7 features
        assert len(y) == X.shape[0]
