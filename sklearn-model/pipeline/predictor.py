import os
import logging
import pickle
import pandas as pd
from typing import Dict, Any
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(self, model_name: str = "titanic_survival_classifier"):
        self.model_name = model_name
        self.pipeline = None
        self.label_encoders = {}
        # Features que el modelo espera (en el orden correcto)
        self.feature_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        
    def load_model(self):
        """Carga modelo y encoders desde disco"""
        logger.info(f"Intentando cargar modelo {self.model_name}...")
        
        model_path = f"/app/models/{self.model_name}.pkl"
        encoder_path = f"/app/models/{self.model_name}_encoders.pkl"
        
        # Verificar que el modelo existe ANTES de intentar cargar
        if not os.path.exists(model_path):
            logger.error(f"❌ Modelo NO encontrado en {model_path}")
            raise FileNotFoundError(f"Modelo no encontrado en {model_path}. Ejecute /train primero.")
        
        # Cargar pipeline
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
        logger.info(f"✅ Pipeline cargado correctamente")
        
        # Cargar encoders (opcional - pueden no existir si el modelo no los necesita)
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                self.label_encoders = pickle.load(f)
            logger.info(f"✅ Encoders cargados: {list(self.label_encoders.keys())}")
        else:
            logger.info("ℹ️ No se encontraron encoders (no es necesario)")
       
    
    def validate_features(self, features: Dict[str, float]) -> None:
        """Valida que todas las features requeridas estén presentes y sean números"""
        # Usar feature_columns, NO required_features (que está vacío)
        missing = [f for f in self.feature_columns if f not in features]
        if missing:
            raise ValueError(f"❌ Features faltantes: {missing}")
        
        # Validar que todos los valores son numéricos
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"❌ Feature '{name}' debe ser numérico, recibió {type(value)}")
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Realiza predicción con el modelo cargado"""
        # VERIFICAR QUE EL MODELO ESTÁ CARGADO
        if self.pipeline is None:
            logger.error("🚨 Modelo no cargado. Intentando cargar...")
            try:
                self.load_model()
            except FileNotFoundError as e:
                raise HTTPException(status_code=503, detail=str(e))
        
        # VALIDAR FEATURES
        self.validate_features(features)
        
        # Convertir a DataFrame con columnas en el ORDEN correcto (CRÍTICO)
        # El modelo espera las columnas en el mismo orden que fue entrenado
        X = pd.DataFrame([features], columns=self.feature_columns)
        logger.info(f"DataFrame creado con shape: {X.shape}")
        logger.info(f"Columnas: {list(X.columns)}")
        logger.info(f"Valores: {X.values[0]}")
        
        # Predicción directa (el pipeline ya incluye scaling)
        try:
            prediction = self.pipeline.predict(X)[0]
            probabilities = self.pipeline.predict_proba(X)[0]
            
            logger.info(f"✅ Predicción: {prediction}, confianza: {max(probabilities):.2%}")
            
            return {
                "prediction": int(prediction),
                "probabilities": probabilities.tolist(),
                "confidence": float(max(probabilities)),
                "model_version": self.model_name
            }
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise HTTPException(status_code=500, detail=f"Error en predicción: {e}")
