"""
Tests para Gradio Frontend
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGradioApp:
    """Tests para la aplicación Gradio"""
    
    def test_environment_variables(self):
        """Test de variables de entorno"""
        from app import LLM_SERVICE_URL, ML_SERVICE_URL, CNN_SERVICE_URL
        
        assert "llm-connector" in LLM_SERVICE_URL or "localhost" in LLM_SERVICE_URL
        assert "sklearn-model" in ML_SERVICE_URL or "localhost" in ML_SERVICE_URL
        assert "cnn-image" in CNN_SERVICE_URL or "localhost" in CNN_SERVICE_URL
    
    def test_function_definitions(self):
        """Test de que las funciones existen"""
        from app import chat_with_llm, predict_ml, train_ml, predict_cnn, train_cnn
        
        assert callable(chat_with_llm)
        assert callable(predict_ml)
        assert callable(train_ml)
        assert callable(predict_cnn)
        assert callable(train_cnn)
