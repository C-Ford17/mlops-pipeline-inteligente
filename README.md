# Pipeline MLOps Inteligente

Sistema de Machine Learning e Inteligencia Artificial con integración de:
- LLM (Gemini 2.5)
- CNN (CIFAR-10)
- Sklearn (Titanic)
- MLflow + MinIO
- Gradio Frontend

## Arquitectura
- **llm-connector**: Servicio de LLM con Google Gemini
- **cnn-image**: Clasificador de imágenes CNN
- **sklearn-model**: Modelo de ML tradicional
- **mlflow-server**: Tracking y registro de modelos
- **minio**: Storage S3 para artifacts
- **gradio-frontend**: Interfaz web unificada

## Tecnologías
- Python 3.10
- FastAPI
- TensorFlow
- Scikit-learn
- Docker + Docker Compose
- MLflow
- MinIO