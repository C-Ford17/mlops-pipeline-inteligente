# Gradio Frontend

## Purpose
Interfaz web unificada para interactuar con los tres servicios: LLM, ML Clásico y CNN. Proporciona una experiencia de usuario intuitiva con tabs separados para cada funcionalidad.

## How to run locally

Install dependencies
pip install -r requirements.txt

Set environment variables
export LLM_SERVICE_URL=http://localhost:8000
export ML_SERVICE_URLhttp://localhost:8001
export CNN_SERVICE_URLhttp://localhost:8002

Run application
python -m src.app


## Environment Variables
- `LLM_SERVICE_URL`: URL del servicio LLM
- `ML_SERVICE_URL`: URL del servicio ML
- `CNN_SERVICE_URL`: URL del servicio CNN
- `GRADIO_SERVER_PORT`: Puerto para la interfaz (default: 7860)

## Features
- **Chat LLM**: Interfaz conversacional con ajuste de temperatura y máximos tokens
- **ML Predictor**: Sliders para características de flores Iris con probabilidades en tiempo real
- **CNN Classifier**: Upload de imágenes con visualización de filtros aplicados y limitaciones

## Communication
La app se comunica con los servicios backend vía HTTP/REST usando la librería `requests`.

## Limitaciones Mostradas
El servicio CNN incluye información explícita sobre las limitaciones del modelo para gestionar expectativas del usuario.
