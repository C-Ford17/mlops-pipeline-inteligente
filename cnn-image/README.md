# CNN Service

## Purpose
Microservicio para clasificación de imágenes con filtros de convolución personalizados (suavizado, detección de bordes, nitidez).

## How to run locally
Install dependencies
pip install -r requirements.txt

Train model (genera datos sintéticos)
python -m src.model

Run API
python -m src.main

## Environment Variables
- `MLFLOW_TRACKING_URI`: URI del servidor MLflow

## Endpoints

### POST /predict
Upload an image file (JPEG, PNG) for classification.

Response:
{
"predicted_class": 0,
"class_name": "class_0",
"confidence": 0.85,
"probabilities": [0.85, 0.1, 0.05],
"limitations": {
"note": "This model uses synthetic data for demonstration",
"accuracy": "Limited to 3 classes with synthetic patterns",
"robustness": "Not suitable for production use"
}
}

### POST /train?epochs=10
Entrena el modelo CNN con datos sintéticos.

### GET /health
Health check del servicio.

## Filtros Aplicados
1. **Suavizado**: Filtro de promedio 3x3
2. **Detección de bordes**: Kernel Sobel 3x3
3. **Nitidez**: Filtro de realce 3x3

## Limitaciones
- Modelo entrenado con datos sintéticos
- 3 clases máximo
- No apto para producción sin dataset real



