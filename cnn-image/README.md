# CNN Image Classification Service

Servicio de clasificación de imágenes usando CNN con dataset CIFAR-10 (5 clases).

## 🎯 Features

- **Arquitectura CNN** con 3 capas convolucionales
- **Filtros personalizados:** Smoothing, Edge Detection, Sharpness
- **Descarga automática** de CIFAR-10 al entrenar
- **MLflow tracking** de experimentos
- **API REST** con FastAPI
- **5 clases:** airplane, automobile, bird, cat, dog

## 🏗️ Arquitectura

Input Image (32x32 RGB)
│
[Filters] ← Smoothing, Edge Detection, Sharpness
│
[Conv2D 32] → BatchNorm → MaxPool → Dropout(0.3)
│
[Conv2D 64] → BatchNorm → MaxPool → Dropout(0.3)
│
[Conv2D 128] → BatchNorm → MaxPool → Dropout(0.3)
│
[Flatten]
│
[Dense 512] → Dropout(0.5)
│
[Dense 5] → Softmax
│
Output (5 classes)


## 📊 Dataset

- **CIFAR-10:** 60,000 imágenes 32x32 RGB
- **5 clases seleccionadas:**
  - 0: airplane
  - 1: automobile
  - 2: bird
  - 3: cat
  - 4: dog (mapeado desde CIFAR-10 class 5)
- **Descarga automática** vía Keras en primer entrenamiento

## 🚀 API Endpoints

### **POST /predict**
Clasifica una imagen.

**Request:**

curl -X POST http://localhost:8002/predict
-F "file=@image.jpg"


**Response:**

{
"class_id": 0,
"class_name": "airplane",
"confidence": 0.89,
"probabilities": [0.89, 0.05, 0.03, 0.02, 0.01],
"filters_applied": ["smoothing", "edge_detection", "sharpness"]
}


### **POST /train**
Entrena el modelo CNN.

**Request:**


curl -X POST http://localhost:8002/train
-H "Content-Type: application/json"
-d '{"epochs": 10}'


**Response:**

{
"status": "success",
"metrics": {
"accuracy": 0.82,
"val_accuracy": 0.78,
"loss": 0.45,
"val_loss": 0.52
},
"model_path": "s3://mlflow/artifacts/..."
}


### **GET /health**
Health check.


curl http://localhost:8002/health


### **GET /info**
Información del servicio.


curl http://localhost:8002/info


## 🔧 Setup

### **Local**


Instalar dependencias
pip install -r requirements.txt

Variables de entorno
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

Ejecutar
uvicorn app.main:app --host 0.0.0.0 --port 8000


### **Docker**


Build
docker build -t cnn-image:latest .

Run
docker run -p 8002:8000
-e MLFLOW_TRACKING_URI=http://mlflow-server:5000
-v $(pwd)/models:/app/models
cnn-image:latest


## 🧪 Testing


Instalar dependencias
pip install -r requirements.txt
pip install pytest pytest-cov httpx Pillow

Ejecutar tests
pytest tests/ -v

Tests rápidos (excluir descarga de datos)
pytest tests/ -v -m "not slow"

Con coverage
pytest tests/ -v --cov=app --cov-report=html


## 📁 Estructura

cnn-image/
├── app/
│ ├── main.py # API FastAPI
│ ├── model.py # Arquitectura CNN
│ └── schemas.py # Pydantic models
├── filters/
│ └── custom_filters.py # Filtros de convolución
├── tests/
│ └── test_main.py # Tests unitarios
├── data/
│ └── raw/ # Dataset (creado automáticamente)
├── models/ # Modelos entrenados
├── Dockerfile
├── requirements.txt
└── README.md


## 🎓 Entrenamiento

### Primera vez (descarga CIFAR-10):

curl -X POST http://localhost:8002/train

Descarga ~170MB + entrena 5-10 min


### Entrenamientos subsecuentes:


### Monitoreo con MLflow:

Abrir MLflow UI
open http://localhost:5000

Ver experimentos CNN
Ver métricas por epoch
Comparar versiones de modelos

Abrir MLflow UI
open http://localhost:5000

Ver experimentos CNN
Ver métricas por epoch
Comparar versiones de modelos


### Edge Detection (Sobel)


kernel = [[-1, -1, -1],
[-1, 8, -1],
[-1, -1, -1]]


### Sharpness

kernel = [[ 0, -1, 0],
[-1, 5, -1],
[ 0, -1, 0]]


## 🐛 Troubleshooting

### Error: CIFAR-10 no encontrado


El primer entrenamiento descarga automáticamente
Si falla, descarga manual:
python -c "from tensorflow.keras.datasets import cifar10; cifar10.load_data()"


### Error: OOM (Out of Memory)


Reducir batch size en pipeline/model.py
BATCH_SIZE = 32 # Cambiar a 16 o 8


### Error: CUDA not available


Normal en CPU, el modelo usa TensorFlow con CPU
Los warnings de CUDA son esperados


## 🚀 Performance Tips

1. **Primera ejecución:** Descarga CIFAR-10 (~2 min)
2. **GPU:** Mejora ~10x velocidad (requiere CUDA)
3. **Batch size:** Aumentar si tienes más RAM
4. **Epochs:** Más epochs = mejor accuracy (pero más lento)

## 📝 Autor

Christian Gomez - Proyecto Final MLOps
