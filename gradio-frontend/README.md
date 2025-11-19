# Gradio Frontend - MLOps Pipeline

Interfaz web unificada para interactuar con los servicios de Machine Learning e Inteligencia Artificial.

## [TARGET] Características

- **Interface unificada** para 3 servicios ML/AI
- **Tabs organizadas** por funcionalidad
- **Feedback en tiempo real** con indicadores de estado
- **Validación de entradas** y manejo de errores
- **Responsive design** con Gradio Themes

##  Arquitectura

┌─────────────────────────────────────┐
│ Gradio Frontend (7860) │
│ Tabs: LLM | ML Titanic | CNN │
└──────┬──────────┬──────────┬────────┘
│ │ │
┌───▼──┐ ┌──▼───┐ ┌───▼────┐
│ LLM │ │ ML │ │ CNN │
│:8000 │ │:8001 │ │ :8002 │
└──────┘ └──────┘ └────────┘


##  Componentes

### **Tab 1: LLM Chat**
- Conversación con Google Gemini 2.5 Flash
- Control de temperatura y max_tokens
- Contexto opcional para prompts
- Contador de tokens usados

### **Tab 2: ML Titanic**
- **Predicción:** Supervivencia con features del Titanic
- **Entrenamiento:** Train modelo RandomForest desde UI
- Métricas de rendimiento en tiempo real
- Versioning de modelos con MLflow

### **Tab 3: CNN Visión**
- **Predicción:** Clasificación de imágenes CIFAR-10
- **Entrenamiento:** Train CNN desde UI (5-10 min)
- 5 clases: airplane, automobile, bird, cat, dog
- Upload de imágenes con preview

## [STARTING] Quick Start

### **Local (sin Docker)**

Instalar dependencias
pip install -r requirements.txt

Configurar variables de entorno
export LLM_CONNECTOR_URL=http://localhost:8000
export SKLEARN_MODEL_URL=http://localhost:8001
export CNN_IMAGE_URL=http://localhost:8002

Ejecutar
python app/main.py

Acceder: http://localhost:7860


### **Con Docker**


Build
docker build -t gradio-frontend:latest .

Run
docker run -p 7860:7860
-e LLM_CONNECTOR_URL=http://llm-connector:8000
-e SKLEARN_MODEL_URL=http://sklearn-model:8000
-e CNN_IMAGE_URL=http://cnn-image:8000
gradio-frontend:latest


##  Testing

Instalar dependencias de test
pip install requirements

Ejecutar tests
pytest tests/ -v

Con coverage
pytest tests/ -v --cov=app --cov-report=html


##  Estructura


gradio-frontend/
├── app/
│ └── main.py # Aplicación Gradio principal
├── tests/
│ └── test_app.py # Tests unitarios
├── Dockerfile # Imagen Docker
├── requirements.txt # Dependencias Python
└── README.md # Este archivo


##  Configuración

### Variables de Entorno

| Variable | Descripción | Default |
|----------|-------------|---------|
| `LLM_CONNECTOR_URL` | URL del servicio LLM | `http://llm-connector:8000` |
| `SKLEARN_MODEL_URL` | URL del servicio ML | `http://sklearn-model:8000` |
| `CNN_IMAGE_URL` | URL del servicio CNN | `http://cnn-image:8000` |
| `GRADIO_SERVER_PORT` | Puerto del servidor | `7860` |

## [METRICS] Funcionalidades por Tab

### LLM Chat
- **Input:** Prompt + contexto opcional
- **Output:** Respuesta del LLM + metadata
- **Parámetros:** Temperature (0.1-1.0), Max Tokens (50-500)

### ML Titanic
- **Predicción:**
  - Inputs: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
  - Output: Supervivencia (Sí/No) + Probabilidades
- **Entrenamiento:**
  - Click "Entrenar Modelo"
  - Ver métricas: Accuracy, Precision, Recall, F1
  - Modelo se guarda en MLflow

### CNN Visión
- **Predicción:**
  - Upload imagen (cualquier tamaño, se redimensiona a 32x32)
  - Output: Clase predicha + Confianza
- **Entrenamiento:**
  - Click "Entrenar CNN"
  - 10 epochs (5-10 minutos)
  - Ver métricas de entrenamiento

## Troubleshooting

### Error: Connection refused


Verificar que los servicios backend estén corriendo
curl http://localhost:8000/health # LLM
curl http://localhost:8001/health # Sklearn
curl http://localhost:8002/health # CNN


### Error: ModuleNotFoundError

Reinstalar dependencias
pip install -r requirements.txt


## Contribuir

1. Tests obligatorios para nuevas funcionalidades
2. Seguir convenciones de código (black, flake8)
3. Documentar cambios en README

## Autor

Christian Gomez - Proyecto Final MLOps
