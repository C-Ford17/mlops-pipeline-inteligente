# Pipeline MLOps Inteligente

Sistema integrado de Machine Learning e Inteligencia Artificial con tres componentes principales:
- **LLM Service**: Conversación con Google Gemini 2.5 Flash
- **ML Service**: Predicción de supervivencia Titanic (RandomForest)
- **CNN Service**: Clasificación de imágenes CIFAR-10 (5 clases)

## Arquitectura

\`\`\`
┌─────────────────┐
│ Gradio Frontend │
│   (Port 7860)   │
└────────┬────────┘
         │
    ┌────┴────┬─────────┬──────────┐
    │         │         │          │
┌───▼──┐  ┌──▼───┐  ┌──▼────┐  ┌──▼──────┐
│ LLM  │  │  ML  │  │  CNN  │  │ MLflow  │
│ :8000│  │ :8001│  │ :8002 │  │  :5000  │
└──────┘  └──────┘  └───────┘  └────┬────┘
                                     │
                                ┌────▼────┐
                                │  MinIO  │
                                │  :9000  │
                                └─────────┘
\`\`\`

## Tecnologías

- **Backend**: Python 3.10, FastAPI, Uvicorn
- **ML/DL**: Scikit-learn, TensorFlow/Keras
- **LLM**: Google Gemini 2.5 Flash
- **Frontend**: Gradio
- **MLOps**: MLflow, MinIO (S3)
- **Orquestación**: Docker Swarm
- **CI/CD**: GitHub Actions

## Quick Start

\`\`\`bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/mlops-pipeline-inteligente.git
cd mlops-pipeline-inteligente

# 2. Configurar variables de entorno
cp infra/.env.example infra/.env
# Editar .env con tu GOOGLE_API_KEY

# 3. Inicializar Swarm
docker swarm init

# 4. Build de imágenes
cd infra
.\build-local.ps1

# 5. Deploy del stack
.\deploy-stack.ps1

# 6. Acceder a servicios
# Gradio UI:    http://localhost:7860
# MLflow UI:    http://localhost:5000
# MinIO Console: http://localhost:9001
\`\`\`

## Tests

\`\`\`bash
# Ejecutar todos los tests
.\run-tests.ps1

# Tests individuales
cd llm-connector && pytest tests/ -v
cd sklearn-model && pytest tests/ -v
cd cnn-image && pytest tests/ -v
cd gradio-frontend && pytest tests/ -v
\`\`\`

## Estructura del Proyecto

\`\`\`
mlops-pipeline-inteligente/
├── llm-connector/          # Servicio LLM (Gemini)
├── sklearn-model/          # Servicio ML (Titanic)
├── cnn-image/              # Servicio CNN (CIFAR-10)
├── gradio-frontend/        # Interfaz web
├── mlflow-server/          # Servidor MLflow
├── infra/                  # Docker Compose/Swarm
└── .github/workflows/      # CI/CD
\`\`\`

## Autor

Christian Gomez - Proyecto Final MLOps
"@ | Out-File -FilePath README.md -Encoding utf8

# Crear .env.example
Write-Host "📝 Creando .env.example..." -ForegroundColor Yellow
@"
# Variables de entorno para MLOps Pipeline

# Google Gemini API
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash

# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# MinIO
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_DEFAULT_REGION=us-east-1
"@ | Out-File -FilePath infra/.env.example -Encoding utf8

# Commit inicial
Write-Host "`n✅ Commit 1: Initial commit" -ForegroundColor Green
git add .gitignore README.md infra/.env.example
git commit -m "chore: initial commit with project structure"

# Crear branch develop
git checkout -b develop