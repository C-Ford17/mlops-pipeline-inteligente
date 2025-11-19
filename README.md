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

## Video presentando
https://drive.google.com/file/d/19OR7y6u8vxvgnWzC2s4OAAA3RomeLfwU/view?usp=drive_link

## Autor

Christian Gomez - Proyecto Final MLOps