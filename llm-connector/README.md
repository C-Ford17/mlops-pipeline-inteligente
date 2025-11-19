# LLM Connector Service

Servicio de conexión con Google Gemini 2.5 Flash para procesamiento de lenguaje natural.

## Features

- **Google Gemini 2.5 Flash** integrado
- **Fallback a Ollama** (LLM local) si Gemini falla
- **MLflow tracking** de todas las interacciones
- **API REST** con FastAPI
- **Streaming support** (preparado para futuras versiones)
- **Rate limiting** y retry logic

## Architecture

Request → FastAPI → LLM Router
│
┌───────┴───────┐
│ │
[Google Gemini] [Ollama Local]
│ │
└───────┬───────┘
│
MLflow Logger
│
Response


## API Endpoints

### **POST /chat**
Envía un prompt al LLM y obtiene respuesta.

**Request:**


curl -X POST http://localhost:8000/chat
-H "Content-Type: application/json"
-d '{
"prompt": "¿Qué es MLOps?",
"context": "",
"max_tokens": 150,
"temperature": 0.7
}'


**Response:**


{
"response": "MLOps es la práctica de combinar Machine Learning con DevOps...",
"model": "gemini-2.5-flash",
"tokens_used": 95,
"processing_time_ms": 1234,
"provider": "google"
}


**Parámetros:**

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `prompt` | string | (required) | Pregunta o instrucción |
| `context` | string | "" | Contexto adicional |
| `max_tokens` | int | 500 | Máximo de tokens en respuesta |
| `temperature` | float | 0.7 | Creatividad (0.0-1.0) |

### **GET /health**
Health check del servicio.

curl http://localhost:8000/health

{"status": "healthy", "service": "llm-service"}


### **GET /info**
Información del servicio.

curl http://localhost:8000/info


**Response:**


{
"service": "llm-service",
"version": "1.0.0",
"provider": "google",
"model": "gemini-2.5-flash",
"fallback": "ollama",
"mlflow_enabled": true
}


## Setup

### **Local**


Instalar dependencias
pip install -r requirements.txt

Variables de entorno
export GOOGLE_API_KEY=your_api_key_here
export GOOGLE_MODEL=gemini-2.5-flash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

Ejecutar
uvicorn app.main:app --host 0.0.0.0 --port 8000


### **Docker**


Build
docker build -t llm-connector:latest .

Run
docker run -p 8000:8000
-e GOOGLE_API_KEY=your_key
-e MLFLOW_TRACKING_URI=http://mlflow-server:5000
llm-connector:latest


## Configuration

### Environment Variables

| Variable | Descripción | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | API Key de Google Gemini | (required) |
| `GOOGLE_MODEL` | Modelo a usar | `gemini-2.5-flash` |
| `LLM_PROVIDER` | Provider principal | `google` |
| `MLFLOW_TRACKING_URI` | URL de MLflow | `http://mlflow-server:5000` |
| `MLFLOW_S3_ENDPOINT_URL` | URL de MinIO | `http://minio:9000` |
| `OLLAMA_HOST` | Host de Ollama (fallback) | `http://host.docker.internal:11434` |

## Testing


Instalar dependencias de test
pip install -r requirements.txt

Ejecutar tests
pytest tests/ -v

Con coverage
pytest tests/ -v --cov=app --cov-report=html


## Project Structure

llm-connector/
├── app/
│ └── main.py # API FastAPI
├── tests/
│ └── test_main.py # Tests unitarios
├── Dockerfile
├── requirements.txt
└── README.md


## LLM Providers

### Google Gemini (Primary)
- **Model:** gemini-2.5-flash
- **Max tokens:** 8192
- **Características:**
  - Respuestas rápidas (1-3 segundos)
  - Multilenguaje
  - Context window grande
  - Grounding con búsqueda Google

### Ollama (Fallback)
- **Model:** llama2 (configurable)
- **Local:** Sin necesidad de API key
- **Ventajas:**
  - Privacidad total
  - Sin límites de rate
  - Offline capability
- **Desventajas:**
  - Requiere GPU para velocidad óptima
  - Modelos más pequeños

## MLflow Integration

### Tracked Metrics
- **Prompt:** Texto de entrada
- **Response:** Texto de salida
- **Model:** Modelo usado
- **Tokens:** Cantidad de tokens
- **Latency:** Tiempo de respuesta
- **Provider:** google/ollama
- **Temperature:** Configuración
- **Success:** True/False

### Viewing Logs


Abrir MLflow UI
open http://localhost:5000

Filtrar por servicio
Experiment: "llm-connector"
Ver métricas por run
Ver artifacts (prompts/responses guardados)


## Examples

### Chat básico

import requests

response = requests.post(
"http://localhost:8000/chat",
json={
"prompt": "Explica Machine Learning en 50 palabras",
"max_tokens": 100,
"temperature": 0.5
}
)

print(response.json()["response"])


### Chat con contexto

response = requests.post(
"http://localhost:8000/chat",
json={
"prompt": "¿Cómo se entrena?",
"context": "Estamos hablando de modelos RandomForest",
"max_tokens": 200,
"temperature": 0.7
}
)


### Ajustar creatividad

Respuesta más determinista (temperature = 0.1)
response = requests.post(
"http://localhost:8000/chat",
json={
"prompt": "¿Cuánto es 2+2?",
"temperature": 0.1 # Más preciso
}
)

Respuesta más creativa (temperature = 0.9)
response = requests.post(
"http://localhost:8000/chat",
json={
"prompt": "Escribe un poema sobre MLOps",
"temperature": 0.9 # Más creativo
}
)


## Error Handling

### Error: GOOGLE_API_KEY no configurada

Verificar variable
echo $GOOGLE_API_KEY

Configurar
export GOOGLE_API_KEY=your_key_here


### Error: Rate limit exceeded
El servicio automáticamente:
1. Espera con exponential backoff
2. Reintenta hasta 3 veces
3. Si falla, intenta con Ollama

### Error: Ollama not available


Instalar Ollama (si quieres fallback local)
curl https://ollama.ai/install.sh | sh

Descargar modelo
ollama pull llama2

Verificar
curl http://localhost:11434/api/tags


## Performance

### Latency típica
- **Gemini 2.5 Flash:** 1-3 segundos
- **Ollama (CPU):** 5-10 segundos
- **Ollama (GPU):** 1-2 segundos

### Throughput
- **Concurrent requests:** 10+ (con async)
- **Rate limit Google:** ~60 requests/min (free tier)
- **Ollama:** Sin límites (hardware dependent)

## Troubleshooting

### Gemini siempre falla


Verificar API key
curl https://generativelanguage.googleapis.com/v1/models?key=YOUR_KEY

Verificar logs
docker logs llm-connector-container


### Response vacía

Aumentar timeout en código
O usar modelo más rápido (gemini-flash)


## Security

### API Key Protection
- **Nunca** commitear API keys
- Usar variables de entorno
- En producción: usar Docker Secrets


Docker Swarm Secrets
echo "your-api-key" | docker secret create google_api_key -

Usar en stack.yml:
secrets:

google_api_key


### Rate Limiting
- Implementar en proxy (nginx/traefik)
- O usar Redis para tracking
- Configurar en API Gateway

## Roadmap

- [ ] Streaming responses (SSE)
- [ ] Multiple LLM providers (Anthropic, OpenAI)
- [ ] Response caching (Redis)
- [ ] Cost tracking per user
- [ ] Fine-tuned models support

## Author

Christian Gomez - Proyecto Final MLOps


