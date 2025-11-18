# LLM Service

## Purpose
Microservicio para interactuar con modelos de lenguaje (LLM) mediante API REST.

## How to run locally
Install dependencies
pip install -r requirements.txt

Set environment variables
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.2

Run service
python -m src.main

## Environment Variables
- `LLM_PROVIDER`: Proveedor de LLM (ollama)
- `OLLAMA_BASE_URL`: URL base de Ollama
- `OLLAMA_MODEL`: Modelo a usar (llama3.2)

## Endpoints
### POST /chat
Request:
{
"prompt": "¿Qué es MLOps?",
"context": "Contexto opcional",
"max_tokens": 100,
"temperature": 0.7
}
Response:
{
"response": "MLOps es...",
"model": "llama3.2",
"tokens_used": 150
}

### GET /health
Health check del servicio.
