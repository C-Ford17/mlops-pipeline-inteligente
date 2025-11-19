"""
LLM Service API - CON MLFLOW LOGGING
Purpose: Provides a REST API for interacting with LLM models with full MLflow tracking
Author: Christian Gomez
Date: 2025-01-17
"""

import os
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import requests
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
# Load environment variables
load_dotenv()

# Configurar MLflow para usar MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)
logger.info(f"MLflow tracking: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

app = FastAPI(title="LLM Service", version="1.0.0")


class LLMRequest(BaseModel):
    prompt: str
    context: str = ""
    max_tokens: int = 500
    temperature: float = 0.7


class LLMResponse(BaseModel):
    response: str
    model: str
    tokens_used: int


# CLIENTE GLOBAL
client = None


def init_google_client():
    """Inicializa cliente Google Gemini"""
    global client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY no encontrada")
    
    client = genai.Client(api_key=api_key)
    logger.info("[SUCCESS] Google Gemini client inicializado")


@app.on_event("startup")
async def startup_event():
    try:
        init_google_client()
    except Exception as e:
        logger.error(f"[ERROR] No se pudo inicializar Google: {e}")


def log_to_mlflow(prompt: str, context: str, response: str, model: str, tokens_used: int, 
                 temperature: float, max_tokens: int, latency_ms: float) -> None:
    """
    Log interacción completa a MLflow
    """
    try:
        # --- CREAR O USAR EXPERIMENTO EXISTENTE ---
        mlflow.set_experiment("llm_interactions")
        # Crear run único para cada interacción
        with mlflow.start_run(run_name=f"llm_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parámetros
            mlflow.log_params({
                "model": model,
                "temperature": temperature,
                "max_tokens_requested": max_tokens,
                "tokens_used": tokens_used,
                "has_context": len(context) > 0
            })
            
            # Log métricas
            mlflow.log_metrics({
                "tokens_used": tokens_used,
                "latency_ms": latency_ms,
                "prompt_length": len(prompt),
                "response_length": len(response)
            })
            
            # Crear artefacto con toda la conversación
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "parameters": {
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                "prompt": prompt,
                "context": context,
                "response": response,
                "tokens_used": tokens_used,
                "latency_ms": round(latency_ms, 2)
            }
            
            # Guardar como JSON
            artifact_file = "conversation.json"
            with open(artifact_file, "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            
            # Log artefacto
            mlflow.log_artifact(artifact_file)
            
            # Log tags
            mlflow.set_tags({
                "model_provider": "google",
                "model_version": model,
                "interaction_type": "chat"
            })
            
            logger.info(f"[SUCCESS] Interacción loggeada en MLflow: {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"Error loggeando a MLflow: {e}")


def query_google_with_retry(prompt: str, context: str, max_tokens: int, temperature: float, retries: int = 3) -> Dict[str, Any]:
    """Query Google Gemini con retry y MLflow logging"""
    if not client:
        raise HTTPException(status_code=500, detail="Google client no inicializado")
    
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    
    valid_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro", 
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite"
    ]
    
    if model not in valid_models:
        logger.warning(f"Modelo {model} no válido. Usando gemini-2.5-flash")
        model = "gemini-2.5-flash"
    
    if context:
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
    else:
        full_prompt = prompt
    
    max_tokens = max(max_tokens, 300)
    
    # Medir latencia
    start_time = time.time()
    
    for attempt in range(retries):
        try:
            logger.info(f"Intentando con {model} (intento {attempt + 1})")
            
            response = client.models.generate_content(
                model=model,
                contents=full_prompt
            )
            
            if not response or not hasattr(response, 'text'):
                raise ValueError("Respuesta inválida")
            
            if not response.text:
                # Fallback a candidates
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if (hasattr(candidate, 'content') and candidate.content and 
                        hasattr(candidate.content, 'parts') and candidate.content.parts):
                        text = candidate.content.parts[0].text
                        logger.info("[SUCCESS] Texto recuperado de candidates")
                        response.text = text
            
            if not response.text:
                raise ValueError("Respuesta vacía después de intentar recuperar")
            
            # Calcular métricas
            tokens_used = len(full_prompt.split()) + len(response.text.split())
            latency_ms = (time.time() - start_time) * 1000
            
            # LOG A MLFLOW
            log_to_mlflow(
                prompt=prompt,
                context=context,
                response=response.text,
                model=model,
                tokens_used=tokens_used,
                temperature=temperature,
                max_tokens=max_tokens,
                latency_ms=latency_ms
            )
            
            return {
                "response": response.text,
                "model": model,
                "tokens_used": tokens_used
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # MANEJO ESPECÍFICO PARA 503
            if "503" in error_msg and attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"503 error, intento {attempt + 1}/{retries}. Esperando {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            # Último intento
            else:
                logger.error(f"Error final: {error_msg}")
                raise HTTPException(status_code=503, detail=f"Google API error: {error_msg}")


def query_google(prompt: str, context: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    return query_google_with_retry(prompt, context, max_tokens, temperature)


def query_ollama(prompt: str, context: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    """Query Ollama local como fallback"""
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    
    full_prompt = f"Context: {context}\n\nQuestion: {prompt}" if context else prompt
    
    # Medir latencia
    start_time = time.time()
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    
    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        tokens_used = result.get("eval_count", 0)
        latency_ms = (time.time() - start_time) * 1000
        
        # LOG A MLFLOW (también para Ollama)
        log_to_mlflow(
            prompt=prompt,
            context=context,
            response=result["response"],
            model=model,
            tokens_used=tokens_used,
            temperature=temperature,
            max_tokens=max_tokens,
            latency_ms=latency_ms
        )
        
        return {
            "response": result["response"],
            "model": model,
            "tokens_used": tokens_used
        }
    except Exception as e:
        logger.error(f"Error en Ollama: {e}")
        raise


def get_llm_response(prompt: str, context: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "google")
    
    try:
        if provider == "google":
            return query_google_with_retry(prompt, context, max_tokens, temperature)
        elif provider == "ollama":
            return query_ollama(prompt, context, max_tokens, temperature)
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")
    except Exception as e:
        # FALLBACK AUTOMÁTICO
        if provider == "google":
            logger.warning(f"Google falló, intentando Ollama: {e}")
            try:
                return query_ollama(prompt, context, max_tokens, temperature)
            except:
                pass
        
        logger.error(f"LLM query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


@app.post("/chat", response_model=LLMResponse)
async def chat(request: LLMRequest):
    logger.info(f"Received chat request with prompt: {request.prompt[:50]}...")
    
    try:
        if request.max_tokens < 300:
            request.max_tokens = 300
            
        result = get_llm_response(
            prompt=request.prompt,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        logger.info("[SUCCESS] LLM response generated and logged to MLflow")
        return result
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "llm-service"}


@app.get("/info")
async def info():
    provider = os.getenv("LLM_PROVIDER", "google")
    model = os.getenv("GOOGLE_MODEL") if provider == "google" else os.getenv("OLLAMA_MODEL")
    return {
        "service": "llm-service",
        "version": "1.0.0",
        "provider": provider,
        "model": model,
        "mlflow_logging": "enabled"
    }
