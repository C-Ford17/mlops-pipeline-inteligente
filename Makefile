# MLOps Pipeline Project Makefile
# Author: Tu Nombre
# Date: 2025-01-17

.PHONY: help install build up down logs test train clean deploy-swarm

# Colors
GREEN = \033[0;32m
BLUE = \033[0;34m
YELLOW = \033[1;33m
RESET = \033[0m

help: ## Muestra esta ayuda
	@echo "$(BLUE)MLOps Pipeline Inteligente - Comandos Disponibles:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Instala dependencias locales para desarrollo
	@echo "$(YELLOW)Instalando dependencias...$(RESET)"
	pip install -r llm-service/requirements.txt
	pip install -r ml-service/requirements.txt
	pip install -r cnn-service/requirements.txt
	pip install -r gradio-app/requirements.txt
	pip install docker-compose

build: ## Construye todas las imágenes Docker
	@echo "$(YELLOW)Construyendo imágenes Docker...$(RESET)"
	docker-compose build

up: ## Levanta el entorno de desarrollo con Docker Compose
	@echo "$(GREEN)Iniciando entorno de desarrollo...$(RESET)"
	docker-compose up -d
	@echo "$(BLUE)Servicios disponibles:$(RESET)"
	@echo "  Gradio UI: http://localhost:7860"
	@echo "  MLflow: http://localhost:5000"
	@echo "  MinIO Console: http://localhost:9001 (user: minioadmin, pass: minioadmin)"

down: ## Detiene el entorno de desarrollo
	@echo "$(YELLOW)Deteniendo servicios...$(RESET)"
	docker-compose down

logs: ## Muestra logs de todos los servicios
	docker-compose logs -f

test: ## Ejecuta tests unitarios en cada servicio
	@echo "$(YELLOW)Ejecutando tests...$(RESET)"
	cd llm-service && python -m pytest tests/
	cd ml-service && python -m pytest tests/
	cd cnn-service && python -m pytest tests/

train: ## Entrena modelos ML y CNN
	@echo "$(BLUE)Entrenando modelos...$(RESET)"
	# ML Service
	curl -X POST http://localhost:8001/train || echo "ML Service no disponible"
	# CNN Service
	curl -X POST http://localhost:8002/train?epochs=5 || echo "CNN Service no disponible"

validate-structure: ## Valida estructura del proyecto
	@echo "$(YELLOW)Validando estructura...$(RESET)"
	@bash -c '\
		for dir in llm-service ml-service cnn-service gradio-app; do \
			if [ ! -d "$$dir/src" ]; then echo "❌ $$dir/src no existe"; exit 1; fi; \
			if [ ! -f "$$dir/Dockerfile" ]; then echo "❌ $$dir/Dockerfile no existe"; exit 1; fi; \
			if [ ! -f "$$dir/requirements.txt" ]; then echo "❌ $$dir/requirements.txt no existe"; exit 1; fi; \
			echo "✅ $$dir estructura válida"; \
		done'
	@echo "$(GREEN)Estructura validada correctamente$(RESET)"

clean: ## Limpia contenedores, volúmenes y redes no usadas
	@echo "$(YELLOW)Limpiando...$(RESET)"
	docker-compose down -v
	docker system prune -f

deploy-swarm: build ## Despliega en Docker Swarm
	@echo "$(BLUE)Desplegando en Docker Swarm...$(RESET)"
	docker stack deploy -c stack.yml mlops-pipeline

status: ## Muestra estado de servicios
	@echo "$(BLUE)Estado de servicios:$(RESET)"
	docker-compose ps

# CI/CD helpers
ci-test: validate-structure test ## Pipeline de CI: validación y tests
	@echo "$(GREEN)CI Pipeline completado$(RESET)"

ci-build: ci-test build ## Pipeline de CI/CD: tests + build
	@echo "$(GREEN)Build Pipeline completado$(RESET)"
