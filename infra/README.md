# Infrastructure - MLOps Pipeline

Configuración de infraestructura Docker para orquestación con Compose y Swarm.

## 🏗️ Arquitectura


┌──────────────────────────────────────────────────┐
│ Gradio Frontend (7860) │
│ Interface unificada para todos los servicios │
└────────┬──────────┬──────────┬─────────────┬────┘
│ │ │ │
┌────▼───┐ ┌───▼────┐ ┌───▼─────┐ ┌───▼────────┐
│ LLM │ │ ML │ │ CNN │ │ MLflow │
│ :8000 │ │ :8001 │ │ :8002 │ │ :5000 │
│ Gemini │ │Titanic │ │CIFAR-10 │ │ Tracking │
└────────┘ └────────┘ └─────────┘ └──────┬─────┘
│
┌────▼────┐
│ MinIO │
│ :9000 │
│ S3 Store│
└─────────┘


## 📦 Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| **gradio-frontend** | 7860 | Interface web unificada |
| **llm-connector** | 8000 | Servicio LLM (Gemini 2.5) |
| **sklearn-model** | 8001 | ML Titanic (RandomForest) |
| **cnn-image** | 8002 | CNN CIFAR-10 (TensorFlow) |
| **mlflow-server** | 5000 | MLflow Tracking UI |
| **minio** | 9000/9001 | S3 Storage + Console |

## 🚀 Quick Start

### **Opción 1: Docker Compose (Recomendado para desarrollo)**

1. Configurar variables de entorno
cp .env.example .env

Editar .env con tu GOOGLE_API_KEY
2. Build de imágenes
.\build-local.ps1

3. Levantar servicios
docker-compose up -d

4. Verificar
docker-compose ps

5. Acceder
Gradio: http://localhost:7860
MLflow: http://localhost:5000
MinIO: http://localhost:9001 (minioadmin/minioadmin)


### **Opción 2: Docker Swarm (Producción/Demo)**


1. Inicializar Swarm
docker swarm init

2. Configurar variables
cp .env.example .env

Editar .env
3. Build de imágenes
.\build-local.ps1

4. Deploy del stack
.\deploy-stack.ps1

5. Verificar
docker stack services mlops

6. Ver logs
docker service logs mlops_gradio-frontend -f


## 📁 Estructura


infra/
├── docker-compose.yml # Compose para desarrollo
├── swarm-stack.yml # Stack para Swarm
├── .env.example # Template de variables
├── .env # Variables (no commitear)
├── build-local.ps1 # Script de build
├── deploy-stack.ps1 # Script de deploy Swarm
├── deploy.ps1 # Script de opciones de deploy
├── build-and-push.ps1 # Script de build + deploy
├── remove-stack.ps1 # Script de remove
└── README.md # Este archivo


## 🔧 Scripts

### **build-local.ps1**
Build de todas las imágenes localmente.


.\build-local.ps1

Build selectivo:
.\build-local.ps1 -Service "llm-connector"


### **deploy-stack.ps1**
Deploy del stack en Docker Swarm.

.\deploy-stack.ps1

Con stack name personalizado:
.\deploy-stack.ps1 -StackName "mi-mlops"

### **deploy.ps1**
Opciones de scrips

.\deployps1 build
.\deploy.ps1 deploy
.\deploy.ps1 rebuild
.\deploy.ps1 remove

### **remove.ps1**
Remove del stack

.\remove-stack.ps1


## 🌐 Variables de Entorno

**Archivo `.env`:**


Google Gemini
GOOGLE_API_KEY=tu_api_key_aqui
GOOGLE_MODEL=gemini-2.5-flash

MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

MinIO (S3)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_DEFAULT_REGION=us-east-1


## 📊 Volúmenes

### Docker Compose:

volumes:
minio-data: # Artifacts de MLflow
sklearn-models: # Modelos Titanic
cnn-models: # Modelos CNN


### Docker Swarm:
Los volúmenes son compartidos entre replicas del servicio.

## 🔒 Networking

### Compose:
- **Red bridge:** Comunicación interna entre servicios
- **Puertos publicados:** Acceso desde host

### Swarm:
- **Red overlay:** `mlops-network` (attachable)
- **Routing mesh:** Load balancing automático
- **Service discovery:** Resolución DNS por nombre de servicio

## 🐛 Troubleshooting

### Servicios no inician

Ver logs
docker-compose logs -f

o
docker service logs mlops_SERVICIO -f

Verificar recursos
docker system df
docker system prune # Limpiar espacio si es necesario


### Puertos en uso

Verificar puertos
netstat -ano | findstr ":7860 :5000 :8000"

Cambiar puertos en docker-compose.yml:
ports:

"7861:7860" # Usar puerto diferente


### Error: GOOGLE_API_KEY no encontrada


Verificar .env
cat .env | grep GOOGLE_API_KEY

Recargar variables (Swarm)
.\deploy-stack.ps1


### Servicios se reinician constantemente


Ver por qué fallan
docker service ps mlops_SERVICIO --no-trunc

Ver exit code
docker inspect CONTAINER_ID


## 📊 Monitoring

### Docker Compose:


Estado de servicios
docker-compose ps

Logs en tiempo real
docker-compose logs -f

Stats de recursos
docker stats


### Docker Swarm:

Estado del stack
docker stack services mlops

Logs por servicio
docker service logs mlops_gradio-frontend -f

Ver replicas
docker service ps mlops_llm-connector

Escalar servicios
docker service scale mlops_llm-connector=3


## 🔄 Actualizar Servicios

### Compose:

Rebuild y restart
docker-compose up -d --build

Solo un servicio
docker-compose up -d --build gradio-frontend


### Swarm:

Update de imagen
docker service update --image gradio-frontend:latest mlops_gradio-frontend

Force update (sin cambio de imagen)
docker service update --force mlops_gradio-frontend


## 🛑 Detener Servicios

### Compose:

Detener
docker-compose stop

Detener y eliminar
docker-compose down

Eliminar volúmenes también
docker-compose down -v


### Swarm:

Eliminar stack
docker stack rm mlops

Salir de Swarm mode
docker swarm leave --force


## 📈 Performance Tips

1. **Recursos:**
   - Min 8GB RAM para todos los servicios
   - CNN requiere más memoria (6GB recomendado)

2. **Volúmenes:**
   - Usar volúmenes nombrados (más rápido que bind mounts)
   - Limpiar volúmenes no usados: `docker volume prune`

3. **Networking:**
   - Swarm: Usar `endpoint_mode: vip` para mejor performance
   - Compose: Usar red bridge personalizada

4. **Build:**
   - Usar BuildKit: `DOCKER_BUILDKIT=1 docker build`
   - Cache de layers: No usar `--no-cache` sin necesidad

## 🔐 Seguridad

1. **Secrets (Swarm):**
Crear secret
echo "mi-api-key" | docker secret create google_api_key -

Usar en stack:
secrets:
google_api_key:
external: true


2. **Credenciales MinIO:**
   - Cambiar `minioadmin/minioadmin` en producción
   - Usar variables de entorno

3. **Network isolation:**
   - No exponer puertos innecesarios
   - Usar redes internas para comunicación entre servicios

## 📝 Autor

Christian Gomez - Proyecto Final MLOps

