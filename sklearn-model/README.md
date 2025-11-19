# Sklearn Model Service  Titanic Survival Prediction

Servicio de Machine Learning para predecir supervivencia en el Titanic usando RandomForest.

##  Features

 **RandomForest Classifier** optimizado
 **Descarga automática** del dataset Titanic
 **Preprocesamiento** automático de features
 **MLflow tracking** de experimentos
 **API REST** con FastAPI
 **Crossvalidation** para validación robusta

##  Dataset

 **Titanic:** Dataset de Kaggle (891 pasajeros)
 **Features:**
   `Pclass`: Clase (1=Primera, 2=Segunda, 3=Tercera)
   `Sex`: Sexo (0=Masculino, 1=Femenino)
   `Age`: Edad
   `SibSp`: Hermanos/Cónyuge a bordo
   `Parch`: Padres/Hijos a bordo
   `Fare`: Precio del ticket
   `Embarked`: Puerto de embarque (0=Southampton, 1=Cherbourg, 2=Queenstown)
 **Target:** `Survived` (0=No, 1=Sí)
 **Descarga automática** desde GitHub en primer entrenamiento

##  API Endpoints

### **POST /predict**
Predice supervivencia de un pasajero.

**Request:**
curl X POST http://localhost:8001/predict
H "ContentType: application/json"
d '{
"features": {
"Pclass": 1,
"Sex": 1,
"Age": 30.0,
"SibSp": 0,
"Parch": 0,
"Fare": 50.0,
"Embarked": 0
}
}'

**Response:**
{
"prediction": 1,
"prediction_label": "Survived",
"confidence": 0.87,
"probabilities": [0.13, 0.87],
"model_version": "v1.2.3",
"features_used": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
}

### **POST /train**
Entrena el modelo RandomForest.

**Request:**

curl X POST http://localhost:8001/train


**Response:**
{
"status": "success",
"message": "Modelo entrenado exitosamente",
"metrics": {
"accuracy": 0.82,
"precision": 0.79,
"recall": 0.76,
"f1_score": 0.77,
"cv_scores": [0.81, 0.83, 0.80, 0.84, 0.82]
},
"model_path": "/app/models/titanic_survival_classifier.pkl",
"training_samples": 714,
"test_samples": 179
}


### **GET /health**
Health check del servicio.

curl http://localhost:8001/health

{"status": "healthy", "service": "sklearnmodel"}
{"status": "unhealthy", "service": "sklearnmodel"} # Si no hay modelo


### **GET /info**
Información del servicio y modelo.
curl http://localhost:8001/info

##  Setup

### **Local**
Instalar dependencias
pip install r requirements.txt

Variables de entorno
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

Ejecutar
uvicorn app.main:app host 0.0.0.0 port 8000

### **Docker**

Build
docker build t sklearnmodel:latest .

Run
docker run p 8001:8000
e MLFLOW_TRACKING_URI=http://mlflowserver:5000
v $(pwd)/models:/app/models
sklearnmodel:latest

##  Testing

Instalar dependencias
pip install r requirements.txt

Ejecutar tests
pytest tests/ v

Tests rápidos (excluir descarga de datos)
pytest tests/ v m "not slow"

Con coverage
pytest tests/ v cov=app cov=report=html


##  Estructura

sklearnmodel/
 app/
  main.py # API FastAPI
  schemas.py # Pydantic models
 pipeline/
  trainer.py # Training pipeline
  predictor.py # Prediction pipeline
 tests/
  test_main.py # Tests unitarios
 data/
  raw/
  titanic.csv # Dataset (descargado automáticamente)
 models/
  *.pkl # Modelos entrenados
 Dockerfile
 requirements.txt
 README.md


##  Entrenamiento

### Primera vez (descarga Titanic.csv):

curl X POST http://localhost:8001/train

Descarga ~100KB + entrena 3060s


### Pipeline de entrenamiento:
1. **Descarga dataset** (si no existe)
2. **Preprocesamiento:**
    Elimina filas con valores faltantes
    Codifica variables categóricas (Sex, Embarked)
    Normaliza features numéricas
3. **Split:** 80% train, 20% test
4. **Entrenamiento:** RandomForest con 100 árboles
5. **Crossvalidation:** 5fold CV
6. **Evaluación:** Métricas en test set
7. **Guardado:**
    Modelo en `/app/models/titanic_survival_classifier.pkl`
    Encoders en `/app/models/encoders.pkl`
    Registro en MLflow

### Monitoreo con MLflow:

Abrir MLflow UI
open http://localhost:5000

Ver experimentos Titanic
Comparar métricas entre versiones
Descargar modelos registrados


##  Métricas Esperadas

| Métrica | Valor |
|||
| Accuracy | ~82% |
| Precision | ~79% |
| Recall | ~76% |
| F1Score | ~77% |
| CV Score (mean) | ~82% |

##  Feature Engineering

### Variables Categóricas:
 **Sex:** LabelEncoder (M=0, F=1)
 **Embarked:** LabelEncoder (S=0, C=1, Q=2)

### Features Numéricas:
 **Age, Fare:** Sin normalización (RandomForest es robusto)
 **Pclass, SibSp, Parch:** Valores discretos

### Feature Importance (típica):
1. **Sex:** ~30%
2. **Pclass:** ~20%
3. **Fare:** ~15%
4. **Age:** ~15%
5. **SibSp, Parch, Embarked:** ~20%

##  Troubleshooting

### Error: Modelo no encontrado

Entrenar el modelo primero
curl X POST http://localhost:8001/train


### Error: Dataset no encontrado

El entrenamiento descarga automáticamente
Si falla, descarga manual:
wget https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
O data/raw/titanic.csv


### Error: Missing features en predicción

Verificar que todos los features estén en el request
Ejemplo correcto:
{
"features": {
"Pclass": 1,
"Sex": 1,
"Age": 30.0,
"SibSp": 0,
"Parch": 0,
"Fare": 50.0,
"Embarked": 0
}
}


## Performance Tips

1. **Modelo preentrenado:** Carga en ~100ms
2. **Predicción:** ~5ms por request
3. **Entrenamiento:** ~3060 segundos
4. **Cache:** Modelo se mantiene en memoria

## Ejemplos de Predicción

### Pasajera de Primera Clase (Alta probabilidad de sobrevivir):

{
"Pclass": 1,
"Sex": 1,
"Age": 25,
"SibSp": 0,
"Parch": 0,
"Fare": 100,
"Embarked": 1
}
//  Prediction: Survived (85% confidence)



### Pasajero de Tercera Clase (Baja probabilidad):

{
"Pclass": 3,
"Sex": 0,
"Age": 40,
"SibSp": 1,
"Parch": 2,
"Fare": 15,
"Embarked": 0
}
//  Prediction: Not Survived (75% confidence)


##  Autor

Christian Gomez  Proyecto Final MLOps

