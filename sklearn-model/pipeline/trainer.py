"""
ML Pipeline Trainer
Purpose: Entrenamiento completo con preprocesamiento, evaluación y logging MLflow
Author: Christian
Date: 2025-01-17
"""

import os
import logging
import pickle
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
import json
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

# Configurar MLflow para usar MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))

logger.info(f"MLflow tracking: {mlflow.get_tracking_uri()}")
logger.info(f"MLflow S3 endpoint: {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")

class MLTrainer:
    def __init__(self, model_name: str = "titanic_survival_classifier"):
        self.model_name = model_name
        self.pipeline = None
        self.metrics = {}
        self.label_encoders = {}
        self.feature_columns = []

    def download_titanic_dataset(self):
        """
        Descarga dataset Titanic si no existe
        """
        data_path = "data/raw/titanic.csv"
        
        if os.path.exists(data_path):
            logger.info(f"✅ Dataset encontrado en {data_path}")
            return data_path
        
        logger.info("📥 Descargando dataset Titanic...")
        
        # Crear directorio si no existe
        os.makedirs("data/raw", exist_ok=True)
        
        try:
            # Descargar desde GitHub
            import requests
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Guardar archivo
            with open(data_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Dataset descargado exitosamente a {data_path}")
            return data_path
            
        except Exception as e:
            logger.error(f"❌ Error descargando dataset: {e}")
            raise
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Cargando dataset Titanic")
        data_path = self.download_titanic_dataset()
        
        logger.info(f"Cargando dataset desde {data_path}")
        df = pd.read_csv(data_path)
        
        df = pd.read_csv(data_path)
        df = self._preprocess_data(df)
        
        X = df.drop(columns=['Survived'])
        y = df['Survived']
        
        logger.info(f"Dataset cargado: {X.shape[0]} filas, {X.shape[1]} features")
        return X, y
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
        
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        
        categorical_cols = ['Sex', 'Embarked']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        df = df.astype(float)
        self.feature_columns = df.columns.tolist()
        return df
    
    def create_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
    
    def train(self, experiment_name: str = "sklearn_model_experiment") -> Dict[str, float]:
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.pipeline = self.create_pipeline()
        mlflow.set_experiment(experiment_name)
        
        # SOLO UN START_RUN (elimina el segundo)
        with mlflow.start_run(run_name=self.model_name) as run:
            # Log parámetros
            params = self.pipeline.get_params()
            mlflow.log_params({
                "model_type": "RandomForestClassifier",
                "n_estimators": params['classifier__n_estimators'],
                "max_depth": params['classifier__max_depth'],
                "test_size": 0.2,
                "dataset": "titanic",
                "scaler": "StandardScaler"
            })
            
            # Entrenar
            logger.info("Iniciando entrenamiento")
            self.pipeline.fit(X_train, y_train)
            
            # Evaluar
            train_pred = self.pipeline.predict(X_train)
            test_pred = self.pipeline.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            cv_scores = cross_val_score(self.pipeline, X, y, cv=5)
            
            # Log métricas
            self.metrics = {
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
            
            mlflow.log_metrics(self.metrics)
            
            # Log classification report
            report = classification_report(y_test, test_pred, output_dict=True)
            
            # Log modelo con registro
            mlflow.sklearn.log_model(
                self.pipeline,
                artifact_path="model",
                registered_model_name="titanic_survival_classifier"
            )
            # GUARDAR CLASSIFICATION REPORT
            report_path = "/app/models/classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(report_path)
            
            # Log dataset
            # GUARDAR DATASET COMO ARTIFACT
            train_data_path = "/app/models/training_data.csv"
            X_train.to_csv(train_data_path, index=False)
            mlflow.log_artifact(train_data_path)

            # GUARDAR FEATURE IMPORTANCE
            importances = self.pipeline.named_steps['classifier'].feature_importances_
            feature_names = X.columns.tolist()
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

            importance_path = "/app/models/feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)

            # Guardar localmente
            os.makedirs("/app/models", exist_ok=True)
            model_path = f"/app/models/{self.model_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.pipeline, f)
            
            encoder_path = f"/app/models/{self.model_name}_encoders.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoders, f)
            
            logger.info(f"✅ Modelo entrenado y registrado en MLflow: {run.info.run_id}")
            
            return self.metrics
