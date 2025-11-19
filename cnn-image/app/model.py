"""
CNN Model with CIFAR-10 Dataset
Purpose: Image classification with real CIFAR-10 data and custom filters
Author: Christian Gomez
Date: 2025-01-17
"""

import json
import os
import logging
import pickle
import numpy as np
from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow
from filters.custom_filters import ConvolutionFilters  # <-- AÑADIR ESTO

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

class CNNImageClassifier:
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3), num_classes: int = 5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ["airplane", "automobile", "bird", "cat", "dog"]
        self.label_encoder = None
        self.filters = ConvolutionFilters()  # <-- INICIALIZAR FILTROS
        
    def _unpickle_file(self, file_path: str) -> Dict:
        with open(file_path, 'rb') as f:
            return pickle.load(f, encoding='bytes')
    
    def download_cifar10(self):
        """
        Descarga CIFAR-10 usando Keras si no existe
        """
        data_dir = "data/raw/cifar-10-batches-py"
        
        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            logger.info(f"[SUCCESS] CIFAR-10 encontrado en {data_dir}")
            return True
        
        logger.info("[DOWNLOADING] Descargando CIFAR-10 (esto puede tardar 1-2 minutos)...")
        
        try:
            # Crear directorio
            os.makedirs(data_dir, exist_ok=True)
            
            # Descargar usando Keras
            from tensorflow.keras.datasets import cifar10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            
            # Guardar en formato pickle (compatible con código actual)
            # Batch 1-5: entrenamiento
            train_size = len(x_train) // 5
            for i in range(5):
                start = i * train_size
                end = (i + 1) * train_size if i < 4 else len(x_train)
                
                batch = {
                    b'data': x_train[start:end].reshape(-1, 3072),
                    b'labels': y_train[start:end].flatten().tolist()
                }
                
                with open(f"{data_dir}/data_batch_{i+1}", 'wb') as f:
                    pickle.dump(batch, f)
            
            # Test batch
            test_batch = {
                b'data': x_test.reshape(-1, 3072),
                b'labels': y_test.flatten().tolist()
            }
            with open(f"{data_dir}/test_batch", 'wb') as f:
                pickle.dump(test_batch, f)
            
            # Metadata
            meta = {
                b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer', 
                                b'dog', b'frog', b'horse', b'ship', b'truck']
            }
            with open(f"{data_dir}/batches.meta", 'wb') as f:
                pickle.dump(meta, f)
            
            logger.info(f"[SUCCESS] CIFAR-10 descargado y guardado en {data_dir}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Error descargando CIFAR-10: {e}")
            raise
        
    def load_cifar10_data(self, num_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        # Descargar si no existe
        self.download_cifar10()
        data_dir = "data/raw/cifar-10-batches-py"
        
        logger.info(f"Cargando CIFAR-10 desde {data_dir}")
        
        train_images = []
        train_labels = []
        
        for batch_num in range(1, 6):
            batch_path = os.path.join(data_dir, f"data_batch_{batch_num}")
            batch_data = self._unpickle_file(batch_path)
            
            images = batch_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            labels = batch_data[b'labels']
            
            train_images.append(images)
            train_labels.extend(labels)
        
        X = np.concatenate(train_images, axis=0)
        y = np.array(train_labels)
        
        mask = y < 5
        X = X[mask][:num_samples]
        y = y[mask][:num_samples]
        
        X = X.astype('float32') / 255.0
        
        logger.info(f"Dataset cargado: {X.shape[0]} imágenes, 5 clases")
        return X, y
    
    def create_model(self) -> models.Sequential:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, epochs: int = 10, experiment_name: str = "cnn_cifar10_experiment") -> Dict[str, float]:
        X, y = self.load_cifar10_data(num_samples=5000)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Aplicar filtros usando la clase externa
        logger.info("Aplicando filtros de convolución")
        X_train_filtered = self.filters.apply_filters(X_train)  # <-- USAR self.filters
        X_test_filtered = self.filters.apply_filters(X_test)    # <-- USAR self.filters
        
        self.model = self.create_model()
        mlflow.set_experiment(experiment_name)
        
        # SOLO UN START_RUN
        with mlflow.start_run(run_name="cnn_cifar10_classifier") as run:
            mlflow.log_params({
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "epochs": epochs,
                "batch_size": 32,
                "dataset": "CIFAR-10 (5 clases)"
            })
            
            # Entrenar
            logger.info("Entrenando CNN con CIFAR-10")
            history = self.model.fit(
                X_train_filtered, y_train,
                validation_data=(X_test_filtered, y_test),
                epochs=epochs,
                batch_size=32,
                verbose=1
            )
            
            # Evaluar
            test_loss, test_accuracy = self.model.evaluate(X_test_filtered, y_test)
            
            metrics = {
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "final_train_accuracy": history.history['accuracy'][-1],
                "final_val_accuracy": history.history['val_accuracy'][-1]
            }
            
            mlflow.log_metrics(metrics)
            
            # SALVAR MODELO LOCAL
            os.makedirs("models", exist_ok=True)
            model_path = "models/cnn_cifar10.h5"
            self.model.save(model_path)
                
            mlflow.tensorflow.log_model(
                self.model,
                artifact_path="model",
                registered_model_name="cnn_cifar10_classifier"
            )
            # GUARDAR IMAGEN DE EJEMPLO COMO ARTIFACT
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.plot(history.history['accuracy'], label='train_accuracy')
            plt.plot(history.history['val_accuracy'], label='val_accuracy')
            plt.title('CNN Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig("training_history.png")
            mlflow.log_artifact("training_history.png")
            
            # Guardar JSON con info del modelo
            model_info = {
                "model_name": "cnn_cifar10_classifier",
                "input_shape": self.input_shape,
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "final_accuracy": float(metrics["test_accuracy"]),
                "epochs_trained": epochs
            }
            with open("model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            mlflow.log_artifact("model_info.json")
            logger.info(f"[SUCCESS] Modelo CNN registrado en MLflow: {run.info.run_id}")
            
            return metrics
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if self.model is None:
            model_path = "models/cnn_cifar10.h5"
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
            else:
                raise FileNotFoundError("Modelo no encontrado. Ejecuta entrenamiento primero.")
        
        if image.shape != self.input_shape:
            image = tf.image.resize(image, self.input_shape[:2]).numpy()
        
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Usar self.filters
        filtered_image = self.filters.apply_filters(image)  # <-- USAR self.filters
        
        predictions = self.model.predict(filtered_image, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        
        return {
            "predicted_class": int(predicted_class),
            "class_name": self.class_names[predicted_class],
            "confidence": float(predictions[predicted_class]),
            "probabilities": predictions.tolist(),
            "limitations": {
                "note": "Modelo entrenado con CIFAR-10 (5 clases)",
                "classes": self.class_names,
                "accuracy": "Limitado a 5 clases de CIFAR-10",
                "robustness": "No apto para producción sin más datos"
            }
        }