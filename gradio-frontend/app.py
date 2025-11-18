"""
Gradio Frontend Application
Purpose: Unified interface for LLM, ML (Titanic), and CNN (CIFAR-10) services
Author: Christian Gomez
Date: 2025-01-17
"""

import os
import requests
import gradio as gr
from PIL import Image
import numpy as np
import json
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === USAR NOMBRES DE SERVICIOS (funcionan en red overlay) ===
LLM_SERVICE_URL = os.getenv("LLM_CONNECTOR_URL", "http://llm-connector:8000")
ML_SERVICE_URL = os.getenv("SKLEARN_MODEL_URL", "http://sklearn-model:8000")
CNN_SERVICE_URL = os.getenv("CNN_IMAGE_URL", "http://cnn-image:8000")

# Para logs
print(f"🔗 LLM Service: {LLM_SERVICE_URL}")
print(f"🔗 ML Service: {ML_SERVICE_URL}")
print(f"🔗 CNN Service: {CNN_SERVICE_URL}")


def chat_with_llm(prompt, context, max_tokens, temperature):
    """Chat with LLM service"""
    try:
        response = requests.post(
            f"{LLM_SERVICE_URL}/chat",
            json={
                "prompt": prompt,
                "context": context,
                "max_tokens": int(max_tokens),
                "temperature": float(temperature)
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        return (
            result["response"],
            f"Model: {result.get('model', 'unknown')}\nTokens: {result.get('tokens_used', 0)}"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "Connection failed"


def train_ml():
    """Train ML model"""
    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/train",
            timeout=120  # Entrenamiento puede tardar
        )
        response.raise_for_status()
        result = response.json()
        
        return (
            f"✅ Entrenamiento exitoso!\n\n{result.get('message', '')}",
            json.dumps(result.get('metrics', {}), indent=2)
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "N/A"


def predict_ml(pclass, sex, age, sibsp, parch, fare, embarked):
    """Predict with Titanic ML service"""
    try:
        features = {
            "Pclass": float(pclass),
            "Sex": float(sex),
            "Age": float(age),
            "SibSp": float(sibsp),
            "Parch": float(parch),
            "Fare": float(fare),
            "Embarked": float(embarked)
        }
        
        response = requests.post(
            f"{ML_SERVICE_URL}/predict",
            json={"features": features},
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        prediction_text = "✅ Sobrevivió" if result["prediction"] == 1 else "❌ No sobrevivió"
        
        return (
            prediction_text,
            f"{result['confidence']:.2%}",
            json.dumps({f"Clase {i}": f"{prob:.2%}" for i, prob in enumerate(result["probabilities"])}, indent=2),
            result["model_version"]
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "N/A", "N/A", "N/A"


def train_cnn():
    """Train CNN model"""
    try:
        response = requests.post(
            f"{CNN_SERVICE_URL}/train",
            json={"epochs": 10},
            timeout=600  # CNN tarda más
        )
        response.raise_for_status()
        result = response.json()
        
        return (
            f"✅ Entrenamiento CNN exitoso!\n\n{result.get('message', '')}",
            json.dumps(result.get('metrics', {}), indent=2)
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "N/A"


def predict_cnn(image):
    """Predict with CNN service"""
    try:
        if image is None:
            return "⚠️ No se subió imagen", "N/A", "N/A", "N/A"
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
        else:
            pil_image = image
        
        # Resize to 32x32 (CIFAR-10 size)
        pil_image = pil_image.resize((32, 32))
        
        # Save to buffer
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Send to service
        files = {"file": ("image.png", buffer, "image/png")}
        response = requests.post(
            f"{CNN_SERVICE_URL}/predict",
            files=files,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        return (
            f"🏷️ {result.get('class_name', 'Unknown')}",
            f"{result.get('confidence', 0):.2%}",
            json.dumps({f"Clase {i}": f"{prob:.2%}" for i, prob in enumerate(result["probabilities"])}, indent=2),
            json.dumps(result.get("limitations", {}), indent=2)
        )
    except Exception as e:
        return f"❌ Error: {str(e)}", "N/A", "N/A", "N/A"


# Create Gradio interface
with gr.Blocks(title="MLOps Pipeline Inteligente", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Pipeline MLOps Inteligente")
    gr.Markdown("Sistema integrado con LLM (Gemini 2.5), ML (Titanic), y CNN (CIFAR-10)")
    
    with gr.Tabs():
        # === LLM Tab ===
        with gr.TabItem("💬 LLM Chat"):
            gr.Markdown("### Conversación con Gemini 2.5 Flash")
            
            with gr.Row():
                with gr.Column(scale=2):
                    prompt = gr.Textbox(
                        label="Pregunta",
                        placeholder="¿Qué es MLOps?",
                        lines=3
                    )
                    context = gr.Textbox(
                        label="Contexto (opcional)",
                        placeholder="Proporciona contexto adicional...",
                        lines=2
                    )
            
            with gr.Row():
                max_tokens = gr.Slider(50, 500, 150, step=50, label="Máximos tokens")
                temperature = gr.Slider(0.1, 1.0, 0.7, step=0.1, label="Temperatura")
            
            chat_btn = gr.Button("🚀 Enviar", variant="primary")
            
            with gr.Row():
                llm_output = gr.Textbox(label="Respuesta", lines=8, interactive=False)
                llm_info = gr.Textbox(label="Información", lines=3, interactive=False)
            
            chat_btn.click(
                fn=chat_with_llm,
                inputs=[prompt, context, max_tokens, temperature],
                outputs=[llm_output, llm_info]
            )
        
        # === ML Titanic Tab ===
        with gr.TabItem("📊 ML Titanic"):
            gr.Markdown("### Predicción de Supervivencia - Titanic Dataset")
            
            with gr.Tabs():
                with gr.TabItem("🔮 Predicción"):
                    with gr.Row():
                        pclass = gr.Slider(1, 3, 1, step=1, label="Clase (1=Primera, 3=Tercera)")
                        sex = gr.Radio([0, 1], label="Sexo (0=Masculino, 1=Femenino)", value=1)
                    
                    with gr.Row():
                        age = gr.Slider(0.0, 80.0, 30.0, step=1, label="Edad")
                        sibsp = gr.Slider(0, 8, 0, step=1, label="Hermanos/Cónyuge a bordo")
                    
                    with gr.Row():
                        parch = gr.Slider(0, 6, 0, step=1, label="Padres/Hijos a bordo")
                        fare = gr.Slider(0.0, 500.0, 50.0, step=1, label="Precio del ticket")
                    
                    embarked = gr.Radio([0, 1, 2], label="Puerto (0=Southampton, 1=Cherbourg, 2=Queenstown)", value=0)
                    
                    predict_ml_btn = gr.Button("🔮 Predecir", variant="primary")
                    
                    with gr.Row():
                        ml_prediction = gr.Textbox(label="Resultado", interactive=False)
                        ml_confidence = gr.Textbox(label="Confianza", interactive=False)
                    
                    ml_probabilities = gr.Textbox(label="Probabilidades", lines=3, interactive=False)
                    ml_model_version = gr.Textbox(label="Versión del Modelo", interactive=False)
                    
                    predict_ml_btn.click(
                        fn=predict_ml,
                        inputs=[pclass, sex, age, sibsp, parch, fare, embarked],
                        outputs=[ml_prediction, ml_confidence, ml_probabilities, ml_model_version]
                    )
                
                with gr.TabItem("🎓 Entrenamiento"):
                    gr.Markdown("### Entrenar modelo RandomForest con dataset Titanic")
                    gr.Markdown("⚠️ Esto tomará aproximadamente 30-60 segundos")
                    
                    train_ml_btn = gr.Button("🎓 Entrenar Modelo", variant="primary")
                    
                    ml_train_output = gr.Textbox(label="Estado del Entrenamiento", lines=5, interactive=False)
                    ml_train_metrics = gr.Textbox(label="Métricas", lines=5, interactive=False)
                    
                    train_ml_btn.click(
                        fn=train_ml,
                        inputs=[],
                        outputs=[ml_train_output, ml_train_metrics]
                    )
        
        # === CNN Tab ===
        with gr.TabItem("🖼️ CNN Visión"):
            gr.Markdown("### Clasificación de Imágenes - CIFAR-10 (5 clases)")
            gr.Markdown("**Clases disponibles:** airplane, automobile, bird, cat, dog")
            
            with gr.Tabs():
                with gr.TabItem("🔮 Predicción"):
                    image_input = gr.Image(label="Subir Imagen (se redimensionará a 32x32)", type="pil")
                    predict_cnn_btn = gr.Button("🔮 Clasificar", variant="primary")
                    
                    with gr.Row():
                        cnn_prediction = gr.Textbox(label="Clase Predicha", interactive=False)
                        cnn_confidence = gr.Textbox(label="Confianza", interactive=False)
                    
                    cnn_probabilities = gr.Textbox(label="Distribución de Probabilidades", lines=5, interactive=False)
                    cnn_limitations = gr.Textbox(label="Limitaciones del Modelo", lines=4, interactive=False)
                    
                    predict_cnn_btn.click(
                        fn=predict_cnn,
                        inputs=[image_input],
                        outputs=[cnn_prediction, cnn_confidence, cnn_probabilities, cnn_limitations]
                    )
                
                with gr.TabItem("🎓 Entrenamiento"):
                    gr.Markdown("### Entrenar modelo CNN con CIFAR-10")
                    gr.Markdown("⚠️ **ADVERTENCIA:** Esto tomará 5-10 minutos")
                    
                    train_cnn_btn = gr.Button("🎓 Entrenar CNN", variant="primary")
                    
                    cnn_train_output = gr.Textbox(label="Estado del Entrenamiento", lines=5, interactive=False)
                    cnn_train_metrics = gr.Textbox(label="Métricas", lines=5, interactive=False)
                    
                    train_cnn_btn.click(
                        fn=train_cnn,
                        inputs=[],
                        outputs=[cnn_train_output, cnn_train_metrics]
                    )
    
    gr.Markdown("---")
    gr.Markdown("**Pipeline MLOps Inteligente** | LLM + ML + CNN | MLflow Tracking Enabled")


if __name__ == "__main__":
    print("🚀 Iniciando Gradio Frontend...")
    print(f"🔗 LLM Service: {LLM_SERVICE_URL}")
    print(f"🔗 ML Service: {ML_SERVICE_URL}")
    print(f"🔗 CNN Service: {CNN_SERVICE_URL}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", 7860)),
        share=False
    )
