FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias de sistema
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código con nueva estructura
COPY app/ ./app/
COPY pipeline/ ./pipeline/

# Crear directorios (sin cambiar ownership aún)
RUN mkdir -p /app/models /app/data

# Usuario no-root
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# CORREGIDO: Usar uvicorn directamente
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
