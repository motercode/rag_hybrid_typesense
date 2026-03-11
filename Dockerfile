# Usamos una imagen oficial de Python ligera
FROM python:3.10-slim
# Evita que Python genere archivos .pyc y fuerza la salida por consola
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ARG DEBIAN_FRONTEND=noninteractive

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos las herramientas de compilación y los motores OCR (Tesseract + Poppler)
RUN apt-get update && apt-get install -y \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-spa \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*
# Copiamos los requerimientos primero para aprovechar el caché de Docker
COPY requirements.txt .

# Instalamos las librerías de Python
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el script de ingesta a la imagen
COPY ingesta_actas.py .

# Comando por defecto al iniciar el contenedor
CMD ["python", "ingesta_actas.py"]