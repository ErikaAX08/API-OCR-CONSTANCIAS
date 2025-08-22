# Dockerfile para API OCR Constancias Fiscales - Versión Ubuntu
# Compatible con arquitecturas x86_64 y ARM64
FROM ubuntu:22.04

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Variables para OpenCV sin GUI
    QT_X11_NO_MITSHM=1 \
    MPLBACKEND=Agg \
    # Python
    PYTHON_VERSION=3.11

# Instalar Python 3.11 y dependencias del sistema
RUN apt-get update && apt-get install -y \
    # Python 3.11
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    # Tesseract OCR y idiomas
    tesseract-ocr \
    tesseract-ocr-spa \
    tesseract-ocr-eng \
    # Librerías para OpenCV
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libfontconfig1 \
    # Librerías GL para Ubuntu 22.04
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    # Herramientas de compilación
    build-essential \
    pkg-config \
    # Herramientas adicionales
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Crear enlaces simbólicos para Python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Actualizar pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Crear usuario no-root
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Crear directorio para modelos EasyOCR y configurar permisos
RUN mkdir -p /home/appuser/.EasyOCR && \
    chown -R appuser:appuser /home/appuser

# Copiar código de la aplicación
COPY main.py .

# Crear directorio para logs
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Cambiar a usuario no-root
USER appuser

# Pre-descargar modelos de EasyOCR (opcional pero recomendado)
RUN python -c "import easyocr; easyocr.Reader(['es', 'en'], gpu=False)" || echo "EasyOCR models will be downloaded on first use"

# Puerto de la aplicación
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]