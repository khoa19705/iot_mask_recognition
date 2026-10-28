FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-cloud.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-cloud.txt
COPY . .
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser:appuser /app
USER appuser
EXPOSE 10000
CMD ["python", "start_cloud.py"]
