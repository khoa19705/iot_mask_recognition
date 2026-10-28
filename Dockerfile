FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MASK_MODEL_PATH=best.onnx INFERENCE_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 APP_ENV=cloud
WORKDIR /app
COPY requirements-cloud.txt ./
RUN pip install --no-cache-dir -r requirements-cloud.txt
COPY . .
RUN useradd --create-home appuser && mkdir -p logs && chown -R appuser:appuser /app
USER appuser
EXPOSE 10000
CMD ["python", "start_cloud.py"]
