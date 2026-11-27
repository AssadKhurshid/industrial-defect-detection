FROM python:3.13-slim

WORKDIR /app

# Pinned dependencies first, so this layer stays cached when only our code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# The model is not in the image: mount the folder with model.onnx + model.onnx.data at /models.
ENV IDD_MODEL=/models/model.onnx
EXPOSE 8000
CMD ["uvicorn", "idd.api:app", "--host", "0.0.0.0", "--port", "8000"]
