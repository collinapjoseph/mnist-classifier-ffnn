# ── Stage 1: train ────────────────────────────────────────────────────────────
# Uses the full PyTorch image to train the model and export best_model.onnx.
# The resulting .onnx file is copied into the leaner runtime stage.
FROM python:3.11-slim AS trainer

WORKDIR /app

# Install training dependencies
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    onnx==1.16.0 \
    matplotlib==3.8.4 \
    numpy==1.26.4

COPY mnist_pipeline.py .

# Train and export — MNIST data is downloaded automatically into /app/data
RUN python mnist_pipeline.py


# ── Stage 2: serve ────────────────────────────────────────────────────────────
# Minimal image: only needs onnxruntime (not full PyTorch) to run inference.
FROM python:3.11-slim AS server

WORKDIR /app

RUN pip install --no-cache-dir \
    flask==3.0.3 \
    onnxruntime==1.17.3 \
    numpy==1.26.4 \
    pillow==10.3.0

# Copy only what the server needs from the trainer stage
COPY --from=trainer /app/best_model.onnx .

COPY server.py .
COPY ui.html .

EXPOSE 5000

ENV HOST=0.0.0.0
ENV PORT=5000

CMD ["python", "server.py"]
