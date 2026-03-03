# ──────────────────────────────────────────────────────────────────────────────
# Base image: PyTorch + CUDA 12.1 (matches bitsandbytes requirements)
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ──────────────────────────────────────────────────────────────────────────────
# System dependencies
# ──────────────────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ──────────────────────────────────────────────────────────────────────────────
# Working directory
# ──────────────────────────────────────────────────────────────────────────────
WORKDIR /app

# ──────────────────────────────────────────────────────────────────────────────
# Python dependencies
# Install before copying source code so this layer is cached unless
# requirements.txt changes.
# ──────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Copy application source
# ──────────────────────────────────────────────────────────────────────────────
COPY llm.py          .
COPY pdf_loader.py   .
COPY vector_store.py .
COPY app.py          .

# ──────────────────────────────────────────────────────────────────────────────
# Hugging Face token – pass at runtime via:
#   docker run -e HUGGINGFACE_TOKEN=hf_xxx ...
# Never bake the token into the image.
# ──────────────────────────────────────────────────────────────────────────────
ENV HUGGINGFACE_TOKEN=""

# HF cache inside the container (mount a volume here to persist downloads)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Gradio settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# ──────────────────────────────────────────────────────────────────────────────
# Expose Gradio port
# ──────────────────────────────────────────────────────────────────────────────
EXPOSE 7860

# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
