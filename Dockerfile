# ===========================================================================
# Dockerfile for Hugging Face Spaces (CPU / GPU compatible)
# Exposes FastAPI server on port 7860
# ===========================================================================
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first for fast image building and reduced image footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and checkpoints
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY configs/ ./configs/
COPY main.py .

# Hugging Face Spaces mandates running under non-root UID 1000
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

# Start FastAPI uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

