# Dockerfile for Hugging Face Spaces Deployment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /app

# Install system dependencies required for PyTorch & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and data files
COPY src/ ./src
COPY data/ ./data
COPY scripts/ ./scripts

# Set Python path to ensure module imports resolve cleanly
ENV PYTHONPATH=/app

# Expose Port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run FastAPI server
CMD ["python", "-m", "src.api_server"]
