# Use an official NVIDIA CUDA + PyTorch base image
# This is required because bitsandbytes 4-bit quantization NEEDS an NVIDIA GPU and CUDA to run
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Ensure standard output is not buffered
ENV PYTHONUNBUFFERED=1

# Install system dependencies (git is required for some pip packages)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (5000 for Flask, change to 7860 if deploying on Hugging Face Spaces)
EXPOSE 5000

# Run the application using Gunicorn (production WSGI server)
# 1 worker with threads is better for memory-constrained GPU environments
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
