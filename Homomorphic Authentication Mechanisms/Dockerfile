# Dockerfile for Python 3.11 with Waters & Boneh-Boyen signatures
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libgmp-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Ensure results directory exists and is writable
RUN mkdir -p /app/results && chmod 777 /app/results

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (skip tenseal - not needed for Waters/LHS)
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    pandas>=2.0.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    pyyaml>=6.0 \
    cryptography>=41.0.0 \
    pycryptodome>=3.18.0 \
    blspy>=1.0.16 \
    tqdm>=4.65.0 \
    colorama>=0.4.6 \
    psutil>=5.9.0 \
    tabulate>=0.9.0 \
    openpyxl>=3.1.0 && \
    pip install --no-cache-dir petlib

# Copy project files
COPY . .

# Default command
CMD ["python", "run.py", "verify"]

