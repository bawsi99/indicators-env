FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy source code
COPY . .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Set Python path so env/ module imports work correctly
ENV PYTHONPATH=/app/env:/app

# Start the FastAPI server
CMD ["uvicorn", "env.indicators_env:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
