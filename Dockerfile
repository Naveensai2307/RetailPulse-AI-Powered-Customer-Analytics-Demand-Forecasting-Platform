# Stage 1: Build stage
FROM python:3.11-slim AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy only the installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy the application code and necessary data
COPY dashboard/ ./dashboard/
COPY data/ ./data/
COPY dataset/ ./dataset/
COPY reports/ ./reports/
COPY scripts/ ./scripts/
COPY dags/ ./dags/

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to ensure the container is running correctly
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set entrypoint to run the Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
