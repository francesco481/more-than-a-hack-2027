FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies first for layer caching
COPY requirements.txt requirements-starter.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -r requirements-starter.txt

# Copy repository code
COPY . .

# Make scripts executable
RUN chmod +x reproduce.sh

# Default entrypoint runs the reproducible pipeline
ENTRYPOINT ["./reproduce.sh", "--local"]
