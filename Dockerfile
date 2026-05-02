FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements_runtime.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements_runtime.txt

# Copy full project
COPY . .

EXPOSE 8000