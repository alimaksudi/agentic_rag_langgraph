# Use an official Python slim image as the base
FROM python:3.11-slim

# Set environmental variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH /app/src

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (including pyproject.toml and src/)
COPY . .

# Install the package itself in editable mode or just its entry points
RUN pip install --no-cache-dir -e .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "src/agentic_rag/app.py"]
