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
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and requirements.txt (if exists)
COPY pyproject.toml requirements.txt* ./

# Install Python dependencies
# We use requirements.txt if generated, otherwise fall back to pip install .
RUN if [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir .; \
    fi

# Copy the rest of the application
COPY . .

# Expose the port Gradio runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "src/agentic_rag/app.py"]
