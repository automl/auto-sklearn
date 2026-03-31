# Use the official Python image as the base
FROM mcr.microsoft.com/devcontainers/python:1-3.9-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.0 \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    curl \
    git \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy pyproject.toml and poetry.lock (if exists)
COPY pyproject.toml poetry.lock* /workspace/

# Install dependencies (without root package)
RUN poetry install --no-root --no-interaction

# Copy the rest of the application
COPY . /workspace/

# Install the package in development mode
RUN poetry install --no-interaction

# Create a non-root user
RUN useradd -m vscode && \
    chown -R vscode:vscode /workspace

# Switch to non-root user
USER vscode

# Set the default shell to bash
CMD ["/bin/bash"]
