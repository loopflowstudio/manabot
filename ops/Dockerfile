FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including ninja and Python development headers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    python3-dev \
    libspdlog-dev \
    libfmt-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Update git submodules (if any)
RUN git submodule update --init --recursive

# Upgrade pip
RUN pip install --upgrade pip

# Install the local submodule (managym) first
RUN pip install -e managym

# Install the main package; pip will automatically install dependencies from pyproject.toml
RUN pip install -e .

# Create required directories
RUN mkdir -p ~/manabot-runs

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HYDRA_FULL_ERROR=1

# Default command: run the training script
CMD ["python", "manabot/scripts/train.py"]
