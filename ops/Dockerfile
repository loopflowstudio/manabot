FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

COPY . .

RUN pip install --upgrade pip uv

RUN uv venv .venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install -e managym && \
    uv pip install -e .

RUN mkdir -p ~/manabot-runs

ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HYDRA_FULL_ERROR=1

CMD ["python", "manabot/model/train.py", "--config-name", "simple"]
