# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.13

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.local/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY . /app

RUN uv python install ${PYTHON_VERSION} && \
    uv sync --frozen --extra dev --extra ops --python ${PYTHON_VERSION} && \
    uv pip install --python .venv/bin/python --no-deps -e managym

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION=3.13

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.local/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HYDRA_FULL_ERROR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app
COPY --from=builder /root/.local /root/.local

RUN mkdir -p /root/manabot-runs

CMD ["python", "manabot/model/train.py"]
