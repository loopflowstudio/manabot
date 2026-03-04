# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    libssl-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
RUN python3 -m pip install --upgrade pip uv

COPY . /app

RUN uv venv .venv --python python3 && \
    . .venv/bin/activate && \
    uv pip install -e managym && \
    uv pip install -e .

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV HYDRA_FULL_ERROR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app /app

RUN mkdir -p /root/manabot-runs

CMD ["python3", "manabot/model/train.py"]
