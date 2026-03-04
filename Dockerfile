FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip uv maturin

RUN uv venv .venv --python 3.12 && \
    . .venv/bin/activate && \
    uv pip install -e "managym" && \
    uv pip install -e ".[dev]"

COPY entry.sh /app/entry.sh
RUN chmod +x /app/entry.sh

ENTRYPOINT ["/app/entry.sh"]
