# Hosted play uses the same packaged Academy server and SPA as local installs.
# Build context: repository root; flyctl deploy . --config deploy/play/fly.toml
FROM node:22-slim AS web
WORKDIR /build/academy/ui
COPY academy/ui/package.json academy/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY academy/ui/scripts/fetch-card-art.mjs scripts/fetch-card-art.mjs
COPY managym/academy/packs/ /build/managym/academy/packs/
COPY managym/academy/fixtures/ /build/managym/academy/fixtures/
RUN node scripts/fetch-card-art.mjs
COPY academy/ui/ ./
RUN ACADEMY_STATIC_BUILD=1 npm run build

FROM python:3.12-bookworm AS engine
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/
WORKDIR /build
COPY managym/ managym/
COPY academy/__init__.py academy/__init__.py
COPY content/ content/
COPY --from=web /build/academy/ui/build/ managym/academy/site/
# Match the release staging project root so explicit site includes are honored.
RUN cp managym/pyproject.toml pyproject.toml
RUN sed -i 's|manifest-path = "Cargo.toml"|manifest-path = "managym/Cargo.toml"|; s|python-source = ".."|python-source = "."|' pyproject.toml
RUN rm managym/pyproject.toml
RUN cp managym/README.md README.md
RUN uv run --no-project --with maturin maturin build --release -i python3.12 -m managym/Cargo.toml -o /wheels
COPY pyproject.toml README.md ./
COPY manabot/ manabot/
COPY managym/pyproject.toml managym/pyproject.toml
COPY docs/experiment-metrics.md docs/experiment-metrics.md
COPY protocol/experience-v1.schema.json protocol/experience-v1.schema.json
RUN uv build --no-sources --wheel --out-dir /wheels

FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/
COPY pyproject.toml uv.lock ./
COPY managym/pyproject.toml managym/pyproject.toml
COPY --from=engine /wheels /wheels
RUN uv sync --locked --only-group play-runtime --no-install-project \
    && uv pip install --no-deps /wheels/*.whl
COPY deploy/play/asgi.py ./
ENV ACADEMY_TRACES_DIR=/data/traces
RUN uv run --no-sync python -c "import asgi; from managym.academy.extensions import bot_extension; assert bot_extension() is not None"
EXPOSE 8000
CMD [".venv/bin/uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
