# Hosted play: managym wheel + static frontend + experience server.
# Build context is the repository root:
#   flyctl deploy . --config deploy/play/fly.toml

# ---- managym wheel ----
FROM python:3.12-bookworm AS engine
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN pip install --no-cache-dir "maturin>=1.5,<2.0"
WORKDIR /build
COPY managym/ managym/
COPY content/ content/
RUN maturin build --release -i python3.12 -m managym/Cargo.toml -o /wheels

# ---- frontend SPA ----
FROM node:22-slim AS web
WORKDIR /build/academy/ui
COPY academy/ui/package.json academy/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
# Card art is versioned content, never a runtime fetch. This layer keys on
# the fetch script and the pack manifest alone — Scryfall is queried once
# per pack change (paced, identified User-Agent), not once per deploy.
COPY academy/ui/scripts/fetch-card-art.mjs scripts/fetch-card-art.mjs
COPY academy/ui/src/lib/packs/ src/lib/packs/
COPY academy/fixtures/*demo*.json /build/academy/fixtures/
RUN node scripts/fetch-card-art.mjs
COPY academy/ui/ ./
RUN ACADEMY_STATIC_BUILD=1 npm run build

# ---- runtime ----
FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/
COPY pyproject.toml uv.lock ./
COPY --from=engine /wheels /wheels
RUN uv sync --locked --only-group play-runtime --no-install-project \
    && uv pip install /wheels/*.whl
COPY academy/ academy/
COPY manabot/ manabot/
COPY academy/ui/src/lib/packs/ academy/ui/src/lib/packs/
COPY deploy/play/asgi.py ./
COPY --from=web /build/academy/ui/build/ academy/ui/build/

ENV PYTHONPATH=/app \
    ACADEMY_FRONTEND_BUILD=/app/academy/ui/build \
    ACADEMY_TRACES_DIR=/data/traces

# Import the actual entrypoint with only the packaged runtime dependencies.
# This catches missing modules before Fly replaces a serving machine.
RUN uv run --no-sync python -c "import asgi"

EXPOSE 8000

CMD [".venv/bin/uvicorn", "asgi:app", "--host", "0.0.0.0", "--port", "8000"]
