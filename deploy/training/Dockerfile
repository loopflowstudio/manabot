# All three inputs identify Linux amd64 manifests, never mutable tags.
FROM docker.io/library/python@sha256:2ed6491b93cd49272ee6de2b5a38440c3448360322c089fc23e370722d74179d AS python
FROM docker.io/astral/uv@sha256:ae1ba4db410fbe1d9eaf16c51add8e7608b271ebbb69bd58a56e80384fd190aa AS uv
FROM docker.io/library/rust@sha256:dacc9e51f252243eb59d2fb4cb4ad8b0d3f607b6a82c398cf8a321e59ff778a7
COPY --from=python /usr/local/ /usr/local/
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/manabot/runtime \
    UV_CACHE_DIR=/opt/manabot/uv-cache \
    UV_PYTHON_DOWNLOADS=never \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/manabot/build
COPY pyproject.toml uv.lock ./
COPY runtime.json prebuilt_gpu_setup.py /opt/manabot/
# Mount wheel bytes only during installation; they never enter an image layer.
RUN --mount=type=bind,source=verified,target=/wheels \
    uv venv --python /usr/local/bin/python3.12 /opt/manabot/runtime && \
    uv pip install --no-cache --python /opt/manabot/runtime/bin/python --offline --no-index \
      --find-links /wheels/wheels --no-deps --require-hashes --only-binary :all: \
      -r /wheels/requirements.txt && \
    mkdir manabot academy && touch manabot/__init__.py academy/__init__.py && \
    uv sync --locked --extra dev --extra play --python /usr/local/bin/python3.12 && \
    uv run --no-sync python /opt/manabot/prebuilt_gpu_setup.py --seal && \
    uv pip uninstall --python /opt/manabot/runtime/bin/python manabot && \
    rm -rf /opt/manabot/build
WORKDIR /workspace
