# syntax=docker/dockerfile:1.7

FROM python:3.10-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install git early (build-time dependency for pip install -e . if needed)
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/satellitepy

COPY requirements.txt setup.py ./
COPY satellitepy ./satellitepy

# Use a pip cache mount to prevent re-downloading huge wheels (like cuDNN)
# when this layer is invalidated by changes earlier in the Dockerfile.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

# Install Codex CLI using the official script, and verify via full path.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl \
    && curl -fsSL https://chatgpt.com/codex/install.sh | sh \
    && /root/.local/bin/codex --version \
    && rm -rf /var/lib/apt/lists/*

COPY configs ./configs
COPY docs ./docs
COPY tools ./tools
COPY README.md LICENSE ./

VOLUME ["/data", "/output", "/patches", "/root/.codex"]

# Keep container alive for VS Code attach and interactive use

RUN mkdir -p /data/FineAir/FR24_dataset
CMD ["sleep", "infinity"]