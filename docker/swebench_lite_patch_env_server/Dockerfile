FROM python:3.11-slim

WORKDIR /app

# Docker client for running SWE-bench instance images (requires mounting docker.sock at runtime).
RUN apt-get update \
  && apt-get install -y --no-install-recommends docker.io git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Build envweave from source (build context must be repo root).
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src

RUN pip install --no-cache-dir -e '.[swebench]'

ENV ENVWEAVE_ENV_ID="pkg://envweave.examples.swebench_lite_patch_env:SWEbenchLitePatchEnv"

EXPOSE 8000
CMD ["python", "-m", "envweave.server"]

