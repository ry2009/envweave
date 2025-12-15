FROM python:3.11-slim

WORKDIR /app

# Build envweave from source (build context must be repo root).
COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY src /app/src

RUN pip install --no-cache-dir -e .

ENV ENVWEAVE_ENV_ID="pkg://envweave.examples.gsm8k_env:GSM8KEnv"

EXPOSE 8000
CMD ["python", "-m", "envweave.server"]

