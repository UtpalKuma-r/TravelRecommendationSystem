FROM python:3.11-slim AS base
WORKDIR /app

# System deps (add if your libs need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app & artifacts (or download in entrypoint)
COPY app ./app
# COPY model.pkl ./  # OR download in code via hf_hub_download / S3

ENV PORT=7860
EXPOSE 7860
# Hugging Face expects the service on 7860 or 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
