# ---- Base ----
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Streamlit defaults for containers
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# System deps for pandas/numpy (and clean up afterwards)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
# If your file is named differently, adjust here (e.g., travel_recommendation_system.py)
COPY app.py /app/app.py

# Non-root user (safer)
RUN useradd -m appuser
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Run the app (you can override CMD at runtime if needed)
CMD ["streamlit", "run", "app.py"]
