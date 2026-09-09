FROM python:3.11-slim-bookworm

# ffmpeg does ingest normalisation; the GL libraries are what MediaPipe's
# native bindings link against even when running on CPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libegl1 \
        libgl1 \
        libgles2 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CLIMB_DATA_DIR=/data \
    CLIMB_MODEL_DIR=/models

WORKDIR /srv

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Opt-in test backend; ordinary images remain MediaPipe-only.
ARG INSTALL_VITPOSE=false
COPY requirements-vitpose.txt ./
RUN if [ "$INSTALL_VITPOSE" = "true" ]; then \
        pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
        && pip install --no-cache-dir -r requirements-vitpose.txt; \
    fi

COPY app ./app
COPY scripts ./scripts
COPY static ./static

# Bake the models into the image so a cold container serves the first request
# without reaching the network, and so CLIMB_POSE_MODEL can be changed without
# one. heavy is ~30MB and worth carrying: on small subjects it more than halves
# landmark jitter against lite.
RUN python scripts/download_models.py lite full heavy

VOLUME ["/data"]
EXPOSE 8000

# Shell form so $PORT is expanded at runtime: Fly, Render and Railway all
# inject the port to listen on. Falls back to 8000 for local runs.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
