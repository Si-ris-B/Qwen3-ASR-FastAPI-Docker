# BUILDER STAGE (Only for gathering wheels)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3-pip python3-dev git ffmpeg build-essential
WORKDIR /build
COPY requirements.txt .
# We just build the standard wheels
RUN pip3 wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# FINAL STAGE
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
WORKDIR /app
# Set PYTHONPATH so the app can find the src and qwen_asr folders
ENV PYTHONPATH=/app:/app/src
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

# Install standard wheels
COPY --from=builder /build/wheels /wheels
RUN pip3 install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Copy source code
COPY qwen_asr /app/qwen_asr
COPY src /app/src

EXPOSE 8000
CMD ["uvicorn", "stt_service.main:app", "--host", "0.0.0.0", "--port", "8000"]