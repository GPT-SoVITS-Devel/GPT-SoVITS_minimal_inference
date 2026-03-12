FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*  && \
    python -m venv /venv && \
    /venv/bin/pip install --upgrade pip

COPY requirements.txt .
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

# # ============ runtime ============
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY . .

# 将 nltk_data, G2PWModel, pretrained_models 从 assets 目录链接到正确的位置
RUN ln -s /app/assets/nltk_data /venv/nltk_data && \
    ln -s /app/assets/G2PWModel /app/GPT_SoVITS/text/G2PWModel && \
    ln -s /app/assets/pretrained_models /app/pretrained_models

EXPOSE 8000

CMD ["python", "api_server.py"]