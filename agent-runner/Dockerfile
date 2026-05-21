FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      nodejs \
      npm \
      ripgrep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY runner.py /app/runner.py

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/runner.py"]
