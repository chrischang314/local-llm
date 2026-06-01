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
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY runner.py /app/runner.py
COPY agent_runner /app/agent_runner

ENV PYTHONDONTWRITEBYTECODE=1
CMD ["python", "/app/runner.py"]
