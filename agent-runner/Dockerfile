FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUNNER_WORKSPACE=/workspace \
    RUNNER_ACCESS_REQUIRED=true \
    RUNNER_ALLOWED_COMMANDS=python,python3,pytest,git,rg \
    RUNNER_COMMAND_PATH=/usr/local/bin:/usr/bin:/bin \
    RUNNER_TMPDIR=/tmp

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git ripgrep \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent_runner ./agent_runner

RUN useradd --create-home --home-dir /home/runner --shell /usr/sbin/nologin --uid 10001 runner \
    && mkdir -p /workspace /tmp \
    && chown -R runner:runner /workspace /home/runner

USER 10001:10001
EXPOSE 8080

CMD ["uvicorn", "agent_runner.main:app", "--host", "0.0.0.0", "--port", "8080"]
