FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        make git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

COPY src ./src
COPY scripts ./scripts
COPY Makefile ./

RUN uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1

CMD ["tail", "-f", "/dev/null"]
