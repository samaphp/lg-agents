FROM python:3.12.3-slim

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT="/usr/local/"
ENV UV_COMPILE_BYTECODE=1
ENV MODE=prod

COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-install-project --no-dev
RUN pip install playwright && playwright install --with-deps


COPY .env .
COPY src/agents/ ./agents/
COPY src/core/ ./core/
COPY src/api_schema/ ./api_schema/
COPY src/service/ ./service/
COPY src/run_service.py .
RUN mkdir logs


CMD ["python", "run_service.py"]
