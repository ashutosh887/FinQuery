FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency file first for layer caching
COPY pyproject.toml .

# Copy source code
COPY finquerygym/ finquerygym/
COPY server/ server/
COPY baseline.py .
COPY openenv.yaml .
COPY README.md .

# Install dependencies
RUN uv pip install --system --no-cache -e .

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
