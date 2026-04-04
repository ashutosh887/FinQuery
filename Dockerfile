FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY README.md .

COPY finquery/ finquery/
COPY server/ server/
COPY scripts/ scripts/
COPY baseline.py .
COPY inference.py .
COPY openenv.yaml .

RUN uv pip install --system --no-cache -e .

ENV PYTHONPATH=/app
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
