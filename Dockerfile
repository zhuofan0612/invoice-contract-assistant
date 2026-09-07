# Two stages so the runtime image carries no build toolchain.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

# Optional extras. Build with --build-arg PROFILE=full to bake in the real
# embedding model and PDF/OCR support; the default image runs the fallbacks,
# which is enough to serve and evaluate the whole pipeline.
ARG PROFILE=slim
COPY requirements-llm.txt requirements-ml.txt requirements-pdf.txt requirements-pg.txt ./

# psycopg is installed in every profile, not as an extra: this image is what
# docker-compose runs, and compose always points it at Postgres. Leaving it to
# the "full" profile means the default build starts, connects to nothing, and
# fails at the first query.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -U pip \
 && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt -r requirements-pg.txt \
 && if [ "$PROFILE" = "full" ]; then \
      /opt/venv/bin/pip install --no-cache-dir \
        -r requirements-llm.txt -r requirements-ml.txt -r requirements-pdf.txt; \
    else \
      /opt/venv/bin/pip install --no-cache-dir -r requirements-llm.txt; \
    fi


FROM python:3.12-slim AS runtime

# Tesseract is only needed for the scanned-contract path; the language pack
# matters because these are Swedish documents and the English model misreads
# å/ä/ö, which in a contract can be the difference between two clauses.
ARG PROFILE=slim
RUN if [ "$PROFILE" = "full" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-swe \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Non-root: the service reads contracts and writes traces, and needs no more
# than that.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY app/ ./app/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY data/ ./data/

RUN mkdir -p /app/var && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
