FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATAGEN_OUTPUT_DIR=/app/output

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY datagen ./datagen

# Run as a non-root user
RUN useradd --create-home appuser && mkdir -p /app/output && chown appuser /app/output
USER appuser

EXPOSE 8000
CMD ["uvicorn", "datagen.api:app", "--host", "0.0.0.0", "--port", "8000"]
