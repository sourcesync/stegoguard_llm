FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY data/ data/

# Cloud Run sets PORT.
CMD exec gunicorn --bind :${PORT:-8080} --workers 1 --threads 4 --timeout 60 main:app
