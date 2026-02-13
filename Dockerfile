FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt

# Exclude editable install entry to reduce build overhead in containers.
RUN pip install --upgrade pip && \
    grep -v "^-e \\\.$" requirements.txt > /tmp/requirements-docker.txt && \
    pip install --no-cache-dir -r /tmp/requirements-docker.txt

COPY . /app

CMD ["python3", "app.py"]