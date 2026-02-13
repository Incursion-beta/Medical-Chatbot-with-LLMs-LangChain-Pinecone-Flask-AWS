FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements-docker.txt

COPY . /app

CMD ["python3", "app.py"]