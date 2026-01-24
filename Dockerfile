# syntax=docker/dockerfile:1
FROM python:3.13-slim

# Set workdir
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .


# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir "pip>=25.3" \
    && pip install --no-cache-dir $(python -c 'import tomllib; print(" ".join([dep for dep in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]]))')

# Expose Flask port
EXPOSE 8000

# Run the Flask app with gunicorn + gevent-websocket (websockets support)
CMD ["gunicorn", "-k", "geventwebsocket.gunicorn.workers.GeventWebSocketWorker", "-b", "0.0.0.0:8000", "app.init:app"]
