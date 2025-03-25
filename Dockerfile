FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Add missing gunicorn to requirements.txt
RUN pip install --no-cache-dir gunicorn

# Set PORT environment variable default
ENV PORT=8080

# Ensure the application runs as a non-root user
RUN useradd -m appuser
USER appuser

# Container starts with gunicorn server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app

