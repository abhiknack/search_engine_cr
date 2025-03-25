#!/bin/bash
# startup.sh - Simple startup script for the application

# Print debug info
echo "Starting application on port $PORT"

# Start the application
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app 