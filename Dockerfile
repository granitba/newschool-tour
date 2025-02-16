# Use an official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for venv and netcat
RUN apt update && apt install -y python3-venv netcat-traditional

# Create venvs
RUN python3 -m venv /app/main/venv && \
    python3 -m venv /app/captcha/venv

# Copy all necessary files to the container (excluding unwanted ones using .dockerignore)
COPY .. .

# Install dependencies in each venv
RUN /app/main/venv/bin/python -m pip install --no-cache-dir -r /app/main/req.txt && \
    /app/captcha/venv/bin/python -m pip install --no-cache-dir -r /app/captcha/req.txt

# Ensure the captcha service starts first
CMD ["/bin/bash", "-c", "echo 'Starting Captcha Service...'; /app/captcha/venv/bin/python /app/captcha/main.py & echo 'Waiting for Captcha Service to be Ready...'; while ! nc -z localhost 8080; do sleep 1; done; echo 'Starting Main Script...'; /app/main/venv/bin/python /app/main/main_rand.py"]