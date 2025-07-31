# FROM python:3.10-slim
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# try with gcr.io/distroless/python3-debian11 (distroless)
#python:3.10-alpine
#ubuntu:24.04



WORKDIR /app

# Copy the project files
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Expose the port (if running an API)
# EXPOSE 8000

# Define the command to run the AI agent
CMD ["python", "Standalone-agent-qwen3.py"]