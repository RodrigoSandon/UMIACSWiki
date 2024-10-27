# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RAG pipeline script into the container
COPY rag-qdrant.py /app/

# Copy the env directory into the container
COPY env /app/env

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "rag-qdrant.py"]
