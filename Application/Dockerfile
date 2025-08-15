# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system packages required by tkinter, OpenCV, and OpenSSL
RUN apt-get update && apt-get install -y \
    python3-tk \
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0 libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

COPY lunginsightcloud-fa31002e7988.json /app/lunginsightcloud-fa31002e7988.json

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/visualizations && chmod -R 777 /app/visualizations

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Run your program when the container launches
CMD ["python", "Application.py"]