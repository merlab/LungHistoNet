# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system packages required by tkinter and OpenCV
RUN apt-get update && apt-get install -y \
    python3-tk \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for state, processed, and final files
RUN mkdir -p /app/processed /app/final

# Run your program when the container launches
CMD ["python", "Application.py"]