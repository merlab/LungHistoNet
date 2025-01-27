FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-tk \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/processed /app/final

CMD ["python", "Application.py"]