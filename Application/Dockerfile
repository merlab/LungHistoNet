FROM python:3.11-slim

LABEL org.opencontainers.image.title="LungInsight Annotator" \
      org.opencontainers.image.description="Lung histology tile annotator with Google Drive sync" \
      org.opencontainers.image.version="1.7"

WORKDIR /app

# tkinter, OpenCV, Plotly/Kaleido, OpenSSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libsm6 libxext6 libxrender-dev libgl1 libglib2.0-0 libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN test -f /app/lunginsightcloud-fa31002e7988.json

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV DISPLAY=:0

CMD ["python", "Application.py"]
