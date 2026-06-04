# LungHistoNet

LungHistoNet is a deep learning project that utilizes Vision Transformers to analyze lung histology images. The goal is to enhance the detection and classification of pulmonary diseases through advanced AI techniques.

[Website](https://lunginsight.ai/)



## Lung Injury Annotation App (Docker)

Desktop annotator backed by Google Drive. The container needs **outbound HTTPS** to Google APIs and a **host X11 display** for the Tkinter UI.

**Image (canonical tag):** `amirhosseinebrahimi/lung-injury:latest_v1.7`

Keep the Docker Hub repository **private**. The image embeds a Google service account JSON; anyone with the image can extract it. Rotate credentials on Google Cloud if the image is leaked.

### Prerequisites

1. [Docker](https://www.docker.com/) installed (Apple Silicon or Intel build on Mac).
2. Docker Hub account: `docker login`
3. Pull the image:

   ```bash
   docker pull amirhosseinebrahimi/lung-injury:latest_v1.7
   ```

### macOS

1. Install Homebrew if needed:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install `socat`:

   ```bash
   brew install socat
   ```

3. **Terminal 1** — leave running:

   ```bash
   socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:"$DISPLAY"
   ```

4. **Terminal 2** — start the app:

   ```bash
   docker run -it --rm -e DISPLAY=host.docker.internal:0 \
     amirhosseinebrahimi/lung-injury:latest_v1.7
   ```

   Or with Compose:

   ```bash
   docker compose -f docker-compose.mac.yml run --rm app
   ```

### Linux (Debian/Ubuntu)

1. Allow Docker to use the host display:

   ```bash
   sudo xhost +local:docker
   ```

2. Run the app:

   ```bash
   docker run -it --rm \
     -e DISPLAY="$DISPLAY" \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     amirhosseinebrahimi/lung-injury:latest_v1.7
   ```

   Or with Compose:

   ```bash
   docker compose run --rm app
   ```

   Or use the helper script (Linux):

   ```bash
   chmod +x lunginjury
   ./lunginjury
   ```

### Verify image (no GUI)

Check that Python dependencies load inside the container:

```bash
docker run --rm amirhosseinebrahimi/lung-injury:latest_v1.7 \
  python -c "import cv2, plotly, PIL; print('ok')"
```

### Build from source

Place `lunginsightcloud-fa31002e7988.json` in `Application/` (file is gitignored). From the repo root:

```bash
docker build -t amirhosseinebrahimi/lung-injury:latest_v1.7 ./Application
```

Run with the same `docker run` / `docker compose` commands as above, using your local tag.

### Publish to Docker Hub (maintainers)

GitHub Actions workflow [`.github/workflows/docker.yml`](.github/workflows/docker.yml) builds and pushes on version tags (`v*`) or manual **workflow_dispatch**.

Repository secrets:

| Secret | Purpose |
|--------|---------|
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full JSON body for the service account file baked into the image |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

Manual push example:

```bash
docker login
docker build -t amirhosseinebrahimi/lung-injury:latest_v1.7 ./Application
docker push amirhosseinebrahimi/lung-injury:latest_v1.7
```

### Run without Docker

From `Application/` with the service account JSON present:

```bash
pip install -r requirements.txt
python Application.py
```
