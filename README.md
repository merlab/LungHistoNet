# LungHistoNet
LungHistoNet is a deep learning project that utilizes Vision Transformers to analyze lung histology images. The goal is to enhance the detection and classification of pulmonary diseases through advanced AI techniques.


# [website](https://lunginsight.ai/)


# Application Setup Tutorial

## MAC

### Step 1: Create Docker Account
1. Create an account at [Docker Hub](https://hub.docker.com/) and set up a password.

### Step 2: Download and Install Docker
1. Download Docker from [Docker](https://www.docker.com/).
   - If using an M1/M2 Mac, select the Apple Core version.
   - If using an Intel Mac, select the Intel Core version.
2. Install the application.

### Step 3: Login and Pull the Application
1. Open Terminal.
2. Login to Docker using your Docker Hub credentials:
   ```bash
   docker login
   ```
   Follow the instructions to log in (ensure you verify your email to log in successfully).
3. Pull the application by running the following command:
   ```bash
   docker pull amirhosseinebrahimi/lung-injury:latset_v1.4
   ```

### Step 4: Install Brew
1. Check if Brew is installed. If not, open Terminal and run:
   ```bash
   /bin/bash -c "$(curl –fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Set up Brew by running the following commands (replace `XXX` with your laptop username):
   ```bash
   echo >> /Users/XXX/.zprofile
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/XXX/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

### Step 5: Install socat
1. Install `socat` using Brew:
   ```bash
   brew install socat
   ```


### Step 6: Set Up Display for the Application
1. Open Terminal and set up the display (leave the Terminal open after pressing Enter):
   
-  MacOS
   ```bash
   socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:"$DISPLAY"
   ```
-  Linux (debian)
   ```bash
   sudo xhost +local:docker
   ````
2. Open another Terminal and run the following command (replace `XXX` with your laptop username):
   
- MacOS

   ```bash
   docker run -it --rm -e DISPLAY=host.docker.internal:0 \
   amirhosseinebrahimi/lung-injury:latest_v1.4
   ```
-  Linux (debian)
   ```BASH
   sudo docker run -it --rm   -e DISPLAY=$DISPLAY  \
   -v /tmp/.X11-unix:/tmp/.X11-unix   \
   amirhosseinebrahimi/lung-injury:latest_v1.4

   ```
   
