# Start from the PyTorch image.
# This image *already has* Python 3, pip, and PyTorch 1.12.1 + CUDA 11.3
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

# Copy the entire latent-diffusion repo
COPY . .

# 1. Install system dependencies (for the script's download/unzip)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Accept UID/GID from build args with defaults
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create group and user matching host UID/GID
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

# Switch to that user
RUN chown -R appuser:appgroup /app
USER appuser

# 2. Install Python dependencies using pip
#    The base image already has torch, torchvision, and numpy.
#    We just need the missing pieces from the environment.yaml.
#    'taming-transformers' is installed this way per the original notebook.
RUN pip install \
    omegaconf \
    einops \
    tqdm \
    pillow \
    requests \
    taming-transformers

# 3. Install the ldm package itself
RUN pip install -e .