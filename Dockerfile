# --- THIS IS THE KEY FIX ---
# We use a base image that matches the environment.yaml (PyTorch 1.7.1, CUDA 11.0, Python 3.8)
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /app

# 1. Copy all your local files
COPY . .

# 2. Install system dependencies (as root)
#    'git' is required for the taming-transformers/CLIP install
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Accept UID/GID from build args with defaults
ARG USER_ID=1000
ARG GROUP_ID=1000

# 4. Create group and user matching host UID/GID
RUN groupadd -g ${GROUP_ID} appgroup && \
    useradd -m -u ${USER_ID} -g appgroup appuser

# 5. Change ownership of all files in /app to the new user
RUN chown -R appuser:appgroup /app

# 6. Switch to the new user. 
USER appuser

# 7. Install Python dependencies in logical groups for better caching and debugging

# --- Group 1: Fixes and Core LDM Dependencies ---
# We TRUST the base image's PyTorch, TorchVision, and NumPy.
# We ONLY install the packaging fix and the core LDM/Lightning dependencies.
RUN pip install \
    packaging==21.3 \
    pytorch-lightning==1.4.2 \
    omegaconf==2.1.1 \
    einops==0.3.0 \
    torchmetrics==0.5.1

# --- Group 3: Simple Utilities ---
RUN pip install \
    tqdm \
    pillow \
    requests

# --- Group 4: The rest of the (more complex) dependencies ---
RUN pip install \
    #albumentations==0.4.3 \
    #opencv-python==4.1.2.30 \
    pandas==1.4.4 \
    pudb==2019.2 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.2 \
    #test-tube>=0.7.5 \
    #streamlit>=0.73.1 \
    #torch-fidelity==0.3.0 \
    transformers==4.3.1 \
    kornia

# 8. Install the editable git-based dependencies
RUN pip install \
    -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers \
    -e git+https://github.com/openai/CLIP.git@main#egg=clip

# 9. Install the ldm package itself (as appuser)
RUN pip install -e .

# 10. Set the default command to keep the container alive (for VSC)
CMD ["sleep", "infinity"]