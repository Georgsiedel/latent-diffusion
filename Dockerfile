# Start from a PyTorch image with CUDA
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

# Copy the entire latent-diffusion repo (including your new script)
COPY . .

# Install dependencies from the environment.yaml
# First, install conda and system dependencies (unzip)
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

# Create the environment and install dependencies
RUN conda env create -f environment.yaml

# Activate the env for all subsequent RUN commands
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# Install the ldm package itself
RUN pip install -e .

# Install taming-transformers, Pillow, and requests
RUN pip install taming-transformers pillow requests

# This is the command that will be run by docker-compose
ENTRYPOINT ["conda", "run", "-n", "ldm", "python"]