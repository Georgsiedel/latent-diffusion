import sys
import os
import argparse
import json
import zipfile
import requests
from tqdm import tqdm

sys.path.append(".")
sys.path.append('./taming-transformers')

from taming.models import vqgan 
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import numpy as np 
from PIL import Image

def setup_model_checkpoint(ckpt_path):
    """
    Checks if the checkpoint exists. If not, downloads and unzips it.
    """
    if os.path.exists(ckpt_path):
        print(f"Checkpoint found at {ckpt_path}. Skipping download.")
        return
    
    print(f"Checkpoint not found at {ckpt_path}. Starting download...")
    
    # Get the directory to download and extract to
    model_dir = os.path.dirname(ckpt_path)
    os.makedirs(model_dir, exist_ok=True)
    
    zip_url = "https://ommer-lab.com/files/latent-diffusion/cin.zip"
    zip_path = os.path.join(model_dir, "cin.zip")
    
    try:
        # Download the file
        with requests.get(zip_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading cin.zip",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

        print("Download complete. Extracting zip file...")
        
        # Unzip the file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        print(f"Extraction complete.")
        
        # Clean up the zip file
        os.remove(zip_path)
        
        # Final check
        if not os.path.exists(ckpt_path):
            print(f"Error: Zip file was extracted, but the expected checkpoint")
            print(f"{ckpt_path} was not found. Please check the zip file contents.")
            sys.exit(1)
            
    except Exception as e:
        print(f"An error occurred during download/extraction: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path) # Clean up partial download
        sys.exit(1)

def load_model_from_config(config, ckpt, device):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def load_class_maps(map_file_path):
    """Loads the class map and creates synset -> index and index -> synset mappings."""
    with open(map_file_path, 'r') as f:
        class_map = json.load(f)
    
    synset_to_index = {}
    index_to_synset = {}
    for index_str, info in class_map.items():
        index = int(index_str)
        synset = info[0] # "n01440764"
        synset_to_index[synset] = index
        index_to_synset[index] = synset
    return synset_to_index, index_to_synset

def load_target_synsets(list_file_path):
    """Loads the target synsets from a text file (e.g., IN100.txt)."""
    with open(list_file_path, 'r') as f:
        synsets = [line.strip() for line in f if line.strip()]
    return synsets

def main(args):
    # --- 1. Set up Model Checkpoint ---
    setup_model_checkpoint(args.ckpt)

    # --- 2. Set up GPU Device ---
    if not torch.cuda.is_available():
        print(f"CUDA not available. Exiting.")
        sys.exit(1)
    
    # This will use the single GPU assigned by docker-compose
    device = torch.device("cuda")
    print(f"Running on {device} (PyTorch sees: {torch.cuda.get_device_name(0)})")

    # --- 3. Load Model and Sampler ---
    config = OmegaConf.load(args.config) 
    model = load_model_from_config(config, args.ckpt, device)
    sampler = DDIMSampler(model)

    # --- 4. Load Class Mappings and Target List ---
    synset_to_index, index_to_synset = load_class_maps(args.class_map_file)
    target_synsets = load_target_synsets(args.class_list_file)
    
    target_indices = [synset_to_index[syn] for syn in target_synsets if syn in synset_to_index]
    print(f"Found {len(target_indices)} valid target classes in {args.class_list_file}.")

    # --- 5. Define the Work ---
    # Use the provided start/end indices to slice the list
    if args.end_index > len(target_indices):
        print(f"Warning: end_index {args.end_index} is out of bounds for {len(target_indices)} classes. Clamping.")
        args.end_index = len(target_indices)
        
    indices_to_generate = target_indices[args.start_index:args.end_index]
    
    print(f"Generating {len(indices_to_generate)} classes (Indices {args.start_index} to {args.end_index-1} from your list)")

    # --- 6. Sampling Loop ---
    batch_size = args.batch_size
    if batch_size > args.n_samples_per_class:
        batch_size = args.n_samples_per_class
        print(f"Batch size > samples. Setting batch size to {args.n_samples_per_class}")

    with torch.no_grad(), model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(batch_size * [1000]).to(device)}
        )
        
        for class_index in tqdm(indices_to_generate, desc=f"Generating Classes {args.start_index}-{args.end_index}"):
            synset_name = index_to_synset[class_index]
            save_path_class = os.path.join(args.outdir, synset_name)
            os.makedirs(save_path_class, exist_ok=True)
            
            xc = torch.tensor(batch_size * [class_index])
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(device)})
            
            for i in range(0, args.n_samples_per_class, batch_size):
                samples_ddim, _ = sampler.sample(S=args.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=batch_size,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=args.scale,
                                                 unconditional_conditioning=uc, 
                                                 eta=args.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                
                for j, x_sample in enumerate(x_samples_ddim):
                    img_index = i + j
                    if img_index >= args.n_samples_per_class:
                        break
                        
                    img = 255. * x_sample.cpu().numpy()
                    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                    img = Image.fromarray(img)
                    
                    img_filename = f"{synset_name}_{img_index:04d}.jpg"
                    img.save(os.path.join(save_path_class, img_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- New Required Args ---
    parser.add_argument("--start_index", type=int, default=0, help="Start index from the IN100.txt list to process")
    parser.add_argument("--end_index", type=int, default=100, help="End index (exclusive) from the IN100.txt list to process")
    
    # --- Path args ---
    parser.add_argument("--outdir", type=str, default="/app/outputs", help="Directory to save images")
    parser.add_argument("--ckpt", type=str, default="/app/models/ldm/cin256/model.ckpt", help="Path to the model checkpoint")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/cin256-v2.yaml", help="Path to the model config")
    parser.add_argument("--class_list_file", type=str, default="data/IN100.txt", help="Path to IN100.txt")
    parser.add_argument("--class_map_file", type=str, default="data/imagenet_class_index.json", help="Path to imagenet_class_index.json")

    # --- Sampling args ---
    parser.add_argument("--n_samples_per_class", type=int, default=10, help="Number of samples to generate per class")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for sampling")
    parser.add_argument("--ddim_steps", type=int, default=250, help="Number of DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale for unconditional guidance")

    args = parser.parse_args()
        
    main(args)