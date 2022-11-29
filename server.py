#!/usr/bin/env python3

import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from pytorch_lightning import seed_everything
from torch import autocast
import random
import time

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--addr",
        type=str,
        nargs=1,
        default="localhost:9001",
        help="address:port to bound http server"
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs=1,
        default="configs/stable-diffusion/v2-inference-v.yaml"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="global seed, default is Unix epoch"
    )

    opts = parser.parse_args()
    seed_everything(opts.seed)

    config = OmegaConf.load(opts.config)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load model
    pl_sd = torch.load(config.checkpoint, map_location="cpu")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.cuda()
    model.eval()
    model = model.to(device)
    print(f"sds: Model loaded")
    if len(m) > 0 or len(u) > 0:
        print(f"sds: Model has missing keys = {m}, unexpected keys = {u}")
    
    # Instantiate sampler
    sampler = None
    if config.sampler == "plms":
        sampler = PLMSSampler(model)
    elif config.sampler == "dpm":
        sampler = DPMSolverSampler(model)
    elif config.sampler == "ddim":
        sampler = DDIMSampler(model)
    else:
        print(f"Invalid sampler: {config.sampler}")

    g_counter = 0

    print("sds: Ready to serve.")

    # Sample
    req = {
        "prompt": "a professional photograph of an astronaut riding a triceratops",
        "steps": int(50),
        "downsample": int(8), # 8 is reasonable trade-off between speed and resolution, <8 is very slow
        "scale": float(9.0), # What does this do?
        "eta": float(0.0),
        "seed": int(4), # Does nothing?
    }

    downsample_factor = req["downsample"]
    batch_size = 1
    shape = [config.channels, config.height // downsample_factor, config.width // downsample_factor]

    for _ in range(10):
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            prompts = batch_size * [req["prompt"]]
            c = model.get_learned_conditioning(prompts)
            scale = req["scale"]
            uc = model.get_learned_conditioning(batch_size * [""]) if scale != 1.0 else None

            samples, _ = sampler.sample(
                S = req["steps"],
                conditioning = c,
                batch_size = batch_size,
                shape = shape,
                verbose = False,
                unconditional_guidance_scale = scale,
                unconditional_conditioning = uc,
                eta = req["eta"],
                x_T = torch.randn([batch_size] + shape, device=device) if req["seed"] else None,
            )

            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            for x_sample in x_samples:
                img_buf = 255.0  * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(img_buf.astype(np.uint8))
                img.save(f"out_{g_counter}.png")
                g_counter += 1
        

