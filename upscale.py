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
from glob import glob
from os import path
import sys
from tile_utils import clever_merge, clever_tiles

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm.util import exists, instantiate_from_config

torch.set_grad_enabled(False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/x4-upscaling.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(time.time()),
        help="seed for sampling, default is Unix epoch",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=75,
        help="number of steps to sample"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=8,
        help="stable diffusion scale factor"
    )
    parser.add_argument(
        "--noise",
        type=int,
        default=20,
        help="stable diffusion noise augmentation"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta for DIMM sampler"
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=512,
        help="tile size, should be multiples of 32, default to 0 (don't tile)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=32,
        help="tile padding, default to 32, must be greater than tile",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="input file glob",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="output directory, default to cwd (current directory)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_sdx4",
        help="output file suffix",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="text prompt for stable diffusion, optional"
    )

    opts = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load model
    config = OmegaConf.load(opts.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(config.checkpoint)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Initialize sampler
    sampler = None
    if config.sampler == "plms":
        sampler = PLMSSampler(model)
    elif config.sampler == "dpm":
        sampler = DPMSolverSampler(model)
    elif config.sampler == "ddim":
        sampler = DDIMSampler(model)
    else:
        print(f"Invalid sampler: {config.sampler}")

    sampler.make_schedule(opts.steps, ddim_eta=opts.eta, verbose=False)

    """
    Returns upscaled image.

    - in_image: A np.ndarray whose shape is (h,w,c), uint8 type
    - prompt: text prompt, optional
    """
    def process_tile(in_image, prompt=""):
        model = sampler.model
        h,w,c = in_image.shape

        seed_everything(opts.seed)
        prng = np.random.RandomState(opts.seed)

        start_code = prng.randn(1, model.channels, h, w)
        start_code = torch.from_numpy(start_code).to(device, dtype=torch.float32)

        def make_noise_augmentation(model, batch, noise_level=None):
            x_low = batch[model.low_scale_key]
            x_low = x_low.to(memory_format=torch.contiguous_format).float()
            x_aug, noise_level = model.low_scale_model(x_low, noise_level)
            return x_aug, noise_level

        with torch.no_grad(), torch.autocast("cuda"):
            image = torch.tensor(in_image).to(device, dtype=torch.float32) / 127.5 - 1.0
            batch = {
                "lr": rearrange(image, 'h w c -> 1 c h w'),
                "txt": 1 * [prompt],
            }
            
            c = model.cond_stage_model.encode(batch["txt"])
            c_cat = list()

            if isinstance(model, LatentUpscaleFinetuneDiffusion):
                for ck in model.concat_keys:
                    cc = batch[ck]
                    if exists(model.reshuffle_patch_size):
                        assert isinstance(model.reshuffle_patch_size, int)
                        cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                    p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)
                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}
                # uncond cond
                uc_cross = model.get_unconditional_conditioning(1, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
            elif isinstance(model, LatentUpscaleDiffusion):
                noise_level = torch.Tensor(1 * [opts.noise]).to(device, dtype=torch.long)
                x_augment, noise_level = make_noise_augmentation(
                    model, batch, noise_level)
                cond = {"c_concat": [x_augment],
                        "c_crossattn": [c], "c_adm": noise_level}
                # uncond cond
                uc_cross = model.get_unconditional_conditioning(1, "")
                uc_full = {"c_concat": [x_augment], "c_crossattn": [
                    uc_cross], "c_adm": noise_level}
            else:
                raise NotImplementedError()

            shape = [model.channels, h, w]
            samples, _ = sampler.sample(
                opts.steps,
                1,
                shape,
                cond,
                verbose=False,
                eta = opts.eta,
                unconditional_guidance_scale=opts.scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
                callback=None
            )

            samples = model.decode_first_stage(samples)
            out_image = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
            out_image = rearrange(out_image, '1 c h w -> h w c')
            out_image = (out_image.cpu().numpy() * 255.0).astype(np.uint8)

            return out_image

    # Process files
    files = glob(opts.input)
    for p in files:
        _, file_name = path.split(p)
        basename, extname = path.splitext(file_name)
        out_file_name = f"{basename}{opts.suffix}.png"
        out_path = os.path.join(opts.output_dir, out_file_name)

        print(f"{file_name} -> {out_path}")

        image = None
        with Image.open(p) as im:
            image = np.asarray(im, dtype=np.uint8)

        tiles, out_image = None, None
        if opts.tile > 0:
            tiles = clever_tiles(image, opts.tile, opts.padding)
        else:
            tiles = [image]

        # Process tiles
        # TODO: Maybe run parallel on multi GPUs, or don't bother with torch pipeline
        out_tiles = list(map(lambda tile: process_tile(tile, prompt=opts.prompt), tiles))

        if opts.tile > 0:
            out_image = clever_merge(out_tiles, image, opts.tile, opts.padding, config.upscale_factor)
        else:
            out_image = out_tiles[0]

        print(out_image.shape)
        Image.fromarray(out_image).save(out_path, optimize=True)

