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
import falcon
import io
from urllib.parse import urlparse
from wsgiref.simple_server import make_server
import socket

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)

"""
Returns a PIL image generated from `model` instantiated by `config` using `req`.

`req` is a dictionary with the following attributes (all are optional, except `prompt`)
  - prompt: <required>, text prompt
  - steps: number of iterations
  - downsample: downsample factor, 8 gives the default size output
  - scale: don't know what that do
  - eta: don't know what that do
  - seed: seems to do nothing

Defaults are applied to missing attributes in `req`.
"""
def generate_one(model, config, req):
    default_req = {
        "steps": int(50),
        "downsample": int(8),
        "scale": float(9.0),
        "eta": float(0.0),
        "seed": int(0xae),
    }
    req = { **default_req, **req }

    assert 'prompt' in req

    downsample_factor = req["downsample"]
    batch_size = 1
    shape = [config.channels, config.height // downsample_factor, config.width // downsample_factor]

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

        assert len(x_samples) == 1
        img_buf = 255.0  * rearrange(x_samples[0].cpu().numpy(), 'c h w -> h w c')
        img = Image.fromarray(img_buf.astype(np.uint8))
        return img

if __name__ == "__main__":
    shard_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    server_id = f"{socket.getfqdn()}-{shard_id}"

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
    parser.add_argument(
        "--serverid",
        type=str,
        default=server_id,
        help="server identifier sent to clients, default to <fqdn>-<cuda_device_id>"
    )

    opts = parser.parse_args()

    # Parse host:port
    # Achtung! urlparse is a hack to handle IPv4 and IPv6 addresses!
    r = urlparse(f"http://{opts.addr}")
    hostname = r.hostname
    port = r.port

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

    print("sds: Ready to serve.")

    class StableDiffusionResource:
        def on_post(self, req, resp):
            start_time = time.time()
            # Validate req
            req = {
                "prompt": "test engineer"
            }

            # A string that identifies random generator seeds for repro.
            js_epoch = int(time.time() * 1000)
            ident=f"t_{js_epoch}__gs_{opts.seed}__s{0}_{server_id}"

            img = generate_one(model, config, req)
            
            # Generate image buffer
            # JPEG q=100 is close to lossless.
            # TODO: consider webp / png
            with io.BytesIO() as buf:
                img.save(buf, format="jpeg", quality=100)
                resp.status = falcon.HTTP_200
                resp.content_type = "image/jpeg"
                resp.set_header("server", "wacky6/sds")
                resp.set_header("content-disposition", f"attachment; filename={ident}.jpg")
                resp.set_header("x-served-by", server_id)
                resp.set_header("x-elapsed-time-ms", int((time.time() - start_time) * 1000))
                resp.data = buf.getvalue()


    app = falcon.App()
    app.add_route("/sd", StableDiffusionResource())

    with make_server(hostname, port, app) as server:
        print(f"sds: listening on {opts.addr}")
        server.serve_forever()


        

