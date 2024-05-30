import os
import torch
import fire
import gradio as gr
from PIL import Image
from functools import partial

import cv2
import time
import numpy as np
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor

import os
import sys
import numpy
import torch
import rembg
import threading
import urllib.request
from PIL import Image
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import streamlit as st
import huggingface_hub
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from mvdiffusion.models.unet_mv2d_condition import UNetMV2DConditionModel
from mvdiffusion.data.single_image_dataset import SingleImageDataset as MVDiffusionDataset
from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from einops import rearrange
import numpy as np
import subprocess
from datetime import datetime

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return ndarr

def save_image_to_disk(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    im.save(fp)

weight_dtype = torch.float16 

_TITLE = '''SYLVA3D: Single Image to 3D using Cross-Domain Diffusion'''
_DESCRIPTION = '''

'''
_NUM_GPUS = 2

if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed():
    torch.distributed.destroy_process_group()

def sam_init(rank):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{rank}")
    predictor = SamPredictor(sam)
    return predictor

def load_wonder3d_pipeline(cfg, rank):
    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    )
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to(f'cuda:{rank}')
    return pipeline

def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[256, 256], bg_color='white', crop_size=crop_size, single_image=single_image)
    return dataset[0]

def run_pipeline(rank, world_size, cfg, single_image, guidance_scale, steps, seed, crop_size, chk_group=None):
    setup_distributed(rank, world_size)
    
    pipeline = load_wonder3d_pipeline(cfg, rank)
    torch.set_grad_enabled(False)

    predictor = sam_init(rank)
    
    if chk_group is not None:
        write_image = "Write Results" in chk_group

    batch = prepare_data(single_image, crop_size)

    pipeline.set_progress_bar_config(disable=True)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)

    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(weight_dtype)

    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(weight_dtype)

    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(weight_dtype)

    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(weight_dtype)

    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")

    out = pipeline(
        imgs_in,
        camera_embeddings,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        output_type='pt',
        num_images_per_prompt=1,
        **cfg.pipe_validation_kwargs,
    ).images

    bsz = out.shape[0] // 2
    normals_pred = out[:bsz]
    images_pred = out[bsz:]
    num_views = 6
    if write_image:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        cur_dir = os.path.join("./outputs", f"cropsize-{int(crop_size)}-cfg{guidance_scale:.1f}")

        scene = 'scene' + datetime.now().strftime('@%Y%m%d-%H%M%S')
        scene_dir = os.path.join(cur_dir, scene)
        normal_dir = os.path.join(scene_dir, "normals")
        masked_colors_dir = os.path.join(scene_dir, "masked_colors")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(masked_colors_dir, exist_ok=True)
        for j in range(num_views):
            view = VIEWS[j]
            normal = normals_pred[j]
            color = images_pred[j]

            normal_filename = f"normals_000_{view}.png"
            rgb_filename = f"rgb_000_{view}.png"
            normal = save_image_to_disk(normal, os.path.join(normal_dir, normal_filename))
            color = save_image_to_disk(color, os.path.join(scene_dir, rgb_filename))

            rm_normal = remove(normal)
            rm_color = remove(color)

            save_image_numpy(rm_normal, os.path.join(scene_dir, normal_filename))
            save_image_numpy(rm_color, os.path.join(masked_colors_dir, rgb_filename))

    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]

    out = images_pred + normals_pred

    cleanup_distributed()
    return out

@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path: str
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    local_rank: int
    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_guidance_scales: List[float]
    validation_grid_nrow: int
    camera_embedding_lr_mult: float
    num_views: int
    camera_embedding_type: str
    pred_type: str  # joint, or ablation
    enable_xformers_memory_efficient_attention: bool
    cond_on_normals: bool
    cond_on_colors: bool

def run_demo():
    from utils.misc import load_config
    from omegaconf import OmegaConf

    cfg = load_config("./configs/mvdiffusion-joint-ortho-6views.yaml")
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    torch.multiprocessing.spawn(run_pipeline, args=(cfg,), nprocs=_NUM_GPUS, join=True)

if __name__ == '__main__':
    fire.Fire(run_demo)
