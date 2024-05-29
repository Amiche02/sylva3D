from datetime import datetime
import os
from einops import rearrange
import torch
import time
import numpy as np
from PIL import Image
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor
from functools import partial
from dataclasses import dataclass
from typing import Dict, Optional, List
from omegaconf import OmegaConf
import yaml
import argparse
import cv2

from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from mvdiffusion.data.single_image_dataset import SingleImageDataset

weight_dtype = torch.float16

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

def save_image(tensor, fp):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

def sam_init(sam_checkpoint):
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:0")
    predictor = SamPredictor(sam)
    return predictor

def sam_segment(predictor, input_image, *bbox_coords):
    bbox = np.array(bbox_coords)
    image = np.asarray(input_image)
    start_time = time.time()
    predictor.set_image(image)
    masks_bbox, scores_bbox, logits_bbox = predictor.predict(box=bbox, multimask_output=True)
    print(f"SAM Time: {time.time() - start_time:.3f}s")
    out_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    out_image[:, :, :3] = image
    out_image_bbox = out_image.copy()
    out_image_bbox[:, :, 3] = masks_bbox[-1].astype(np.uint8) * 255
    torch.cuda.empty_cache()
    return Image.fromarray(out_image_bbox, mode='RGBA')

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def preprocess(predictor, input_image, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if segment:
        image_rem = input_image.convert('RGBA')
        image_nobg = remove(image_rem, alpha_matting=True)
        arr = np.asarray(image_nobg)[:, :, -1]
        x_nonzero = np.nonzero(arr.sum(axis=0))
        y_nonzero = np.nonzero(arr.sum(axis=1))
        x_min = int(x_nonzero[0].min())
        y_min = int(y_nonzero[0].min())
        x_max = int(x_nonzero[0].max())
        y_max = int(y_nonzero[0].max())
        input_image = sam_segment(predictor, input_image.convert('RGB'), x_min, y_min, x_max, y_max)
    if rescale:
        image_arr = np.array(input_image)
        in_w, in_h = image_arr.shape[:2]
        out_res = min(RES, max(in_w, in_h))
        ret, mask = cv2.threshold(np.array(input_image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(mask)
        max_size = max(w, h)
        ratio = 0.75
        side_len = int(max_size / ratio)
        padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
        center = side_len // 2
        padded_image[center - h // 2 : center - h // 2 + h, center - w // 2 : center - w // 2 + w] = image_arr[y : y + h, x : x + w]
        rgba = Image.fromarray(padded_image).resize((out_res, out_res), Image.LANCZOS)
        rgba_arr = np.array(rgba) / 255.0
        rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
        input_image = Image.fromarray((rgb * 255).astype(np.uint8))
    else:
        input_image = expand2square(input_image, (127, 127, 127, 0))
    return input_image, input_image.resize((320, 320), Image.Resampling.LANCZOS)

def load_wonder3d_pipeline(cfg):
    pipeline = MVDiffusionImagePipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    )
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    pipeline.unet.enable_xformers_memory_efficient_attention()
    return pipeline

def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(
        root_dir='', 
        num_views=6, 
        img_wh=[256, 256], 
        bg_color='white', 
        crop_size=crop_size, 
        single_image=single_image
    )
    return dataset[0]

def run_pipeline(pipeline, cfg, single_image, guidance_scale, steps, seed, crop_size, write_image=False):
    batch = prepare_data(single_image, crop_size)
    pipeline.set_progress_bar_config(disable=True)
    seed = int(seed)
    generator = torch.Generator(device=pipeline.unet.device).manual_seed(seed)
    imgs_in = torch.cat([batch['imgs_in']] * 2, dim=0).to(torch.float16)
    camera_embeddings = torch.cat([batch['camera_embeddings']] * 2, dim=0).to(torch.float16)
    task_embeddings = torch.cat([batch['normal_task_embeddings'], batch['color_task_embeddings']], dim=0).to(torch.float16)
    camera_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1).to(torch.float16)
    imgs_in = rearrange(imgs_in, "Nv C H W -> (Nv) C H W")

    out = pipeline(
        imgs_in,
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

    if write_image:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
        num_views = len(VIEWS)

        cur_dir = os.path.join("./outputs", f"cropsize-{int(crop_size)}-cfg{guidance_scale:.1f}")
        scene = 'scene' + datetime.now().strftime('@%Y%m%d-%H%M%S')
        scene_dir = os.path.join(cur_dir, scene)
        normal_dir = os.path.join(scene_dir, "normals")
        colors_dir = os.path.join(scene_dir, "rgb")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(colors_dir, exist_ok=True)

        for j in range(num_views):
            view = VIEWS[j]
            normal = normals_pred[j]
            color = images_pred[j]
            normal_filename = f"normals_000_{view}.png"
            rgb_filename = f"rgb_000_{view}.png"
            save_image(normal, os.path.join(normal_dir, normal_filename))
            save_image(color, os.path.join(colors_dir, rgb_filename))

    return normals_pred, images_pred

def process_images(input_path, output_dir, cfg, guidance_scale=1.0, steps=50, seed=42, crop_size=192, segment=True, rescale=False, write_image=False):
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    predictor = sam_init(sam_checkpoint)
    pipeline = load_wonder3d_pipeline(cfg)
    
    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    else:
        input_files = [input_path]

    for file_path in input_files:
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            input_image = Image.open(file_path)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            main_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(main_output_dir, exist_ok=True)
            normals_dir = os.path.join(main_output_dir, "normals")
            colors_dir = os.path.join(main_output_dir, "rgb")
            os.makedirs(normals_dir, exist_ok=True)
            os.makedirs(colors_dir, exist_ok=True)
            processed_image_highres, processed_image = preprocess(predictor, input_image, segment, rescale)
            normals_pred, images_pred = run_pipeline(pipeline, cfg, processed_image_highres, guidance_scale, steps, seed, crop_size, write_image)
            for i, (normal, image) in enumerate(zip(normals_pred, images_pred)):
                save_image(normal, os.path.join(normals_dir, f'normal_{i}.png'))
                save_image(image, os.path.join(colors_dir, f'rgb_{i}.png'))
            print(f"Processed and saved results for {file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images into 3D representations.")
    parser.add_argument("--config", type=str, default="configs/mvdiffusion-joint-ortho-6views.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--input_dir", type=str, help="Directory containing input images.")
    parser.add_argument("--input_image", type=str, help="Path to a single input image file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output images.")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for the diffusion model.")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion inference steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--crop_size", type=int, default=192, help="Crop size for the input images.")
    parser.add_argument("--segment", action="store_true", help="Enable background removal.")
    parser.add_argument("--rescale", action="store_true", help="Enable rescaling and recentering of the image.")
    parser.add_argument("--write_image", action="store_true", help="Write results to the output directory.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = OmegaConf.merge(OmegaConf.structured(TestConfig), OmegaConf.load(f))

    input_path = args.input_image if args.input_image else args.input_dir
    process_images(
        input_path,
        args.output_dir,
        cfg,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        seed=args.seed,
        crop_size=args.crop_size,
        segment=args.segment,
        rescale=args.rescale,
        write_image=args.write_image
    )
