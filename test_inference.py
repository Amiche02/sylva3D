import os
import torch
import fire
from PIL import Image
import numpy as np
from rembg import remove
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
from functools import partial
import cv2
import time
from einops import rearrange

from mvdiffusion.pipelines.pipeline_mvdiffusion_image import MVDiffusionImagePipeline
from mvdiffusion.data.single_image_dataset import SingleImageDataset

def save_image(tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return ndarr

def save_image_to_disk(tensor, fp):
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255)
    tensor = tensor.to(torch.uint8)
    ndarr = tensor.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)
    return ndarr

weight_dtype = torch.float16
_GPU_ID = 0

def sam_init():
    sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_pt", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device=f"cuda:{_GPU_ID}")
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

def preprocess(predictor, input_image, chk_group=None, segment=True, rescale=False):
    RES = 1024
    input_image.thumbnail([RES, RES], Image.Resampling.LANCZOS)
    if chk_group is not None:
        segment = "Background Removal" in chk_group
        rescale = "Rescale" in chk_group
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
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to('cuda:0')
    return pipeline

def prepare_data(single_image, crop_size):
    dataset = SingleImageDataset(root_dir='', num_views=6, img_wh=[256, 256], bg_color='white', crop_size=crop_size, single_image=single_image)
    return dataset[0]

def run_pipeline(pipeline, cfg, single_image, guidance_scale, steps, seed, crop_size, chk_group=None):
    global scene
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
        scene = 'scene'+datetime.now().strftime('@%Y%m%d-%H%M%S')
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
    normals_pred = [save_image(normals_pred[i]) for i in range(bsz)]
    images_pred = [save_image(images_pred[i]) for i in range(bsz)]
    out = images_pred + normals_pred
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

def main(input_image, guidance_scale, steps, seed, output_dir):
    from utils.misc import load_config
    from omegaconf import OmegaConf

    # Load configuration
    cfg = load_config("./configs/mvdiffusion-joint-ortho-6views.yaml")
    schema = OmegaConf.structured(TestConfig)
    cfg = OmegaConf.merge(schema, cfg)

    # Initialize the pipeline and predictor
    pipeline = load_wonder3d_pipeline(cfg)
    torch.set_grad_enabled(False)
    pipeline.to(f'cuda:{_GPU_ID}')

    predictor = sam_init()

    # Load and preprocess the input image
    input_image = Image.open(input_image).convert("RGBA")
    processed_image_highres, processed_image = preprocess(predictor, input_image, ['Background Removal'])

    # Run the pipeline
    results = run_pipeline(pipeline, cfg, processed_image_highres, guidance_scale, steps, seed, 192, ['Write Results'])

    # Save the results
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        output_path = os.path.join(output_dir, f'output_{i+1}.png')
        save_image_to_disk(torch.tensor(result), output_path)

    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    fire.Fire(main)
