from trident import OpenSlideWSI
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple
import os 
import torch
from torch.amp import autocast

def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create the heatmap overlay based on scores and coordinates.
    
    Args:
        scores (np.ndarray): Normalized scores.
        coords (np.ndarray): Coordinates of patches.
        patch_size_level0 (int): Patch size at level 0.
        scale (np.ndarray): Scaling factors.
        region_size (Tuple[int, int]): Dimensions of the region.
    
    Returns:
        np.ndarray: Heatmap overlay.
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords = np.ceil(coords * scale).astype(int)
    
    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)
    
    for idx, coord in enumerate(coords):
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += scores[idx]
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1
    
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # Set areas with no data to NaN
    
    return overlay

def apply_colormap(overlay: np.ndarray, cmap_name: str):
    """
    Apply a colormap to the heatmap overlay and prepare a mappable for colorbar.
    
    Args:
        overlay (np.ndarray): Heatmap overlay.
        cmap_name (str): Colormap name.

    Returns:
        Tuple[np.ndarray, matplotlib.cm.ScalarMappable]:
            - Colored overlay image (uint8 RGB).
            - Mappable for creating a colorbar.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid

    # prepare mappable for colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(overlay[valid_mask])

    return overlay_colored, mappable

def normalize_attn(attn, clip=(1, 99), eps=1e-8):
    """Return attention normalized to [0,1], optional percentile clipping."""
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().float().cpu().numpy()
    attn = np.asarray(attn)
    attn = np.squeeze(attn)  # e.g., [1,1,N] -> [N]
    if attn.ndim != 1:
        raise ValueError(f"Expected flat 1D attention after squeeze, got {attn.shape}")
    if clip is not None:
        lo, hi = np.percentile(attn, [clip[0], clip[1]])
        attn = np.clip(attn, lo, hi)
    a_min, a_max = attn.min(), attn.max()
    if (a_max - a_min) < eps:
        return np.zeros_like(attn, dtype=np.float32)
    return ((attn - a_min) / (a_max - a_min + eps)).astype(np.float32)

def visualize_attn(de_id: str,
                   model,
                   preloaded_features,
                   preloaded_coords,
                   data_path: str,
                   labels,
                   data_format: str = 'svs',
                   save_dir = None,
                   vis_level = 2,
                   precision = 16,
                   cmap: Optional[str] = 'jet',
                   top_k = 10,
                   thumb_factor = 1,
                   show_top_k = True,
                   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    features = preloaded_features[de_id]
    coords = preloaded_coords[de_id]['coords']
    coords_attrs = preloaded_coords[de_id]['coords_attrs']
    
    with torch.no_grad():
        model.eval()
        features = features.unsqueeze(0).to(device)
        # autocast for mixed precision
        with autocast(device_type=device.type, enabled=(precision == 16)):
            logit, log_dict = model(features, return_raw_attention=True)
            attn = log_dict['attention']
            if model.n_classes == 1:
                pred = torch.sigmoid(logit).cpu().numpy().squeeze()
                pred_label = (pred >= 0.5).astype(int)
                actual_label = labels[de_id].item()
                annotation = f"Pred: {pred_label} (Raw: {pred:.4f}) | Actual: {actual_label}"

            else:
                pred = torch.softmax(logit, dim=-1).cpu().numpy().squeeze()
                pred_label = np.argmax(pred, axis=-1)
                actual_label = labels[de_id].item()
                annotation = f"Pred: {pred_label} (Raw: {pred}) | Actual: {actual_label}"

    print(annotation)
    print(f"Attention shape: {attn.shape}")

    slide = OpenSlideWSI(slide_path=f'{data_path}/{de_id}.{data_format}', lazy_init=False)
    scores = normalize_attn(attn, clip=(1, 99), eps=1e-8)
    
    if len(slide.level_dimensions) <= vis_level:
        print(f"Warning: Visualization level {vis_level} exceeds available levels in the slide. Using last level {len(slide.level_dimensions) - 1}.")
        target_level = vis_level
        vis_level = len(slide.level_dimensions) - 1
        # using thumb_factor to downsample the image
        # 1 level means 2 times downsample
        thumb_factor = 2 ** (vis_level - target_level)
        print(f"Using thumb factor: {thumb_factor}")
    else:
        vis_level = vis_level
        thumb_factor = 1  # no downsampling if using the same level
        
    downsample = slide.level_downsamples[vis_level]
    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(slide.level_dimensions[0]) * scale).astype(int))
    print(f"Region size at level {vis_level}: {region_size}")

    overlay = create_overlay(scores = scores, 
                            coords = coords, 
                            patch_size_level0 = coords_attrs['patch_size_level0'], 
                            scale = scale, 
                            region_size = region_size)
    
    overlay_colored, mappable = apply_colormap(overlay, cmap)
    print(f"Colored Overlay shape: {overlay_colored.shape}")

    img_pil = slide.read_region((0,0), vis_level, slide.level_dimensions[vis_level]) \
        .convert("RGB") \
        .resize(region_size, resample=Image.Resampling.BICUBIC)

    blended_pil = Image.fromarray(
        cv2.addWeighted(
            np.array(img_pil), 0.6,
            overlay_colored,  0.4,
            0
        )
    )
    
    w, h = img_pil.size
    thumb_size = (w // thumb_factor, h // thumb_factor)
    img_thumb     = img_pil.resize(thumb_size,     resample=Image.Resampling.LANCZOS)
    blended_thumb = blended_pil.resize(thumb_size, resample=Image.Resampling.LANCZOS)

    # convert for plotting
    thumb_np   = np.array(img_thumb)
    blend_np   = np.array(blended_thumb)

    # 2d) Plot side-by-side with colorbar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax1.imshow(thumb_np)
    ax1.set_title(f"{de_id} — Raw WSI")
    ax1.axis('off')

    ax2.imshow(blend_np)
    ax2.set_title(f"{de_id} — Blended")
    ax2.axis('off')

    # add shared colorbar
    cbar = fig.colorbar(
        mappable,
        ax=[ax1, ax2],
        fraction=0.04,
        pad=0.02,
        label="Normalized Attention Score"
    )
    
    fig.text(
        0.5,  # x-position (center)
        0.01, # y-position (just above the bottom of the figure)
        annotation,
        ha='center',
        va='bottom',
        fontsize=14
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"attention_overlay.png"))
        # print(f"Saved attention overlay to {os.path.join(save_dir, f'attention_overlay.png')}")
        
    plt.show()   
    plt.close(fig)
    
    if save_dir:
        # save the blended image
        blended_save_path = os.path.join(save_dir, f"blended_attention.png")
        # convert blend_np to PIL Image and save
        blended_save = Image.fromarray(blend_np)
        blended_save.save(blended_save_path)
    
    # 3) Show top-k patches based on attention scores
    topk_indices = np.argsort(scores)[-top_k:]

    for idx, i in enumerate(topk_indices):
        x, y = coords[i]
        patch = slide.read_region((x, y), 0, (coords_attrs['patch_size_level0'], coords_attrs['patch_size_level0']))
        # show patch
        plt.figure(figsize=(5, 5))
        plt.imshow(patch)
        plt.axis('off')
        plt.title(f"Patch {idx + 1} (Score: {scores[i]:.2f})")
        
        if save_dir:
            os.makedirs(os.path.join(save_dir, "top_patches"), exist_ok=True)
            patch_save_path = os.path.join(save_dir, "top_patches", f"{de_id}_patch_{idx + 1}.png")
            patch.save(patch_save_path)
            # print(f"Saved patch {idx + 1} to {patch_save_path}")
            
        if show_top_k:
            plt.show()
        else:
            plt.close()