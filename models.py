"""
models.py — DR Analysis inference module
==========================================
Extracted from:
  - segmentation_STAGE1.ipynb  (FR-UNet vessel segmentation)
  - model1upgaded.ipynb        (ConvNeXt-Base ordinal DR grading)

Usage:
    from models import load_models, predict_single, compute_saliency_maps
    clf, seg, device = load_models()
    result  = predict_single(pil_image, clf, seg, device)
    saliency = compute_saliency_maps(pil_image, clf, device, result["grade"])
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRADE_NAMES   = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEG_IMG_SIZE  = (512, 512)   # FR-UNet trained at 512×512
CLF_IMG_SIZE  = (224, 224)   # ConvNeXt trained at 224×224
N_CLASSES     = 5
N_THRESHOLDS  = N_CLASSES - 1   # 4 ordinal logits

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLAHE_CLIP    = 2.0
CLAHE_GRID    = (8, 8)
SEG_THRESHOLD = 0.5   # fixed — not a user-facing parameter

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers  (from model1upgaded.ipynb, Cell 8)
# ─────────────────────────────────────────────────────────────────────────────

def crop_fundus(img_rgb: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Remove black border from a fundus image.
    Finds the bounding box of pixels brighter than tol and returns the crop.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows_on, cols_on = np.where(mask)
    if rows_on.size == 0:
        return img_rgb
    r0, r1 = int(rows_on.min()), int(rows_on.max())
    c0, c1 = int(cols_on.min()), int(cols_on.max())
    return img_rgb[r0:r1+1, c0:c1+1]


def apply_clahe(img_rgb: np.ndarray,
                clip_limit: float = CLAHE_CLIP,
                tile_grid: tuple  = CLAHE_GRID) -> np.ndarray:
    """
    Apply CLAHE on the L channel of LAB colour space.
    Enhances local contrast without hue shifts.
    """
    lab        = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b    = cv2.split(lab)
    enhancer   = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = enhancer.apply(l)
    lab_eq     = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)


def preprocess_for_classification(pil_image: Image.Image) -> torch.Tensor:
    """
    RGB → fundus crop → CLAHE → resize 224×224 → normalize → tensor
    Returns: (1, 3, 224, 224) float32
    """
    img = np.array(pil_image.convert("RGB"))
    img = crop_fundus(img)
    img = apply_clahe(img)
    img = cv2.resize(img, CLF_IMG_SIZE)
    tensor = TF.to_tensor(img)
    tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0)


def preprocess_for_segmentation(pil_image: Image.Image) -> torch.Tensor:
    """
    RGB → resize 512×512 → normalize → tensor
    Returns: (1, 3, 512, 512) float32
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, SEG_IMG_SIZE)
    tensor = TF.to_tensor(img)
    tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0)


def get_preprocessed_f32(pil_image: Image.Image) -> np.ndarray:
    """
    Returns the preprocessed image (crop + CLAHE + resize to 224×224)
    as float32 in [0, 1] — used as the base image for CAM overlays.
    """
    img = np.array(pil_image.convert("RGB"))
    img = crop_fundus(img)
    img = apply_clahe(img)
    img = cv2.resize(img, CLF_IMG_SIZE)
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────
# FR-UNet Architecture  (from segmentation_STAGE1.ipynb, Cell 11)
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv     = DoubleConv(in_channels, out_channels)
        self.downsamp = nn.Conv2d(out_channels, out_channels,
                                  kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        skip = self.conv(x)
        out  = self.downsamp(skip)
        return skip, out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv     = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FRBlock(nn.Module):
    def __init__(self, fr_channels, enc_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(enc_channels, fr_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fr_channels),
            nn.ReLU(inplace=True),
        )
        self.dw_conv = nn.Conv2d(fr_channels, fr_channels, kernel_size=3,
                                 padding=1, groups=fr_channels, bias=False)
        self.pw_conv = nn.Conv2d(fr_channels, fr_channels, kernel_size=1, bias=False)
        self.bn      = nn.BatchNorm2d(fr_channels)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, fr_feat, enc_feat, target_size):
        enc_proj = self.proj(enc_feat)
        enc_proj = F.interpolate(enc_proj, size=target_size,
                                 mode='bilinear', align_corners=True)
        fr_feat  = fr_feat + enc_proj
        fr_feat  = self.relu(self.bn(self.pw_conv(self.dw_conv(fr_feat))))
        return fr_feat


class FRUNet(nn.Module):
    """
    Full-Resolution UNet for retinal vessel segmentation.
    Input  : (B, 3, 512, 512)
    Output : (B, 1, 512, 512) raw logits — apply sigmoid for probabilities
    """
    def __init__(self, in_channels=3, base_channels=32, fr_channels=16):
        super().__init__()
        c = base_channels

        self.enc1 = EncoderBlock(in_channels, c)
        self.enc2 = EncoderBlock(c,     c * 2)
        self.enc3 = EncoderBlock(c * 2, c * 4)
        self.enc4 = EncoderBlock(c * 4, c * 8)

        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.dec4 = DecoderBlock(c * 16, c * 8,  c * 8)
        self.dec3 = DecoderBlock(c * 8,  c * 4,  c * 4)
        self.dec2 = DecoderBlock(c * 4,  c * 2,  c * 2)
        self.dec1 = DecoderBlock(c * 2,  c,      c)

        self.fr_init = nn.Sequential(
            nn.Conv2d(c, fr_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(fr_channels),
            nn.ReLU(inplace=True),
        )
        self.fr2   = FRBlock(fr_channels, c * 2)
        self.fr3   = FRBlock(fr_channels, c * 4)
        self.fr4   = FRBlock(fr_channels, c * 8)
        self.fr_bn = FRBlock(fr_channels, c * 16)

        self.fuse_conv = nn.Conv2d(c + fr_channels, c, kernel_size=3, padding=1, bias=False)
        self.fuse_bn   = nn.BatchNorm2d(c)
        self.fuse_relu = nn.ReLU(inplace=True)
        self.out_conv  = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        full_res_size = (x.shape[2], x.shape[3])

        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        fr = self.fr_init(skip1)
        fr = self.fr2(fr, skip2, full_res_size)
        fr = self.fr3(fr, skip3, full_res_size)
        fr = self.fr4(fr, skip4, full_res_size)
        fr = self.fr_bn(fr, x,   full_res_size)

        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        x = torch.cat([x, fr], dim=1)
        x = self.fuse_relu(self.fuse_bn(self.fuse_conv(x)))
        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Ordinal inference helpers  (from model1upgaded.ipynb, Cell 20)
# ─────────────────────────────────────────────────────────────────────────────

def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    """Decode K-1 ordinal threshold logits → integer grade (0…K-1)."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def ordinal_to_class_probs(logits: torch.Tensor, n_classes: int = N_CLASSES) -> np.ndarray:
    """
    Convert K-1 ordinal logits to per-class soft probabilities.
    Returns: (n_classes,) float array
    """
    thresh_p = torch.sigmoid(logits).cpu().numpy().flatten()
    cls_prob  = np.zeros(n_classes)
    cls_prob[0] = 1.0 - thresh_p[0]
    for k in range(1, n_classes - 1):
        cls_prob[k] = thresh_p[k - 1] - thresh_p[k]
    cls_prob[-1] = thresh_p[-1]
    return np.clip(cls_prob, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_models(
    clf_weight_path: str = "weights/convnext_ordinal_checkpoint.pth",
    seg_weight_path: str = "weights/best_weights_only.pth",
):
    """
    Load both models from local weight files.
    weights_only=False is required for checkpoints that contain numpy scalars
    (safe — these are your own trained weights).

    Returns: clf, seg, device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classification — ConvNeXt-Base with N_THRESHOLDS=4 outputs
    clf      = timm.create_model("convnext_base", pretrained=False, num_classes=N_THRESHOLDS)
    clf_ckpt = torch.load(clf_weight_path, map_location=device, weights_only=False)

    if isinstance(clf_ckpt, dict) and "weights" in clf_ckpt:
        clf.load_state_dict(clf_ckpt["weights"])
    elif isinstance(clf_ckpt, dict) and "state_dict" in clf_ckpt:
        clf.load_state_dict(clf_ckpt["state_dict"])
    else:
        clf.load_state_dict(clf_ckpt)
    clf.eval().to(device)

    # Segmentation — FR-UNet
    seg      = FRUNet(in_channels=3, base_channels=32, fr_channels=16)
    seg_ckpt = torch.load(seg_weight_path, map_location=device, weights_only=False)

    if isinstance(seg_ckpt, dict) and "model" in seg_ckpt:
        seg.load_state_dict(seg_ckpt["model"])
    elif isinstance(seg_ckpt, dict) and "state_dict" in seg_ckpt:
        seg.load_state_dict(seg_ckpt["state_dict"])
    else:
        seg.load_state_dict(seg_ckpt)
    seg.eval().to(device)

    return clf, seg, device


# ─────────────────────────────────────────────────────────────────────────────
# Single-image inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    pil_image: Image.Image,
    clf:       nn.Module,
    seg:       nn.Module,
    device:    torch.device,
):
    """
    Run classification + segmentation on one PIL image.

    Returns dict with:
        grade, grade_name, class_probs,
        mask_prob, mask_binary, overlay_rgb
    """
    # Classification
    clf_tensor  = preprocess_for_classification(pil_image).to(device)
    clf_logits  = clf(clf_tensor)
    grade       = int(ordinal_predict(clf_logits).item())
    grade       = max(0, min(grade, N_CLASSES - 1))
    class_probs = ordinal_to_class_probs(clf_logits)

    # Segmentation
    seg_tensor  = preprocess_for_segmentation(pil_image).to(device)
    seg_logits  = seg(seg_tensor)
    mask_prob   = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
    mask_binary = (mask_prob > SEG_THRESHOLD).astype(np.uint8)

    # Vessel overlay
    base_img    = np.array(pil_image.convert("RGB").resize(SEG_IMG_SIZE))
    overlay_rgb = base_img.copy()
    overlay_rgb[mask_binary == 1] = [220, 50, 50]

    return {
        "grade":       grade,
        "grade_name":  GRADE_NAMES[grade],
        "class_probs": class_probs,
        "mask_prob":   mask_prob,
        "mask_binary": mask_binary,
        "overlay_rgb": overlay_rgb,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM++ and EigenCAM  (from model1upgaded.ipynb, Cell 36)
# ─────────────────────────────────────────────────────────────────────────────

def compute_saliency_maps(
    pil_image: Image.Image,
    clf:       nn.Module,
    device:    torch.device,
    grade:     int,
):
    """
    Compute GradCAM++ and EigenCAM saliency maps for the predicted grade.

    Mirrors the exact approach from model1upgaded.ipynb Cell 36:
      - Target layer : clf.stages[-1]  (last ConvNeXt stage)
      - Target class : predicted grade index (clamped to N_THRESHOLDS-1)
      - Base image   : preprocessed (crop + CLAHE + resize to 224×224) as float32 [0,1]

    Parameters
    ----------
    pil_image : original uploaded PIL image
    clf       : loaded ConvNeXt model in eval mode
    device    : torch.device
    grade     : predicted integer grade from predict_single()

    Returns
    -------
    dict with keys:
        base_img_f32    : (224,224,3) float32 [0,1] — the preprocessed input shown as base
        cam_gradcam     : (224,224) float32 — raw GradCAM++ activation map
        overlay_gradcam : (224,224,3) uint8  — GradCAM++ heatmap blended on image
        cam_eigen       : (224,224) float32 — raw EigenCAM activation map
        overlay_eigen   : (224,224,3) uint8  — EigenCAM heatmap blended on image
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        raise ImportError(
            "grad-cam package not found. "
            "Install it with:  pip install grad-cam"
        )

    # Preprocessed float image for overlay
    base_img_f32 = get_preprocessed_f32(pil_image)   # (224,224,3) float32 [0,1]

    # Input tensor with grad enabled for GradCAM
    inp = preprocess_for_classification(pil_image).to(device)
    inp.requires_grad_(True)

    # Target layer: last ConvNeXt stage — identical to notebook
    target_layer = [clf.stages[-1]]

    # Target class: clamp to valid ordinal threshold index
    target_idx = min(grade, N_THRESHOLDS - 1)
    targets    = [ClassifierOutputTarget(target_idx)]

    # GradCAM++
    with GradCAMPlusPlus(model=clf, target_layers=target_layer) as gcam:
        cam_gradcam     = gcam(input_tensor=inp, targets=targets)[0]
        overlay_gradcam = show_cam_on_image(base_img_f32, cam_gradcam, use_rgb=True)

    # EigenCAM
    with EigenCAM(model=clf, target_layers=target_layer) as ecam:
        cam_eigen     = ecam(input_tensor=inp, targets=targets)[0]
        overlay_eigen = show_cam_on_image(base_img_f32, cam_eigen, use_rgb=True)

    return {
        "base_img_f32":    base_img_f32,
        "cam_gradcam":     cam_gradcam,
        "overlay_gradcam": overlay_gradcam,
        "cam_eigen":       cam_eigen,
        "overlay_eigen":   overlay_eigen,
    }
