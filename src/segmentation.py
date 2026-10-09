"""
Leaf segmentation — cut the leaf out of its background before classification.

Why: every PlantVillage photo is a single leaf on the same plain lab background,
so a classifier trained on them can lean on background and lighting cues that
do not exist in field photos (soil, grass, other leaves, hands). Cutting the
leaf out — and, during training, pasting it onto random backgrounds — forces
the model to look at the leaf itself.

Algorithm (classical, no extra model weights):
  1. Colour prior — saturation + excess-green flag "plant-coloured" pixels.
     Lesion colours (yellow, brown, rust) are saturated; PlantVillage's
     grey/beige background is not.
  2. GrabCut — seeded with: thin image border = background, plant-coloured
     pixels = probable leaf, plant-coloured pixels in the image centre =
     leaf. GrabCut refines this with colour GMMs and an edge-aware graph cut,
     which copes with field backgrounds where colour alone fails.
  3. Clean-up — keep the component that overlaps the centre most, fill holes
     so lesions / insect holes inside the leaf are kept, smooth the edge.
  4. Sanity check — if the mask is implausibly small or covers the whole
     frame, fall back to the full image rather than a broken cutout.

The cutout is cropped to the leaf's bounding box and padded to a square, which
also normalises scale: a leaf that fills 20% of a phone photo ends up the same
size as a PlantVillage leaf.

Usage:
    # Precompute masks for a whole ImageFolder dataset (run once):
    python -m src.segmentation data/raw/PlantVillage data/processed/PlantVillage_masks

    # Single image:
    from src.segmentation import cut_out_leaf, composite
    rgba, ok = cut_out_leaf(Image.open("leaf.jpg"))
    rgb = composite(rgba)          # leaf on black, ready for the classifier
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

WORK_SIZE = 128          # GrabCut runs at this resolution; the mask is upsampled after.
GRABCUT_ITERS = 2        # 128px/2 iters ≈ 30 ms/image, masks match 256px/4 iters (IoU 0.96) at ~8x the speed
MIN_AREA_FRAC = 0.03     # smaller masks are treated as failures
MAX_AREA_FRAC = 0.97     # "everything is leaf" is a failure too (or a full-frame close-up)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _plant_prior(rgb: np.ndarray) -> np.ndarray:
    """Boolean map of pixels whose colour looks like leaf tissue (healthy or diseased)."""
    f = rgb.astype(np.float32)
    total = f.sum(axis=2) + 1e-6
    r, g, b = (f[..., i] / total for i in range(3))
    exg = 2 * g - r - b                                     # excess-green index
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    # OpenCV hue is 0–180: ~10–90 covers brown → yellow → green
    leafy_hue = (hue >= 10) & (hue <= 90)
    return ((exg > 0.06) | (leafy_hue & (sat > 0.30))) & (val > 0.12)


def _centre_ellipse(h: int, w: int, radius: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - h / 2) / (h * radius)) ** 2 + ((xx - w / 2) / (w * radius)) ** 2 <= 1


BORDER_MATCH_FRAC = 0.6  # share of border pixels that must look like the centre leaf
BORDER_MATCH_DE = 18.0   # Lab colour distance counted as "looks like the centre leaf"


def _border_matches_centre(rgb: np.ndarray, core: np.ndarray, border: np.ndarray) -> bool:
    """True if most of the image border has the same colour as the centre (leaf fills the frame)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255                     # OpenCV 8-bit Lab → L in 0–100, a/b offset by 128
    centre = np.median(lab[core], axis=0)
    dist = np.linalg.norm(lab[border] - centre, axis=1)
    return (dist < BORDER_MATCH_DE).mean() >= BORDER_MATCH_FRAC


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background regions (lesions, holes) inside the leaf."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (mask | (flood[1:-1, 1:-1] == 0)).astype(np.uint8)


def leaf_mask(img: Image.Image) -> tuple[np.ndarray, bool]:
    """
    Segment the main leaf.

    Returns:
        mask: uint8 array (H, W) at the input resolution, 255 = leaf.
        ok:   False if segmentation failed — the mask is then all 255 so
              callers fall back to the full image.
    """
    rgb_full = np.asarray(img.convert("RGB"))
    h0, w0 = rgb_full.shape[:2]
    scale = min(1.0, WORK_SIZE / max(h0, w0))
    rgb = cv2.resize(rgb_full, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                     interpolation=cv2.INTER_AREA) if scale < 1 else rgb_full
    h, w = rgb.shape[:2]

    plant = _plant_prior(rgb)
    core = _centre_ellipse(h, w, 0.18)
    border = np.zeros((h, w), bool)
    bw = max(2, round(0.02 * min(h, w)))
    border[:bw], border[-bw:], border[:, :bw], border[:, -bw:] = True, True, True, True

    gc = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    plant_frac = plant.mean()
    if 0.02 < plant_frac < 0.6:
        # Colour prior is informative (plain / non-plant background)
        gc[plant] = cv2.GC_PR_FGD
        gc[core & plant] = cv2.GC_FGD
        gc[border & ~plant] = cv2.GC_BGD
    else:
        # Colour prior found nothing, or the whole frame looks like plant
        # (foliage, soil, wooden table): rely on the photo being centred on
        # the leaf and let GrabCut's colour model separate it from the rest.
        if _border_matches_centre(rgb, core, border):
            # The leaf (or look-alike foliage) runs off the frame: any cut would
            # carve an arbitrary blob out of it, so keep the whole image.
            return np.full((h0, w0), 255, np.uint8), False
        gc[_centre_ellipse(h, w, 0.40)] = cv2.GC_PR_FGD
        gc[core] = cv2.GC_FGD
        gc[border] = cv2.GC_BGD

    try:
        cv2.setRNGSeed(0)   # GrabCut's GMM init is random — fix it so masks are reproducible
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc, None, bgd, fgd,
                    GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.full((h0, w0), 255, np.uint8), False
    mask = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)

    # Clean-up: drop specks, keep the component that best covers the centre
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.full((h0, w0), 255, np.uint8), False
    centre = _centre_ellipse(h, w, 0.35)
    scores = [(np.count_nonzero(centre & (labels == i)), stats[i, cv2.CC_STAT_AREA], i)
              for i in range(1, n)]
    best = max(scores)[2]
    mask = _fill_holes((labels == best).astype(np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    frac = mask.mean()
    if not (MIN_AREA_FRAC <= frac <= MAX_AREA_FRAC):
        return np.full((h0, w0), 255, np.uint8), False

    mask = cv2.GaussianBlur(mask.astype(np.float32) * 255, (3, 3), 0)   # soft edge
    if scale < 1:
        mask = cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return mask.clip(0, 255).astype(np.uint8), True


def apply_mask(img: Image.Image, mask: np.ndarray, margin: float = 0.06) -> Image.Image:
    """
    Attach `mask` as alpha, crop to the leaf's bounding box (+margin) and pad
    to a transparent square. Returns an RGBA image.
    """
    rgba = img.convert("RGB")
    if mask.shape != (rgba.height, rgba.width):
        mask = cv2.resize(mask, rgba.size, interpolation=cv2.INTER_LINEAR)
    rgba.putalpha(Image.fromarray(mask))

    ys, xs = np.nonzero(mask > 127)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        m = round(margin * max(x1 - x0, y1 - y0))
        rgba = rgba.crop((max(0, x0 - m), max(0, y0 - m),
                          min(rgba.width, x1 + m), min(rgba.height, y1 + m)))

    side = max(rgba.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
    return square


def cut_out_leaf(img: Image.Image, margin: float = 0.06) -> tuple[Image.Image, bool]:
    """Segment and crop the leaf. Returns (RGBA square cutout, segmentation_ok)."""
    mask, ok = leaf_mask(img)
    return apply_mask(img, mask, margin), ok


def composite(rgba: Image.Image, background: Image.Image | tuple = (0, 0, 0)) -> Image.Image:
    """Paste an RGBA cutout onto a background (solid colour or image). Returns RGB."""
    if isinstance(background, Image.Image):
        bg = background.convert("RGB").resize(rgba.size)
    else:
        bg = Image.new("RGB", rgba.size, background)
    bg.paste(rgba, (0, 0), rgba)
    return bg


class RandomBackground:
    """
    Training augmentation: paste an RGBA leaf cutout onto a random background
    so the classifier cannot use the background as a cue. Black is included
    because that is what inference composites onto.

    backgrounds_dir: optional folder of real background photos (soil, grass,
    canopy, hands...). Random crops of these are used ~25% of the time.
    """

    def __init__(self, backgrounds_dir: str | None = None, p_black: float = 0.3):
        self.p_black = p_black
        self.backgrounds = sorted(p for p in Path(backgrounds_dir).rglob("*")
                                  if p.suffix.lower() in IMG_EXTS) if backgrounds_dir else []

    @staticmethod
    def _texture(size: tuple[int, int]) -> Image.Image:
        # Low-res colour noise upsampled → smooth blotches (out-of-focus soil/foliage)
        k = random.randint(2, 12)
        base = [random.randint(0, 255) for _ in range(3)]
        spread = random.randint(10, 80)
        px = bytes(max(0, min(255, c + random.randint(-spread, spread)))
                   for _ in range(k * k) for c in base)
        return Image.frombytes("RGB", (k, k), px).resize(size, Image.BICUBIC)

    def _photo(self, size: tuple[int, int]) -> Image.Image:
        with Image.open(random.choice(self.backgrounds)) as im:
            im = im.convert("RGB")
            side = min(im.size)
            x, y = random.randint(0, im.width - side), random.randint(0, im.height - side)
            return im.crop((x, y, x + side, y + side)).resize(size)

    def __call__(self, rgba: Image.Image) -> Image.Image:
        r = random.random()
        if r < self.p_black:
            bg = (0, 0, 0)
        elif r < self.p_black + 0.2:
            bg = tuple(random.randint(0, 255) for _ in range(3))
        elif self.backgrounds and r < self.p_black + 0.45:
            bg = self._photo(rgba.size)
        else:
            bg = self._texture(rgba.size)
        return composite(rgba, bg)


# ── Precompute masks for an ImageFolder dataset ──────────────────────────────

def mask_path_for(image_path: Path, image_root: Path, mask_root: Path) -> Path:
    return (mask_root / image_path.relative_to(image_root)).with_suffix(".png")


def _segment_one(args: tuple[str, str]) -> bool:
    src, dst = args
    try:
        with Image.open(src) as im:
            mask, ok = leaf_mask(im)
    except OSError:
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(dst, optimize=True)
    return ok


def segment_dataset(image_root: str, mask_root: str, workers: int | None = None) -> None:
    """
    Write one PNG mask per image, mirroring image_root's folder layout.
    Masks are small (a few KB each); originals are left untouched. Images whose
    mask already exists are skipped, so the job can be resumed.
    """
    image_root, mask_root = Path(image_root), Path(mask_root)
    jobs = [(str(p), str(mask_path_for(p, image_root, mask_root)))
            for p in sorted(image_root.rglob("*"))
            if p.suffix.lower() in IMG_EXTS
            # skip hidden folders (.git) inside the dataset — not in image_root's own path ("../data")
            and not any(part.startswith(".") for part in p.relative_to(image_root).parts)]
    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"{len(jobs):,} images, {len(jobs) - len(todo):,} already done, {len(todo):,} to segment")

    failed = 0
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for i, ok in enumerate(pool.map(_segment_one, todo, chunksize=32), 1):
            failed += not ok
            if i % 2000 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,}  (fallback to full image: {failed:,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Precompute leaf masks for an ImageFolder dataset.")
    ap.add_argument("image_root")
    ap.add_argument("mask_root")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    segment_dataset(a.image_root, a.mask_root, a.workers)
