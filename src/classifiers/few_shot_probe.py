"""Few-shot linear probe on CLIP embeddings for the appearance classes.

This is the "trained head" explicitly deferred in the design doc
(docs/designs/activity-detection-baseline.md) — CLIP zero-shot (clip_zeroshot.py)
needs no labeled data; this module is what you reach for once labeled clips
exist for a class, per the design doc's own future-work note.

Data format (one labeled example = one short video clip OR one still image —
freely mixed within a folder):

    data/few_shot/<class_key>/positive/*.{mp4,avi,webm,mov,mkv,jpg,jpeg,png,webp}
    data/few_shot/<class_key>/negative/*.{mp4,avi,webm,mov,mkv,jpg,jpeg,png,webp}

`<class_key>` matches a key in configs/classes.yaml (fire_smoke, fighting,
vandalism, theft). A class with too few examples (see MIN_EXAMPLES_PER_CLASS)
is skipped, loudly, rather than trained on data too thin to mean anything.

Images work as well as video clips here, and are often *better* few-shot
examples: CLIP scores a video window by averaging independent per-frame
embeddings (see clip_zeroshot.py::score_window) — a still image and a video
frame are the same kind of input to the model. A handful of frames sampled
from one clip are highly correlated (same scene, ~1s apart), while N distinct
photos are N genuinely independent examples. Images are also far easier to
source in bulk with clear licensing than short video clips.

A trained probe is a single linear layer (logistic regression) on top of
frozen CLIP image embeddings — cheap enough to train on CPU in seconds, and
simple enough to explain in an interview without hand-waving. Deliberately
not a deeper head: with few-shot data, more parameters means more overfitting,
not more accuracy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.utils import resolve_device

logger = logging.getLogger(__name__)

MIN_EXAMPLES_PER_CLASS = 4  # per side (positive/negative) — below this, training is refused, not attempted on noise
FRAMES_PER_CLIP = 6         # sampled evenly across each labeled video clip, each frame is one training example

VIDEO_EXTS = {".mp4", ".avi", ".webm", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ProbeTrainResult:
    class_key: str
    num_positive_frames: int
    num_negative_frames: int
    train_accuracy: float
    epochs: int


def _sample_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    indices = np.linspace(0, max(frame_count - 1, 0), n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def discover_few_shot_classes(data_dir: str | Path = "data/few_shot") -> dict[str, dict[str, list[Path]]]:
    """Returns {class_key: {"positive": [paths], "negative": [paths]}} for every
    class subdirectory found, regardless of whether it has enough examples —
    callers decide what to do with too-thin classes."""
    data_dir = Path(data_dir)
    found: dict[str, dict[str, list[Path]]] = {}
    if not data_dir.exists():
        return found
    media_exts = VIDEO_EXTS | IMAGE_EXTS
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        pos_dir, neg_dir = class_dir / "positive", class_dir / "negative"
        if not (pos_dir.exists() and neg_dir.exists()):
            continue
        pos = sorted(p for p in pos_dir.iterdir() if p.suffix.lower() in media_exts)
        neg = sorted(p for p in neg_dir.iterdir() if p.suffix.lower() in media_exts)
        found[class_dir.name] = {"positive": pos, "negative": neg}
    return found


class ClipEmbedder:
    """Wraps the CLIP image encoder only — shared by zero-shot scoring and
    probe training/inference so embeddings are computed identically in both
    (no train/inference skew from e.g. different preprocessing)."""

    def __init__(self, device: str = "auto", model_name: str = "ViT-B-32", pretrained: str = "openai"):
        import open_clip

        self.device = resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def embed_frames(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """Returns L2-normalized CLIP embeddings, shape [n_frames, embed_dim]."""
        images = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(self.preprocess(Image.fromarray(rgb)))
        batch = torch.stack(images).to(self.device)
        feats = self.model.encode_image(batch)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def embed_video(self, video_path: Path, n_frames: int = FRAMES_PER_CLIP) -> torch.Tensor:
        frames = _sample_frames(video_path, n_frames)
        if not frames:
            raise ValueError(f"Could not read any frames from {video_path}")
        return self.embed_frames(frames)

    @torch.no_grad()
    def embed_image(self, image_path: Path) -> torch.Tensor:
        """Returns a single L2-normalized CLIP embedding, shape [1, embed_dim]."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image {image_path}")
        return self.embed_frames([frame])

    def embed_media(self, path: Path, n_frames: int = FRAMES_PER_CLIP) -> torch.Tensor:
        """Dispatches to embed_image or embed_video based on file extension —
        the one entry point train_probe uses so callers don't need to care
        which kind of file a labeled example is."""
        path = Path(path)
        if path.suffix.lower() in IMAGE_EXTS:
            return self.embed_image(path)
        return self.embed_video(path, n_frames)


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_probe(
    embedder: ClipEmbedder,
    class_key: str,
    positive_videos: list[Path],
    negative_videos: list[Path],
    epochs: int = 200,
    lr: float = 0.05,
    weight_decay: float = 0.05,
    seed: int = 42,
) -> tuple[LinearProbe, ProbeTrainResult] | None:
    """Trains one logistic-regression probe for one class. Returns None (and
    logs why) if either side has fewer than MIN_EXAMPLES_PER_CLASS clips —
    training on fewer examples than that produces a probe that has memorized
    noise, not learned a class, so it's refused rather than silently attempted.
    """
    if len(positive_videos) < MIN_EXAMPLES_PER_CLASS or len(negative_videos) < MIN_EXAMPLES_PER_CLASS:
        logger.warning(
            "Skipping '%s': needs >=%d positive and >=%d negative clips, has %d/%d.",
            class_key, MIN_EXAMPLES_PER_CLASS, MIN_EXAMPLES_PER_CLASS,
            len(positive_videos), len(negative_videos),
        )
        return None

    torch.manual_seed(seed)
    pos_embeds = torch.cat([embedder.embed_media(p) for p in positive_videos], dim=0)
    neg_embeds = torch.cat([embedder.embed_media(p) for p in negative_videos], dim=0)
    X = torch.cat([pos_embeds, neg_embeds], dim=0).float().cpu()
    y = torch.cat([torch.ones(len(pos_embeds)), torch.zeros(len(neg_embeds))]).float()

    probe = LinearProbe(X.shape[1])
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(probe(X)) >= 0.5).float()
        train_accuracy = (preds == y).float().mean().item()

    result = ProbeTrainResult(
        class_key=class_key,
        num_positive_frames=len(pos_embeds),
        num_negative_frames=len(neg_embeds),
        train_accuracy=train_accuracy,
        epochs=epochs,
    )
    return probe, result


def save_probe(probe: LinearProbe, class_key: str, checkpoint_dir: str | Path = "checkpoints/clip_probes") -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{class_key}.pt"
    torch.save({"state_dict": probe.state_dict(), "embed_dim": probe.linear.in_features}, path)
    return path


def load_probe(class_key: str, checkpoint_dir: str | Path = "checkpoints/clip_probes") -> LinearProbe | None:
    path = Path(checkpoint_dir) / f"{class_key}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    probe = LinearProbe(ckpt["embed_dim"])
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    return probe


def score_with_probe(probe: LinearProbe, embedding: torch.Tensor) -> float:
    """embedding: single L2-normalized CLIP embedding, shape [embed_dim]."""
    with torch.no_grad():
        logit = probe(embedding.float().cpu().unsqueeze(0))
        return torch.sigmoid(logit).item()
