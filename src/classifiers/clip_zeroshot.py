"""CLIP appearance classifier for the 4 appearance-defined classes.

Two scoring paths, chosen per class independently:

- **Zero-shot** (default, no training data needed): each class gets several
  text-prompt paraphrases (configs/classes.yaml) — a window's score for a
  class is the max image-text similarity across that class's paraphrases,
  then softmax-normalized across all zero-shot classes and an internal
  "background" pseudo-class together (see "Score normalization" in the
  design doc). Without the background class, a forced choice among only the
  real classes picks a "winner" even in an ordinary scene — observed
  directly during development (Vandalism spuriously spanning almost an
  entire ordinary pedestrian video; see report.md), not a hypothetical
  concern.
- **Trained few-shot probe** (opt-in, only if `checkpoints/clip_probes/<class>.pt`
  exists — train one with `train_clip_probe.py`): a logistic-regression head
  on the same frozen CLIP embedding, trained on labeled clips
  (`src/classifiers/few_shot_probe.py`). Scored independently per class, not
  mixed into the zero-shot softmax.

A class with a trained probe uses it exclusively; a class without one keeps
using zero-shot. Nothing needs to be configured to get this — it's automatic
based on what's present in checkpoints/clip_probes/.

## A known, deliberately-unfixed coupling issue

The shared N-way softmax means removing a class from it (by giving it a
trained probe) measurably changes the *remaining* zero-shot classes'
calibration — confirmed directly: with 3 classes moved to trained probes,
Theft's zero-shot score on an unchanged video jumped from 0.72 to 0.88
confidence, purely from having fewer competitors (see
`data/few_shot/PROVENANCE.md`).

A fix was built and measured: score every class independently as a binary
softmax against background only (no competition between real classes at
all). It does remove the coupling, and slightly raised true-positive
confidence on genuine detections (Fire/Smoke 0.99→1.00, Fighting 0.97→0.98).
But measured on `vtest.avi` (the one video with real ground truth), it also
let multiple classes clear threshold simultaneously where the original
competitive softmax had suppressed them — 3 new Vandalism false positives
appeared, and F1@tIoU0.5 dropped from **0.909 to 0.714** (best achievable
with re-tuning the threshold: 0.727, still well below baseline). The
competitive N-way softmax was doing real false-positive suppression work
that an independent per-class check doesn't replicate.

**Decision: reverted, not shipped.** The coupling bug only manifests when
trained probes are partially adopted (some classes probed, some not) — which
isn't the current state (no probes are currently deployed; see
`data/few_shot/PROVENANCE.md` for why the trained ones underperformed and
were reverted too). Paying a real, measured accuracy cost today to fix a bug
that isn't currently active isn't a good trade. Revisit this specific fix
(or a better one — e.g. isotonic recalibration per class instead of a full
architecture swap) at the point partial probe deployment actually happens.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.classifiers.few_shot_probe import LinearProbe, load_probe
from src.utils import resolve_device
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

APPEARANCE_CLASSES = ("fire_smoke", "fighting", "vandalism", "theft")
BACKGROUND_CLASS = "background"


class CLIPZeroShotClassifier:
    def __init__(
        self,
        classes_config: dict[str, Any],
        device: str = "auto",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        checkpoint_dir: str | Path = "checkpoints/clip_probes",
    ):
        import open_clip

        self.device = resolve_device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        self.class_keys = [c for c in APPEARANCE_CLASSES if c in classes_config]

        self.probes: dict[str, LinearProbe] = {}
        for class_key in self.class_keys:
            probe = load_probe(class_key, checkpoint_dir)
            if probe is not None:
                self.probes[class_key] = probe
                logger.info("Loaded trained few-shot probe for '%s' from %s", class_key, checkpoint_dir)

        self.zero_shot_keys = [c for c in self.class_keys if c not in self.probes]
        self.has_background = BACKGROUND_CLASS in classes_config
        softmax_keys = self.zero_shot_keys + ([BACKGROUND_CLASS] if self.has_background else [])

        self.prompt_features: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for class_key in softmax_keys:
                prompts = classes_config[class_key]["prompts"]
                tokens = tokenizer(prompts).to(self.device)
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self.prompt_features[class_key] = feats
        self._softmax_keys = softmax_keys

    def score_window(self, frames_bgr: list[np.ndarray]) -> dict[str, float]:
        """Returns {class_key: probability} for one window's sampled frames —
        trained-probe classes and zero-shot classes both included, scored by
        whichever mechanism applies to each."""
        images = []
        for frame in frames_bgr:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(self.preprocess(Image.fromarray(rgb)))
        batch = torch.stack(images).to(self.device)

        with torch.no_grad():
            img_feats = self.model.encode_image(batch)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            window_feat = img_feats.mean(dim=0, keepdim=True)
            window_feat = window_feat / window_feat.norm(dim=-1, keepdim=True)

            scores: dict[str, float] = {}

            for class_key, probe in self.probes.items():
                logit = probe(window_feat.float().cpu())
                scores[class_key] = torch.sigmoid(logit).item()

            if self._softmax_keys:
                max_sims = []
                for class_key in self._softmax_keys:
                    sims = (window_feat @ self.prompt_features[class_key].T).squeeze(0)
                    max_sims.append(sims.max().item())

                logit_scale = self.model.logit_scale.exp().item()
                logits = np.array(max_sims) * logit_scale
                probs = np.exp(logits - logits.max())
                probs = probs / probs.sum()

                # background competed in the softmax above but is never returned —
                # it exists only so the zero-shot classes can lose to "nothing happening"
                for class_key, p in zip(self._softmax_keys, probs):
                    if class_key != BACKGROUND_CLASS:
                        scores[class_key] = float(p)

        return scores
