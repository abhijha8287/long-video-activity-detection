#!/usr/bin/env python3
"""Train few-shot linear probes on CLIP embeddings for the appearance classes.

    python train_clip_probe.py

Reads labeled clips or images from data/few_shot/<class_key>/{positive,negative}/*
(video: mp4/avi/webm/mov/mkv, image: jpg/jpeg/png/webp/bmp — freely mixed),
trains one logistic-regression probe per class with enough examples, and
saves each to checkpoints/clip_probes/<class_key>.pt. inference.py picks up
any trained probe automatically (see src/classifiers/clip_zeroshot.py) —
classes without a trained probe keep using zero-shot, no config needed.

This is NOT a substitute for real evaluation: train_accuracy printed below is
fit on the same tiny set it trained on (no held-out split — there usually
isn't enough data for one), so it will look artificially good. It's a sanity
check that training ran, not a performance claim. See --demo for an explicit,
clearly-labeled smoke test using the bundled sample videos instead of real
labeled data.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import src.classifiers.few_shot_probe as few_shot_probe
from src.classifiers.few_shot_probe import (
    ClipEmbedder,
    discover_few_shot_classes,
    save_probe,
    train_probe,
)
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train few-shot CLIP linear probes.")
    p.add_argument("--data-dir", default="data/few_shot")
    p.add_argument("--checkpoint-dir", default="checkpoints/clip_probes")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument(
        "--demo", action="store_true",
        help="Build a throwaway data/few_shot/ from the bundled sample videos (1 positive clip per "
        "class, sample videos from OTHER classes as negatives) and train on that. This is a code-"
        "correctness smoke test, not meaningful training — 1 example per class cannot generalize. "
        "Says so loudly before and after running.",
    )
    p.add_argument(
        "--force-min-examples", type=int, default=None,
        help="Override the minimum examples-per-class safety guard (default: refuse below "
        "MIN_EXAMPLES_PER_CLASS=%d). Only for forcing a code-path demonstration on too-thin data "
        "(e.g. --demo) — never use this to sneak a real training run past the guard; it exists "
        "because a probe trained on <4 examples per side has learned noise, not a class."
        % few_shot_probe.MIN_EXAMPLES_PER_CLASS,
    )
    return p.parse_args()


def build_demo_dataset(data_dir: Path) -> None:
    """One labeled positive per appearance class from data/samples/, using the
    OTHER classes' sample clips (plus vtest.avi, which has none of the 4
    appearance classes) as negatives. Real per-class positive/negative pairs
    don't exist anywhere in this project — this is the closest honest
    approximation for proving the training code path actually works."""
    samples = Path("data/samples")
    class_clips = {
        "fire_smoke": [samples / "fire_smoke_01.mp4"],
        "fighting": [samples / "fighting_01.mp4", samples / "fighting_02.mp4"],
        "vandalism": [samples / "vandalism_01.mp4"],
        "theft": [],  # no sample clip exists for Theft — see data/samples/PROVENANCE.md
    }
    negative_pool = [samples / "vtest.avi"]  # no appearance-class content at all

    if data_dir.exists():
        shutil.rmtree(data_dir)
    for class_key, positives in class_clips.items():
        positives = [p for p in positives if p.exists()]
        if not positives:
            continue
        pos_dir = data_dir / class_key / "positive"
        neg_dir = data_dir / class_key / "negative"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)
        for p in positives:
            (pos_dir / p.name).symlink_to(p.resolve())
        # negatives: every OTHER class's clips, plus the shared negative pool
        others = [
            clip for k, clips in class_clips.items() if k != class_key for clip in clips if clip.exists()
        ] + negative_pool
        for n in others:
            dest = neg_dir / n.name
            if not dest.exists():
                dest.symlink_to(n.resolve())


def main() -> int:
    args = parse_args()
    setup_logging()

    data_dir = Path(args.data_dir)

    if args.demo:
        print(
            "\n*** --demo mode: building a throwaway few-shot dataset from data/samples/. ***\n"
            "*** This is a code-correctness smoke test — 1-2 clips per class cannot     ***\n"
            "*** generalize to anything. Do not treat the accuracy below as real.       ***\n"
        )
        build_demo_dataset(data_dir)

    if args.force_min_examples is not None:
        print(
            f"*** --force-min-examples={args.force_min_examples}: safety guard overridden. Any "
            f"probe trained below the real minimum ({few_shot_probe.MIN_EXAMPLES_PER_CLASS}) has "
            f"learned noise, not a class. For demonstration only. ***\n"
        )
        few_shot_probe.MIN_EXAMPLES_PER_CLASS = args.force_min_examples

    classes = discover_few_shot_classes(data_dir)
    if not classes:
        print(f"No labeled data found under {data_dir}/. Expected:")
        print(f"  {data_dir}/<class_key>/positive/*.{{mp4,jpg,png,...}}")
        print(f"  {data_dir}/<class_key>/negative/*.{{mp4,jpg,png,...}}")
        print("(--demo builds a throwaway version of this from data/samples/ for a smoke test)")
        return 1

    embedder = ClipEmbedder(device=args.device)
    trained_any = False
    for class_key, clips in classes.items():
        result = train_probe(
            embedder, class_key, clips["positive"], clips["negative"], epochs=args.epochs,
        )
        if result is None:
            continue
        probe, stats = result
        path = save_probe(probe, class_key, args.checkpoint_dir)
        print(
            f"{class_key}: trained on {stats.num_positive_frames} positive / "
            f"{stats.num_negative_frames} negative frames, train_accuracy={stats.train_accuracy:.2f} "
            f"-> {path}"
        )
        trained_any = True

    if not trained_any:
        print("\nNo class had enough labeled clips to train (need >=4 positive and >=4 negative each).")
        return 1

    print(
        "\nDone. inference.py will use these probes automatically for classes that have one, "
        "falling back to zero-shot for classes that don't."
    )
    if args.demo:
        print(
            "\nReminder: this ran on the --demo dataset (1-2 clips per class from data/samples/), "
            "not real labeled data. Delete checkpoints/clip_probes/ before using this for anything "
            "beyond confirming the code runs — see README.md's few-shot probe section."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
