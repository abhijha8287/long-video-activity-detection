# Activity Detection & Classification in Long Videos

Detects the time intervals in a long, untrimmed video where an activity occurs and
classifies each interval into one of 10 target classes, with a confidence score.
Built as a technical take-home baseline — see `report.md` for the full design
rationale, measured results, and honest failure analysis. This file is setup + run
instructions only.

## Problem definition

Given a long video that may contain multiple activities, idle/background periods,
and activities of varying duration, output predicted segments:

```json
{"start": 12.5, "end": 25.3, "label": "Fighting / Physical Assault", "confidence": 0.87}
```

## Activity classes

1. Fire / Smoke
2. Fighting / Physical Assault
3. Falling (person collapse)
4. Intrusion / Trespassing
5. Loitering
6. Vandalism
7. Theft / Chain Snatching
8. Crowd Formation / Unusual Gathering
9. Running (in a non-running-permitted zone)
10. Abandoned Object

## Approach in one paragraph

No public dataset has temporal ground truth for all 10 classes at once, and 6 of
them (Falling, Intrusion, Loitering, Crowd Formation, Running, Abandoned Object)
are defined by *position and motion over time relative to a scene*, not by
appearance — a pretrained video classifier has no notion of "restricted zone." So
the system is a two-branch hybrid, decided in `docs/designs/activity-detection-baseline.md`
after inspecting this exact constraint set (no GPU, no provided dataset):

- **Geometry branch** (`src/rules/`) — YOLO11n detection + ByteTrack tracking feed
  a deterministic rule engine, one rule per class (zone occupancy, dwell time,
  velocity, density, bbox-shape change, owner separation). No training data needed.
- **Appearance branch** (`src/classifiers/`) — CLIP (ViT-B/32) zero-shot image-text
  similarity over sliding windows, for Fire/Smoke, Fighting, Vandalism, Theft. An
  internal "background" pseudo-class competes in the same softmax so an ordinary
  scene can score low on all 4 real classes instead of one always "winning" —
  added after this was observed as a real failure mode during development (see
  report.md §8).

Both branches emit per-timestamp/per-window confidence scores in `[0,1]`;
`src/pipeline/segment_builder.py` applies shared smoothing, thresholding, and
merging to turn them into the final `{start, end, label, confidence}` segments.

## Pipeline

```
video (+ optional --zones config)
  -> geometry branch: YOLO11n -> ByteTrack -> rule engine -> per-class scores
  -> appearance branch: sliding windows -> CLIP zero-shot -> per-class scores
  -> shared: smoothing -> thresholding -> segment merging -> final segments
```

## Dataset format

There is no bundled training dataset — this baseline needs none (see "Why no
training data" below). What does exist:

- `data/samples/<video>` — a video to run the pipeline on.
- `data/ground_truth/<video>.json` — optional, only needed for `evaluate.py`:
  `{"video": ..., "segments": [{"start": 12.5, "end": 25.3, "label": "..."}]}`.
- `configs/zones/<name>.yaml` — optional, per-camera/site pixel-polygon zones for
  the Intrusion and Running classes (see "Zone configs" below).

## Installation

Requires Python 3.12 (the repo was built and tested with Homebrew's
`/opt/homebrew/bin/python3.12`; the bare `python3` on many Macs is an older
system Python — check `python3 --version` first) and `ffmpeg` on PATH
(`brew install ffmpeg`). No GPU required — runs on CPU or Apple Metal (MPS)
automatically; tested on Apple Silicon (M5).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v        # 41 unit tests, no video/network needed
```

YOLO weights (`yolo11n.pt`, ~5MB) and CLIP weights (ViT-B/32, ~600MB) download
automatically on first run — from Ultralytics' GitHub releases and Hugging Face
respectively — and are cached after that (`yolo11n.pt` in the repo root,
CLIP under `~/.cache/huggingface`). No large dataset is downloaded; only pretrained
weights.

## Why no training by default

This baseline uses **zero-shot and rule-based** components by default,
specifically because no training data was provided and no GPU is available
(see the constraints section of `docs/designs/activity-detection-baseline.md`).
Nothing needs to be trained to run `inference.py`. The closest equivalent to
training for the geometry-branch rules is validation-based **config
selection** (window size, stride, confidence threshold, `merge_gap_s`) — see
`scripts/run_experiments.py` and `report.md` §7-8 for the one real, measured
tuning pass done this way.

An **optional** training path exists for the 4 appearance classes — see below.

## Training a few-shot CLIP probe (optional, only if you have labeled clips)

The appearance branch (Fire/Smoke, Fighting, Vandalism, Theft) defaults to
CLIP zero-shot, which needs no labeled data. If you have labeled examples for
a class — **video clips or still images, freely mixed** — you can train a
small logistic-regression probe on top of frozen CLIP embeddings instead —
cheap enough to train on CPU in seconds.

Images work as well as video clips, and are often *better* few-shot
examples: a video window is scored by averaging independent per-frame CLIP
embeddings, so a still image and a video frame are the same kind of input to
the model — but a handful of frames sampled from one clip are highly
correlated (same scene, ~1s apart), while N distinct photos are N genuinely
independent examples. Images are also far easier to source in bulk with
clear licensing.

```bash
mkdir -p data/few_shot/fire_smoke/{positive,negative}
# put labeled clips (.mp4/.avi/.webm/.mov/.mkv) and/or images
# (.jpg/.jpeg/.png/.webp/.bmp) in each folder, freely mixed, then:
python train_clip_probe.py
```

Any class with **at least 4 positive and 4 negative examples** gets a trained
probe saved to `checkpoints/clip_probes/<class>.pt`; classes with fewer are
skipped with a warning rather than trained on data too thin to mean anything
(`MIN_EXAMPLES_PER_CLASS` in `src/classifiers/few_shot_probe.py`).
`inference.py` picks up any trained probe automatically — no config change
needed — and a class with no probe keeps using zero-shot.

**This project has no labeled data of its own**, so nothing is trained by
default. `python train_clip_probe.py --demo` builds a throwaway dataset from
`data/samples/` (1-2 clips per class) to prove the training code runs
end-to-end — it prints a loud warning and, by design, **refuses to actually
train** on that few examples unless you also pass
`--force-min-examples 1` to override the safety guard. Verified directly:
forcing it through anyway and running inference with the resulting probes
made `fire_smoke_01.mp4` go from a correct 0.99 zero-shot detection to *zero*
detections — real, measured proof the 4-example minimum isn't just caution
for its own sake. Those throwaway probes were deleted after verifying this.

**Real probes were then trained on real, sourced, visually-verified data**
(4-10 examples/class for Fire/Smoke, Fighting, Vandalism — Theft still has
none) and tested against every sample video in this project. **Result: worse
across the board** — every previously-correct zero-shot detection
disappeared, and an existing Theft false positive got more confident (a
side effect of removing classes from the shared softmax, not overfitting on
Theft itself, which has no probe). Reverted; the trained checkpoints are
preserved at `checkpoints/_evaluated_underperformed/` as a real result, not
shipped. `checkpoints/clip_probes/` is empty in this repo — `inference.py`
uses zero-shot for all 4 appearance classes. Full numbers and root-cause
analysis: `data/few_shot/PROVENANCE.md`.

**The shared-softmax coupling bug named above was then fixed and measured,
too** — scoring every zero-shot class independently against `background`
instead of sharing one softmax. It does remove the coupling and slightly
raises confidence on real detections, but on `vtest.avi` (the only video
with real ground truth) it let 3 new Vandalism false positives through that
the original competitive softmax had suppressed, dropping F1@tIoU0.5 from
0.909 to 0.714 — no threshold recovered the baseline. **Also reverted**, for
the same reason as the probes: worse than what's shipped, checked against
real numbers rather than assumed. `src/classifiers/clip_zeroshot.py` still
uses the shared softmax; full writeup in its module docstring and
`data/few_shot/PROVENANCE.md`.

## Web UI (upload a video, see results in the browser)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload any video (or pick a bundled sample),
choose a zone config and threshold in the sidebar, click **Run detection**, and
see the segment table, a timeline plot, and an optional geometry-branch debug
overlay (bbox/track/zone) — all backed by the exact same `src/pipeline/run.py`
code `inference.py` uses, no separate logic. YOLO/CLIP are loaded once and
cached across runs (`st.cache_resource`), so trying a second video is fast.
Validated headlessly with Streamlit's `AppTest` framework (no exceptions,
output matches the CLI byte-for-byte on the same video).

## Running inference on a video (including a completely new one)

```bash
python inference.py --video path/to/any_video.mp4
```

This is the exact command to run on a new sample at interview time — **no code
changes required**. It:

1. Loads YOLO11n + CLIP (downloading weights on first run only).
2. Reads the video, runs both branches, applies temporal post-processing.
3. Prints a readable summary of detected activities to the terminal.
4. Writes the full predictions to `outputs/predictions/<video_stem>.json` (or
   `--output <path>` to choose your own).

Example output:

```
Detected activities:

00:24.6 - 00:31.0
Intrusion / Trespassing
Confidence: 1.00
```

Optional flags: `--output`, `--threshold` (override the confidence threshold),
`--window-size` / `--stride` (appearance-branch sliding window, seconds),
`--sample-fps` (geometry-branch detection sampling rate), `--device`
(`auto`/`cpu`/`mps`/`cuda`), `--zones <name>` (see below), `--config` (alternate
`classes.yaml`), `--weights` (alternate YOLO weights).

### Zone configs — only needed for Intrusion / Running

Those two classes are defined relative to a restricted/no-running region, which
can't be inferred from pixels alone — it's a site policy, not a visual property.
Without `--zones`, `inference.py` runs normally and reports those two classes as
not applicable for that run (never guesses at a zone). To add one for a new
camera/site (reused across every video from that camera, authored once):

```bash
ffmpeg -i your_video.mp4 -vf "select=eq(n\,0)" -vframes 1 /tmp/frame0.jpg   # grab a frame
# open the frame, eyeball pixel coordinates of the restricted area's corners
cp configs/zones/vtest.yaml configs/zones/your_camera.yaml   # edit polygons + frame_size
python inference.py --video your_video.mp4 --zones your_camera
```

`configs/zones/vtest.yaml` is a worked example with comments on how it was
authored.

## Evaluating against ground truth

```bash
python -m scripts.evaluate --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json
```

Prints overall precision/recall/F1 at tIoU thresholds 0.3/0.5/0.7, per-class
metrics, and a confusion matrix; writes the full result to
`outputs/metrics/<predictions_stem>_metrics.json`. Matching strategy: greedy
highest-tIoU-first, one-to-one (a ground-truth segment can only be matched once);
a prediction is a true positive only if its label matches **and** tIoU is above
the threshold. See `report.md` §5 for why tIoU/precision/recall/F1 are used
instead of frame-level accuracy.

## Visualization

```bash
python -m src.eval.visualize --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json \
    --video data/samples/vtest.avi --zones vtest
```

Saves a ground-truth-vs-prediction timeline plot to
`outputs/visualizations/<video>_timeline.png`, plus a handful of bbox/track/zone
overlay frames (geometry-branch debugging) to
`outputs/visualizations/<video>_frames/`. Add `--skip-overlay` to skip the
(slower) overlay frames.

## Sample videos (for demo runs)

`data/samples/` has 8 clips: Fire/Smoke (1), Fighting (2), Falling (1),
Running (2, each zone-configured), Vandalism (1) — sourced from Wikimedia
Commons / Internet Archive — plus `vtest.avi` (Crowd Formation, Intrusion,
the one video with real ground truth). Real sample-run predictions live in
`outputs/predictions/`, timelines in `outputs/visualizations/`. **These are
demo material, not scored evaluation** — no ground truth exists for any of
them except `vtest.avi`. Full source/license per clip, and an honest read of
what each sample run actually showed — correct detections, real false
positives, and two independently-diagnosed reasons Running doesn't fire on
either candidate clip — is in `data/samples/PROVENANCE.md`. Theft/Chain-
Snatching and Abandoned Object have no sample clip after two rounds of
searching — explained there, not silently skipped. Re-run any of them
yourself:

```bash
python inference.py --video data/samples/fire_smoke_01.mp4
```

## Reproducing results

```bash
python inference.py --video data/samples/vtest.avi --zones vtest --output outputs/predictions/vtest.json
python -m scripts.evaluate --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json
python -m src.eval.visualize --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json --video data/samples/vtest.avi --zones vtest
python -m scripts.run_experiments   # window/stride/threshold sensitivity sweep — see report.md §7 for the honest caveat on what this sweep can and can't claim
```

A fixed random seed (`src/utils.set_seed`, 42) is set at the start of every entry
point; the pipeline is otherwise deterministic (no sampling in inference), so
re-running produces identical output.

## Output format

`inference.py` and the ground-truth JSON share one schema:

```json
{"video": "path/to/video.mp4", "segments": [{"start": 12.5, "end": 25.3, "label": "Fighting / Physical Assault", "confidence": 0.87}]}
```

`start`/`end` are seconds from video start; `confidence` is in `[0,1]`; segments
of *different* classes may overlap in time (e.g. Crowd Formation concurrent with
Running); segments of the *same* class never overlap after merging.

## Limitations

Stated in full, with evidence, in `report.md` §10 — summary:

- **Only one real annotated video** (`data/samples/vtest.avi`, OpenCV's
  public-domain pedestrian courtyard clip) exists in this project. It gives
  genuine quantitative coverage for 2 of the 10 classes (Crowd Formation,
  Intrusion); the other 8 have zero ground-truth instances here and are not
  quantitatively evaluated — a real, disclosed gap, not an oversight.
- Tracker ID churn (ByteTrack on subsampled frames) fragmented one continuous
  event into two short predictions — diagnosed and fixed by raising
  `merge_gap_s` (F1 @ tIoU≥0.5: 0.615 → 0.909, see report.md §6/§8). The
  underlying tracker fragility isn't eliminated, only bridged for gaps under
  ~3s — see `outputs/metrics/error_analysis.md` for the full mechanism and
  the sweep that shows the fix's limits.
- CLIP zero-shot showed a real, measured false-positive tendency on Vandalism
  before background-class + threshold fixes; residual risk on visually-subtle
  classes (Theft, Vandalism) is still named as unresolved in the report.
- **Generalization smoke test on two fresh, unrelated videos** (different
  resolution/fps/content, no ground truth — not part of the scored evaluation)
  found the pipeline runs end-to-end with no crashes and no code changes on
  both, but CLIP's background rejection is imperfect on genuinely
  out-of-distribution footage: `tree.avi` (no people) still confidently
  mislabeled Vandalism even after broadening background prompts. Named as an
  open, unresolved risk in `report.md` §10, not silently patched away.
- `scripts/run_experiments.py`'s sweep is a single-video sensitivity analysis,
  not a validation/test split — disclosed explicitly rather than presented as
  more rigorous than it is.

## Future improvements

See `report.md` §10 for the full list with reasoning; highlights: a second
annotated video (or several) to get real coverage on the remaining 8 classes and
enable an honest validation/test split (the `merge_gap_s` fix above was chosen
via a single-video sensitivity sweep, not held-out validation, for exactly this
reason); track re-identification, so dwell/velocity counters survive a tracker
ID switch without needing a merge-gap workaround at all; a trained few-shot
linear head on CLIP embeddings once labeled clips exist for the appearance
classes (explicitly scoped out of this build in the design doc, not attempted
here).
