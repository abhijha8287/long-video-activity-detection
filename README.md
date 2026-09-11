# Activity Detection & Classification in Long Videos

Detects time intervals in a long video where an activity occurs and
classifies each into one of 10 classes, with a confidence score. Full
design rationale, measured results, and honest failure analysis:
[`report.md`](report.md).

**Live demo:** [long-video-activity-detection-by-abhishek.streamlit.app](https://long-video-activity-detection-by-abhishek.streamlit.app)
 runs on Streamlit Community Cloud's free tier (single shared, throttled
CPU, no GPU), so expect a slow first load, slower inference than local,
and a ~30-60s wake-up if it's been idle. Correctness is unaffected, only
speed run it [locally](#run-locally) for full speed.

## Classes

Fire/Smoke, Fighting, Falling, Intrusion, Loitering, Vandalism, Theft,
Crowd Formation, Running (in a restricted zone), Abandoned Object.

## Approach

6 classes are defined by *position/motion relative to a scene* (no
pretrained model has a concept of "restricted zone"); 4 are
*appearance*-defined. So this is a two-branch hybrid, not a single model:

- **Geometry branch** — YOLO11n + ByteTrack tracking feed a deterministic
  rule engine (zone occupancy, dwell time, velocity, density, shape
  change, owner separation). No training data needed.
- **Appearance branch** — CLIP (ViT-B/32) zero-shot, scored against an
  internal "background" class so an ordinary scene doesn't force a wrong
  "winner."

Both emit `[0,1]` scores; shared post-processing (smoothing, thresholding,
merging) turns them into `{start, end, label, confidence}` segments.

## Run locally

Requires Python 3.12 and `ffmpeg` on `PATH`. No GPU needed (CPU or Apple
Metal).

```bash
git clone https://github.com/abhijha8287/long-video-activity-detection
cd Cerebralzip2
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -v        # 53 tests, no video/network needed

python inference.py --video data/samples/fire_smoke_01.mp4   # CLI
streamlit run app.py                                          # browser UI at localhost:8501
```


Useful flags for `inference.py`: `--output`, `--threshold`, `--zones
<name>` (needed only for Intrusion/Running — see `configs/zones/vtest.yaml`
for a worked example), `--device auto|cpu|mps|cuda`. Full flag list:
`python inference.py --help`.

## Evaluate / visualize

```bash
python -m scripts.evaluate --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json
python -m src.eval.visualize --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json --video data/samples/vtest.avi --zones vtest
```

## Output format

```json
{"video": "path/to/video.mp4", "segments": [{"start": 12.5, "end": 25.3, "label": "Fighting / Physical Assault", "confidence": 0.87}]}
```


## Limitations

Only one real annotated video (`vtest.avi`) exists, covering 2 of 10
classes — the rest are demoed, not scored. Full list with evidence: see
"Limitations, stated plainly" in `report.md`.
