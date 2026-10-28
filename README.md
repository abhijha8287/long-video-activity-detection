# Activity Detection in Long Videos

Finds the time windows in a long, untrimmed video where an activity happens, and labels each with one of 10 classes and a confidence score.

## Why this exists
Built as a take-home assignment. The full design reasoning, measured results and failure analysis are in [`report.md`](report.md).

## What it does
- Detects 10 classes: Fire/Smoke, Fighting, Falling, Intrusion, Loitering, Vandalism, Theft, Crowd Formation, Running in a restricted zone, Abandoned Object
- **Geometry branch**: YOLO11n detection and ByteTrack tracking feed a rule engine (zone occupancy, dwell time, speed, density, shape change, owner separation); no training data needed
- **Appearance branch**: CLIP (ViT-B/32) zero-shot scoring against a "background" class, so an ordinary scene does not force a wrong label
- Smooths, thresholds and merges both branches' scores into `{start, end, label, confidence}` segments
- Runs from the command line or a Streamlit app, on CPU, Apple Metal or CUDA

## Project structure
```
long-video-activity-detection/
├── src/
│   ├── detection/    YOLO11n detection and tracking
│   ├── rules/        Rule engine and geometry
│   ├── classifiers/  CLIP zero-shot and few-shot probe
│   ├── pipeline/     Run loop and segment building
│   └── eval/         Metrics and visualisation
├── configs/          Class list and zone definitions
├── data/             Samples, ground truth, few-shot images
├── scripts/          Evaluation and experiment runners
├── tests/            Unit tests (no video needed)
├── outputs/          Predictions, metrics, visualisations
├── docs/designs/     Baseline design document
├── inference.py      Command-line entry point
├── app.py            Streamlit app
├── report.md         Design, results and limitations
├── packages.txt      System packages (ffmpeg)
└── requirements.txt  Python dependencies
```

## Getting started
Requirements: Python 3.12, `ffmpeg` on `PATH`. No GPU needed.
```bash
git clone https://github.com/abhijha8287/long-video-activity-detection.git
cd long-video-activity-detection
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py --video data/samples/fire_smoke_01.mp4
streamlit run app.py                              # http://localhost:8501
```
Useful `inference.py` flags: `--output`, `--threshold`, `--zones <name>` (needed for Intrusion and Running; see `configs/zones/vtest.yaml`), `--device auto|cpu|mps|cuda`. Run `python inference.py --help` for all flags.

Evaluate and visualise against ground truth:
```bash
python -m scripts.evaluate --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json
python -m src.eval.visualize --pred outputs/predictions/vtest.json --gt data/ground_truth/vtest.json --video data/samples/vtest.avi --zones vtest
```

## Tests
```bash
python -m pytest tests/ -v
```

## Output format
```json
{"video": "path/to/video.mp4", "segments": [{"start": 12.5, "end": 25.3, "label": "Fighting / Physical Assault", "confidence": 0.87}]}
```

## Tech stack
Python, PyTorch, Ultralytics YOLO11n, ByteTrack, OpenAI CLIP, OpenCV, ffmpeg, Streamlit, pytest

## Notes
Live demo: https://long-video-activity-detection-by-abhishek.streamlit.app (free CPU tier, so the first load and inference are slow). Only one real annotated video (`vtest.avi`) exists, covering 2 of the 10 classes; the rest are demonstrated, not scored. See "Limitations, stated plainly" in `report.md`.
