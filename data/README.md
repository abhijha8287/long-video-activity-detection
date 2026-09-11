# data/

No dataset ships with this project — none was provided, and this baseline needs
none (see the "Why no training data" section of the top-level README).

- `samples/vtest.avi` — OpenCV's public-domain `samples/data/vtest.avi`
  (BSD-licensed project sample), sourced during development as the one real
  video with hand-authored ground truth. Not staged for this assignment.
- `ground_truth/vtest.json` — hand-authored ground truth for `vtest.avi`; see
  its `_provenance` field for exactly how it was built, and `report.md`,
  "Data — or the lack of it," for the honest coverage limitations (2 of 10
  classes).

To run inference on your own video, it does not need to live in this
directory — `python inference.py --video <any/path>.mp4` works from anywhere.
