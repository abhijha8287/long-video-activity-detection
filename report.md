# Activity Detection & Classification in Long Videos Report

## Why I built it this way

This was a take home assignment: given a long video, find the time windows
where one of 10 activities happens and label each one with a confidence
score.

I went in believing a working, explainable baseline beats an ambitious
system.

## The problem

Input is one video that may contain multiple activities, idle stretches,
and events of very different lengths. Output is a list of
`{start, end, label, confidence}` segments for these 10 classes:
Fire/Smoke, Fighting/Assault, Falling, Intrusion/Trespassing, Loitering,
Vandalism, Theft/Chain Snatching, Crowd Formation, Running (in a
restricted zone), Abandoned Object. It has to work on a video it's never
seen before, without touching any code `python inference.py --video
<new>.mp4`.


## How the system works

Six of the ten classes are really about *where someone is and for how
long*, not what they look like Loitering, Intrusion, Running, Crowd
Formation, Abandoned Object, Falling. No pretrained video model has ever
seen the concept of "this is a restricted zone," because that's a property
of the site, not the pixels. The other four Fire/Smoke, Fighting,
Vandalism, Theft are genuinely about appearance and are a much better
fit for a pretrained vision-language model.

Given that split, and given I had no GPU and no training data, I built two
branches instead of forcing one model to do both jobs:

**Geometry branch.** YOLO11n detects people and objects, ByteTrack keeps
track of who's who across frames, and a small deterministic rule engine
watches that track history zone occupancy, dwell time, velocity, local
density, how much a bounding box's shape changes, whether an object has
been left without anyone nearby. Every rule is written out in
`configs/classes.yaml`, not hardcoded, and needs no training data at all.

**Appearance branch.** CLIP (ViT-B/32) scores each sliding window of the
video against a handful of text prompts per class. I added an internal
"background" class that competes in the same comparison, because without
it the model was forced to pick one of the 4 real classes as a "winner"
even on a completely ordinary scene I only found this because I ran it
on real footage and watched it happen (more on that below).

Both branches output a plain 0-to-1 score, and since they never score the
same class there's no need to calibrate one against the other. A shared
post-processing step (smoothing, thresholding, merging nearby detections,
dropping anything too short) turns those scores into the final segments.
Different classes are allowed to overlap in time Crowd Formation and
Running can genuinely happen together but the same class never produces
overlapping segments after merging.

For timing: the geometry rules just run continuously on the track history.
The appearance branch uses 3-second windows with 1.5-second overlap a
starting guess, since there was no dataset to derive a duration
distribution from, and I checked later whether that
guess actually mattered.

I deliberately did not train anything for the first version. The
appearance branch runs zero shot; adding a trained head was something I
scoped out on purpose rather than half building it, because there was no
labeled data to train on yet.

## Setup and how I measured things

Everything ran on an Apple M5 (CPU/Metal, no CUDA GPU) — Python 3.12,
PyTorch 2.14, Ultralytics 8.4, open_clip. On the one video with ground
truth, the geometry branch takes about 10.5s and the appearance branch
about 8.5s, roughly 19s total for a 79.5s clip — so about a quarter of
real-time on this hardware. I fixed the random seed (42) everywhere, and
there are 53 unit tests covering the metrics, geometry rules, and segment
building that don't need any video or network access to run.

For the actual scoring metric I used temporal IoU precision, recall, and
F1 at IoU thresholds of 0.3, 0.5, and 0.7, matched greedily one-to-one so
a ground truth event can't be double counted. A prediction only counts as
correct if both the label and the timing overlap enough. I specifically
didn't use plain accuracy as the headline number most of a long video is
background, so a model that predicts "nothing happening" almost
everywhere would score very high on accuracy while catching literally
nothing real. Precision/recall/F1 actually tells you whether real events
were found and how many predictions were noise.

## Results

With the final settings (3s/1.5s window, 8fps sampling, 0.6 confidence
threshold, 2s gap merging explained below):

| tIoU | Precision | Recall | F1 | Avg tIoU on matches |
|---|---|---|---|---|
| 0.3 | 0.833 | 1.000 | 0.909 | 0.764 |
| 0.5 | 0.833 | 1.000 | 0.909 | 0.764 |
| 0.7 | 0.500 | 0.600 | 0.545 | 0.894 |

Per class at tIoU 0.5, for the two classes I actually have ground truth
for: Crowd Formation scores 0.67/1.00/0.80, Intrusion scores a clean
1.00/1.00/1.00. All 5 ground-truth events were found, with one leftover
false positive. There was zero label confusion every match got the
right class name, nothing was mislabeled as something else.

The gap merging setting made a real difference, not a cosmetic one:

| Setting | Precision | Recall | F1 |
|---|---|---|---|
| merge_gap = 1.0s (starting point) | 0.500 | 0.800 | 0.615 |
| merge_gap = 2.0s (what's shipped) | 0.833 | 1.000 | 0.909 |

Both rows are from actually running the pipeline, not projected the
first is reproducible by setting `merge_gap_s` back to 1.0 in
`configs/classes.yaml` and re-running the evaluator.

## What I actually tuned, and what I didn't

I swept the appearance branch window size, stride, and confidence
threshold using `scripts/run_experiments.py`. Worth saying up front: with
only one annotated video, this is a sensitivity check, not a proper
validation split I don't want to oversell it.

| Config | Window/Stride | F1 at threshold 0.4 / 0.5 / 0.6 |
|---|---|---|
| baseline | 3.0 / 1.5 | .53 / .50 / .62 |
| no overlap | 3.0 / 3.0 | .57 / .57 / .62 |
| wider window | 5.0 / 2.5 | .53 / .57 / .62 |

Recall stayed flat at 0.8 across every config, because it's driven
entirely by the geometry branch the appearance branch had no ground
truth to recall in this video regardless of settings. Threshold 0.6 won
in every configuration by cutting appearance branch false positives
without losing a single geometry-branch true positive, so that's what
shipped. Window/stride made no measurable difference here, so I kept the
finer default since it should help responsiveness on videos where the
appearance branch actually has something to find.

## Bugs I found and fixed by actually watching the output

I generated the real timeline plot and the bbox/zone overlay frames and
looked at them, which is how I caught two real problems, not by staring
at the code.

**Loitering fired on people just walking through.** The rule only checked
whether someone was inside the zone continuously for 15+ seconds it
never checked whether they were actually staying still. I fixed it by
requiring the track to stay within a small radius of one anchor point,
not just inside the zone polygon. After the fix, Loitering correctly
predicts nothing on this video.

**CLIP called almost the whole video "Vandalism."** On the first real run,
zero-shot CLIP flagged Vandalism at 0.77 confidence for nearly the entire
clip. The problem was structural: with a forced 4-way choice among the
real classes, the model has no way to say "none of these apply." Adding
the background class cut it down to three much shorter false positives
around 0.53-0.59, and the 0.6 threshold cleaned up what was left.

The bigger issue, though, was fragmentation. One real Intrusion event
(14.0s-21.0s in the ground truth) was originally detected as two separate
short fragments 1.9 seconds apart, and neither fragment alone cleared the
IoU threshold so it counted as a miss even though the pipeline basically
saw it. The cause was that the dwell time counter reset itself on any gap
in zone membership, including a brief tracker ID switch that had nothing
to do with the person actually leaving. Raising `merge_gap_s` from 1s to
2s bridges that 1.9s gap without accidentally merging two genuinely
separate Intrusion events later in the video that are 3.3s apart. I
checked this by sweeping the gap from 1s up to 10s: F1 climbs to 0.909
around 2-3s, then falls off a cliff to 0.40 by 4s once distinct events
start getting merged into one. So 2s isn't an arbitrary number it's the
largest gap that fixes the real bug without creating a new one.

One Crowd Formation false positive I couldn't fully resolve either way
it's genuinely ambiguous whether it's a real error or my ground truth
being too conservative, and I'd rather leave that open than force a
conclusion the evidence doesn't support. And to repeat the obvious: 8 of
the 10 classes never occur in this video, so none of this analysis touches
them.

## Limitations, stated plainly

- Real evaluation exists for only 2 of 10 classes, on one video. The other
  8 are demoed on sample clips but not scored against ground truth.
- The window/threshold sweep above is a sensitivity check on one video,
  not a held out validation set I don't have the data to do better yet.
- Tracker fragility is bridged, not solved. The merge gap fix covers gaps
  under about 3 seconds; a longer dropout would still fragment an event.
- Abandoned Object's notion of "owner" is just "any person nearby," not a
  re-identified specific individual I never got a real abandoned-object
  clip to test this against.
- Zone polygons were drawn by eye off a single frame, not measured
  precisely.
- CLIP's background rejection isn't perfect on footage unlike anything in
  its prompt vocabulary. I ran two unrelated test videos with no ground
  truth as a rough generalization check: an animated clip correctly
  produced zero detections, but a plain video of trees with no people in
  it still got flagged as Vandalism at 0.90 confidence broadening the
  background prompts brought that down to 0.75, still above threshold.
  This is a real open risk I haven't solved, not something I've quietly
  worked around.

What I'd do next, in order: get a second annotated video so I have real
coverage on the other 8 classes and can do an actual validation split
instead of a one-video sensitivity check; add track re-identification so
dwell/velocity counters survive an ID switch cleanly instead of leaning on
the merge gap fix; get a second person to independently label the
ambiguous Crowd Formation case.


**Attempt one: train probes on that data.** Train accuracy looked
reasonable (0.76-0.90), but tested against the same sample videos I'd
already been using, it made things worse across the board every
previously correct zero shot detection disappeared, and an existing false
positive on Theft actually got more confident even though Theft has no
trained probe at all. That second part surprised me: pulling three classes
out of the shared softmax changed the calibration of the untouched fourth
class, because softmax scores are relative to each other, not independent.
19-30 training examples per class just isn't enough to beat a model
pretrained on 400 million image-text pairs. I reverted this — the
checkpoints are kept in `checkpoints/_evaluated_underperformed/` as a
record, but they're not what's running.

## Running it

Same pipeline code runs whether you use the CLI (`inference.py`) or the
Streamlit app (`app.py`) they give byte identical output on
the same video. The app is deployed on Streamlit's free community tier,
which means a single shared, throttled CPU and no GPU, so it's noticeably
slower there than the numbers above and can take 30-60 seconds to wake up
if it's been idle.

