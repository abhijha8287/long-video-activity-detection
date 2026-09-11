# Interview notes

Concise answers describing the actual implementation — not aspirational
architecture. Cross-references point at the exact file/section to go deeper.

**1. Why did you choose this model?**
Not one model — a hybrid. 6 of the 10 classes (Loitering, Intrusion, Running,
Crowd Formation, Abandoned Object, Falling) are defined by position/motion
relative to a scene, which no pretrained video classifier has a concept of
("restricted zone" isn't a Kinetics label). So: YOLO11n + ByteTrack + a
deterministic rule engine for those, CLIP ViT-B/32 zero-shot for the 4
appearance-defined classes (Fire/Smoke, Fighting, Vandalism, Theft). No
training data existed and no GPU was available, which ruled out fine-tuning
anything from scratch. `report.md`, "How the system works."

**2. Why not use a simple image classifier?**
Single-frame classification can't see motion or duration — it can't tell
"someone standing" from "someone loitering for 15s" or "someone walking" from
"someone running." The geometry branch's rules operate on *track history*
(position over time), not single frames, specifically because several classes
require that.

**3. How do you handle long videos?**
Two passes over the video, not one: the geometry branch samples frames at
8fps and runs detection+tracking continuously; the appearance branch runs
CLIP over sliding windows (3.0s window, 1.5s stride). Both produce per-
timestamp/per-window scores that get merged onto one timeline and
post-processed together. `src/pipeline/run.py::compute_class_series`.

**4. How do you detect start/end times?**
Per-class score series → median smoothing → threshold → group consecutive
active samples into a raw segment → merge segments of the same class
separated by a small gap → drop segments shorter than a minimum duration.
`src/pipeline/segment_builder.py`.

**5. Why did you choose the window size?**
3.0s/1.5s (50% overlap) was a starting hypothesis — no activity-duration
distribution existed to derive it empirically (no dataset was provided). A
measured sweep (`scripts/run_experiments.py`) found window/stride had no
effect on the one video available; the finer default was kept for
responsiveness on videos where the appearance branch actually contributes.
`report.md`, "What I actually tuned, and what I didn't."

**6. How do you handle background?**
Two mechanisms. Geometry branch: rules simply don't fire absent a triggering
condition — no explicit "background" state needed. Appearance branch: an
internal "background" pseudo-class competes in CLIP's softmax alongside the
4 real classes (added after a real bug — see Q19) so the softmax can express
"none of these" instead of always picking a winner. Neither ever appears in
output — filtered out by construction in `build_segments_from_series`.

**7. How do you prevent duplicate segments?**
Median smoothing before thresholding avoids single-frame flicker becoming its
own segment. Consecutive active samples become one raw segment (not one per
sample). Segments of the same class within `merge_gap_s` of each other get
merged into one. Cross-class overlap is allowed by design (Crowd Formation
and Running can genuinely co-occur) — only same-class duplication is
prevented.

**8. How do you evaluate temporal localization?**
Temporal IoU with greedy highest-IoU-first, one-to-one matching (a ground
truth segment can't be matched twice). A prediction is a true positive only
if both the label matches and tIoU clears the threshold. Swept at 0.3/0.5/0.7.
`src/eval/metrics.py`; `report.md`, "Setup and how I measured things."

**9. What is temporal IoU?**
Intersection over union of two time intervals: overlap duration divided by
combined duration. `intersection / union` on `[start, end]` ranges.

**10. Why is accuracy not enough?**
Long background periods dominate most video. A system predicting "nothing"
almost everywhere scores high frame-level accuracy while catching zero real
activity. Precision/recall/F1 on segments actually reflects whether real
events were found and how many predictions were spurious. `report.md`, "Setup and how I measured things."

**11. How did you avoid data leakage?**
There's no train/val/test split to leak across — nothing is trained. The one
place leakage-like risk exists is the `merge_gap_s` fix (Q17): it was chosen
by sweeping against the only annotated video available, which is a disclosed
single-video sensitivity analysis, not a validation-set-verified choice. This
is named explicitly as a limitation, not hidden. `report.md`, "What I actually tuned, and what I didn't" and "Limitations, stated plainly."

**12. How do you handle class imbalance?**
N/A in the conventional sense — there's no training set to be imbalanced.
The real analogue: only 2 of 10 classes have any ground-truth coverage at
all, which is disclosed everywhere (README, report, PROVENANCE.md) rather
than presented as balanced coverage.

**13. Why use tracking?**
Detection alone gives per-frame boxes with no identity — dwell time,
velocity, and "has this object been separated from a person for N seconds"
all require knowing that box *i* in frame *t* is the same object as box *j*
in frame *t+1*. ByteTrack (Ultralytics' built-in tracker) provides that.

**14. How does loitering detection work?**
A person's position is tracked; if they stay within a stillness radius (60px)
of an anchor point for ≥15s while inside a "loitering" zone, the dwell score
ramps up via a sigmoid centered on the threshold. An earlier version checked
zone *presence* only, not stillness, and incorrectly flagged ordinary walking
— see Q19. `src/rules/engine.py::dwell_time_scores`.

**15. How does abandoned-object detection work?**
An object (backpack/suitcase/handbag) becomes a candidate once stationary for
≥10s; it's flagged once no person has been within a proximity radius of it
for ≥10s. Simplification, stated explicitly in code and report: "owner" is
approximated as "any nearby person," not a re-identified specific individual
— re-identification across occlusion was out of scope. Untested on real
footage (no abandoned-object events in the one available video).
`src/rules/engine.py::abandoned_object_scores`.

**16. How would you improve the system?**
In order: (1) a second annotated video, to get real coverage on the 8
untested classes and enable a genuine validation/test split instead of the
single-video sensitivity analysis used for the `merge_gap_s` fix; (2) track
re-identification, so dwell/velocity counters survive a tracker ID switch
without leaning on a merge-gap workaround; (3) a few-shot linear probe on
CLIP embeddings once labeled appearance-class clips exist; (4) a second
annotator to resolve the one remaining ambiguous Crowd Formation false
positive. `report.md`, "Limitations, stated plainly."

**17. What are the biggest limitations?**
Only 2 of 10 classes have real quantitative evaluation (one video, honestly
disclosed). CLIP's background rejection is imperfect on genuinely
out-of-distribution footage — a graffiti mural was misread as active
Vandalism, and an old grainy safety film was misread as Vandalism instead of
being correctly read as Falling (`data/samples/PROVENANCE.md`). Tracker
fragility is bridged, not eliminated, by the `merge_gap_s` fix — it fixes
gaps up to ~3s, not arbitrary ones.

**18. How does the system behave on an unseen video?**
Tested directly, not assumed: ran on 2 structurally unrelated videos never
seen during development (different resolution/fps/content) plus 5 sourced
sample clips, all with no code changes. It ran end-to-end on all of them with
no crashes; results ranged from correct (Fire/Smoke 0.99) to informative
failures (Falling missed on a grainy old film, Vandalism false-positived on
a legal mural) — all documented rather than cherry-picked.

**19. What happens if the model sees an activity outside the ten classes?**
Geometry branch: nothing fires unless a specific rule's condition is met —
an out-of-scope activity with no zone/dwell/density/shape signature produces
no detection. Appearance branch: this was a real, measured bug before the
fix — a forced 4-way softmax with no "background" option always picks a
"winner" among the 4 real classes even for content matching none of them
(confirmed: Vandalism spuriously flagged for nearly an entire unrelated
pedestrian video at 0.77 confidence). Fixed by adding an internal background
pseudo-class that competes in the same softmax. Residual risk remains on
content unlike anything in the prompt vocabulary (Q17).

**20. What would you change with more training data?**
This was actually tried, not just planned. The training path exists
(`train_clip_probe.py`, `src/classifiers/few_shot_probe.py`) — a logistic-
regression probe on frozen CLIP embeddings, auto-loaded by `inference.py` for
any class with a trained checkpoint, falling back to zero-shot otherwise.
Built *after* the original submission, once asked, rather than earlier
without real data to justify it.

Real data was then gathered — CC-licensed video clips and images from
Wikimedia Commons, visually verified one by one (4 downloaded images were
rejected after actually looking at them; their titles were misleading) —
reaching 4-10 examples per class for Fire/Smoke, Fighting, and Vandalism
(Theft still has none — genuinely couldn't find appropriate source material
after two separate search rounds). Trained real probes (train_accuracy
0.76-0.90) and tested them against every sample video already used in this
project.

**Result: it made things worse, measurably, across the board.** Every
previously-correct zero-shot detection (Fire/Smoke 0.99, Fighting 0.97 and
0.63) disappeared — raw probe scores on genuinely-fire footage came out
0.54-0.58, barely above chance and below the 0.6 threshold. Worse: pulling 3
classes out of the shared zero-shot softmax changed the calibration of
*Theft too*, even though Theft has no trained probe — with fewer competing
classes, an existing Theft false positive went from 0.72 to 0.88 confidence.
Root cause: 19-30 training frames is nowhere near enough to beat a model
calibrated on 400M+ image-text pairs, and the softmax-sharing architecture
means partial probe adoption has knock-on effects on untouched classes.

**The trained checkpoints were reverted** (moved to
`checkpoints/_evaluated_underperformed/`, not deleted) rather than shipped,
because they're worse than the baseline. This is the real answer to "what
would more training data do": it would need to be *substantially* more (tens
of examples per class, not single digits) to have a chance, and the softmax-
sharing side effect would need fixing first regardless of data volume. Full
before/after numbers: `data/few_shot/PROVENANCE.md`.

**The softmax-sharing side effect was then actually fixed, measured, and
also reverted.** Rewrote the scorer so every zero-shot class competes
against `background` independently instead of sharing one N-way softmax —
this does remove the coupling (confirmed) and slightly raises confidence on
real detections, but it also removes false-positive suppression that came
free from classes competing for probability mass: on `vtest.avi`, the one
video with real ground truth, F1@tIoU0.5 dropped from 0.909 to 0.714 (3 new
Vandalism false positives). A threshold sweep couldn't recover the
baseline. Reverted for the same reason as the probes: worse than what's
shipped, on the only video with real numbers to check against. The
underlying coupling bug is real but currently dormant (no probes are
deployed, so nothing is currently missing from the shared softmax) — not
worth a measured accuracy cost to fix something with no live effect right
now. Full numbers and reasoning: `data/few_shot/PROVENANCE.md`, and the
`src/classifiers/clip_zeroshot.py` module docstring.

**The throughline for an interview**: two independent, real fix attempts
this session, both measured rigorously against the same ground truth, both
found to net-regress, both reverted rather than shipped on vibes. That's the
actual answer to "why not just add training data" — it was tried, honestly
evaluated, and correctly rejected twice, which is a stronger signal of
engineering judgment than a single clean success would have been.

**21. Why this approach and not [commercial API / pretrained video model /
train-from-scratch] — there's a lot of other options in the market?**

Four real alternatives were available; each was ruled out for a specific,
concrete reason tied to this project's actual constraints (no GPU, no
provided training data, a fixed 10-class taxonomy defined by geometry as
much as appearance, and the need to explain every decision, not just report
a score):

- **Commercial cloud APIs (AWS Rekognition, Google Video Intelligence, Azure
  Video Indexer).** Fastest to a demo, but wrong for this task on two
  counts: (1) none of them have a "restricted zone" or "loitering in this
  specific area" concept — those are defined relative to *this* scene's
  geometry, which only a system with access to zone coordinates can express;
  a generic activity-recognition API returns generic activity labels, not
  "intrusion into zone X." (2) It's a black box for an interview — I can't
  explain *why* Rekognition tagged a clip "suspicious activity," which
  directly conflicts with the goal of being able to justify every result.
  Also a nonstarter without committing to a paid, internet-dependent
  service for what's meant to be a self-contained take-home.

- **Pretrained video-anomaly models trained on UCF-Crime / XD-Violence.**
  Closest in spirit — pretrained, no GPU-heavy fine-tuning needed. Ruled out
  because (a) their label sets don't match this assignment's 10 classes
  (they're binary anomaly/normal or a different taxonomy entirely — Crowd
  Formation and Abandoned Object aren't in either), and (b) using a model
  trained on real crime footage sits on the wrong side of the same ethics
  line I held during data-gathering this session (declined to download real
  police-arrest documentation even under permissive licensing) — a model
  trained on that footage inherits the same problem one step removed, and
  I'd have no way to audit what it actually saw.

- **Train a custom video classifier from scratch (X3D, VideoMAE,
  SlowFast).** The "textbook correct" answer for video classification, and
  the one I'd reach for with a real training set and a GPU. Neither existed
  here — no labeled video data was provided, and fine-tuning any of these
  from scratch is a multi-GPU-day undertaking, not a take-home-scale task.
  Attempting it anyway would have produced an undertrained model with no
  real evaluation possible, worse than being honest about the constraint.

- **Why the hybrid (rules + CLIP zero-shot), then.** Because the 10 classes
  split cleanly into two kinds of signal, and forcing one model to do both
  is worse than two specialized ones: 6 classes (Loitering, Intrusion,
  Running, Crowd Formation, Abandoned Object, Falling) are defined by
  *position and motion over time relative to the scene* — no pretrained
  video classifier has "inside this polygon for N seconds" as a concept,
  because it's not a property of the pixels, it's a property of the scene
  geometry the deployer defines. A deterministic rule engine on tracked
  boxes handles these exactly and explainably (Q14, Q15). The remaining 4
  (Fire/Smoke, Fighting, Vandalism, Theft) are genuinely *appearance*-based
  and have no geometry component — CLIP's zero-shot image-text alignment,
  trained on 400M+ pairs, already has usable priors for "fire," "people
  fighting," "graffiti" without any task-specific training, which matters
  specifically because no training data existed. The few-shot probe path
  (Q20/this session) exists for exactly the case where zero-shot isn't
  enough and labeled data eventually shows up — built and genuinely tested,
  not just designed, and honestly found not to help yet at the data volumes
  gathered so far.

- **The honest summary, one sentence:** commercial APIs and pretrained
  crime-anomaly models don't fit this task's actual class taxonomy or
  transparency requirement, training from scratch doesn't fit its actual
  data/compute constraints, and the hybrid is the one option that both fits
  the constraints as they really were and stays fully explainable end to
  end — which two rigorously-tested, honestly-reverted improvement attempts
  this session (Q20 above) demonstrate wasn't a one-time design decision but
  an ongoing standard applied to every change since.

---

**Running this in the interview:** either `python inference.py --video
<path>` or `streamlit run app.py` (upload a video in the browser at
`localhost:8501`) — same pipeline code (`src/pipeline/run.py`) either way, so
results are identical between the two interfaces. README has full commands
for both.
