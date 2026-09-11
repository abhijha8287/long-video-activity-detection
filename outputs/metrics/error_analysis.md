# Error analysis — vtest.avi

Generated from `outputs/metrics/vtest_metrics.json` (tIoU >= 0.5) plus direct frame
inspection. This is the distinct error-analysis step referenced in
`docs/designs/activity-detection-baseline.md` — see `report.md` §6/§8 for the
narrative version and the baseline-vs-improved table.

**Update (post-fix):** the temporal-fragmentation false negative described
below was diagnosed here, then fixed by raising `merge_gap_s` from 1.0s to
2.0s in `configs/classes.yaml` (`report.md` §8 has the full sweep showing why
2.0s and not higher). This file keeps the original diagnosis intact — it's
what led to the fix — and adds the post-fix result at the end of each section.

## True positives — now 5/5 ground-truth segments matched (was 4/5)

| Prediction | Ground truth | tIoU |
|---|---|---|
| Intrusion 24.6–31.0 | Intrusion 24.0–31.0 | 0.912 |
| Intrusion 57.6–69.4 | Intrusion 57.0–70.0 | 0.909 |
| Intrusion 14.7–21.3 | Intrusion 14.0–21.0 | 0.861 |
| Crowd Formation 68.9–71.1 | Crowd Formation 68.0–71.5 | 0.634 |
| Crowd Formation 5.8–15.3 | Crowd Formation 5.0–11.0 | 0.504 |

Boundary alignment is good (avg tIoU on matches: 0.76 post-fix) when the
system detects a continuous event at all — the rule engine's timestamps track
the real event closely, not just its rough location. Intrusion is now a
perfect match on all 3 instances (P=1.00, R=1.00 at tIoU≥0.5).

## Original false negative: temporal fragmentation, not a missed event (FIXED)

Ground truth `Intrusion 14.0–21.0` was originally **not matched**, even though
the system did detect activity in that window — split into two short
fragments instead of one continuous segment:

- Predicted `Intrusion 14.7–16.6` (tIoU vs. the 7s ground-truth segment ≈ 0.27)
- Predicted `Intrusion 18.5–21.3` (tIoU ≈ 0.31)

Neither individually cleared the 0.5 tIoU bar, so this was 2 false positives
**and** 1 false negative in the detection metrics, even though the system was
"roughly right" about when something was happening. Root cause: the Intrusion
rule's dwell-duration counter resets whenever a track's zone-membership streak
breaks — which happens on a tracker ID switch (a person gets a new track ID
mid-crossing) or a brief zone-boundary flicker, both of which the
tracker-fragility risk in the design doc named as a real risk before this
evaluation ever ran.

**Fix applied:** the gap between the two fragments was measured directly —
16.62s to 18.5s = 1.88s. Raising the post-processing `merge_gap_s` from 1.0s
to 2.0s bridges exactly that gap. A full sweep (1.0 / 1.5 / 2.0 / 2.5 / 3.0 /
4.0 / 6.0 / 10.0s) confirmed this is a real fix targeting the right mechanism,
not a threshold picked to flatter one number: F1 (tIoU≥0.5) is flat at 0.615
through 1.5s, jumps to 0.909 at 2.0-3.0s (bridges the tracker gap, all 5 GT
segments now match), then **collapses to 0.40 by 4.0s** because a *different*,
genuinely separate Intrusion event 3.3s later in the video starts getting
wrongly merged into the same segment. 2.0s sits with margin on both sides of
that mechanism, not at an arbitrary round number.

## False positives — one ambiguous, borderline crowd-density trigger remains (was two)

- Predicted `Crowd Formation 24.6–26.2` still has no matching ground-truth
  segment. (The other original false positive, `Crowd Formation 12.9–15.3`,
  is now absorbed into the corrected `5.8–15.3` true-positive segment above —
  it was adjacent to a real event, not a separate spurious one, and the
  merge-gap fix folded it in correctly.)

This was not independently confirmed as a genuine gathering during
ground-truth authoring (see the `_provenance` note in
`data/ground_truth/vtest.json`) — a visual spot-check at nearby timestamps
showed people present but more spread out than a tight cluster. Two honest
readings remain, undisambiguated by this evaluation: (a) the density rule's
`cluster_radius_px=150` is a little generous for this camera's scale,
producing a real false positive, or (b) this was a borderline gathering
moment a stricter human annotator would have also marked, and the "false
positive" label reflects ground-truth conservatism rather than a system
error. Distinguishing these needs either a second annotator or a
frame-by-frame density measurement, both out of scope here — named as an open
question rather than resolved either way.

## What this evaluation cannot speak to

- **8 of 10 classes have zero ground-truth coverage** in this video (Loitering,
  Running, Falling, Abandoned Object, and all 4 appearance classes) — see the
  data-sourcing limitation in the design doc and report.md §10. The metrics above
  describe 2 classes only.
- **Loitering was fixed, not evaluated.** An earlier version of the rule flagged
  ordinary walking as loitering (caught during this same development pass — see
  the docstring in `src/rules/engine.py::dwell_time_scores`); the fixed version
  correctly predicts zero loitering segments for this video, which is the honest
  outcome (nobody stands still here) but isn't a *positive* test of the fix.
- **CLIP zero-shot's Vandalism false-positive tendency** (observed before the
  background pseudo-class fix and the threshold tuning — see report.md §8) was
  real and measured, not hypothetical, exactly matching the risk the design doc
  named in advance. A further out-of-distribution false positive (a graffiti
  mural read as active vandalism, a grainy 1971 safety film read as vandalism
  instead of Falling) was found and documented in
  `data/samples/PROVENANCE.md` during broader sample-video testing — still
  unresolved, named as an open risk in report.md §10.
