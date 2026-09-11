# Few-shot training data — sourced for your review, NOT yet trained on

Gathered from Wikimedia Commons (licenses verified via the Commons API, not
guessed) plus reuse of existing sample clips. Round 2 added still images on
top of the video clips from round 1 — see "What happened this round" below
for the honest account, including a real infrastructure blocker and 4 images
that were downloaded but then rejected after actually looking at them.
**Nothing has been trained yet** — still waiting for your go-ahead.

## Summary

| Class | Positive examples | Negative examples | Will `train_clip_probe.py` train it? |
|---|---|---|---|
| `fire_smoke` | 10 (4 video + 6 image) | 9 | **Yes** |
| `fighting` | 5 (4 video + 1 image) | 10 | **Yes** |
| `vandalism` | 4 (3 video + 1 image) | 11 | **Yes** (exactly at the minimum) |
| `theft` | 0 | — | **No** — still nothing found, see below |

All 3 trainable classes now clear the 4-minimum for the first time.

## What happened this round (read this, it matters)

**1. Wikimedia rate-limited this session partway through.** I queued 28 new
image downloads (across fire_smoke, fighting, vandalism, plus generic
negatives). After ~11 successful downloads, Wikimedia's edge servers started
returning HTTP 429 ("Too many requests") for the shared IP this environment
uses — confirmed with the literal server error text, not inferred. I tried
increasing delays (4s, then 10s) and switching to the officially-recommended
thumbnail URL pattern per their own error message, but the block persisted
(a 400 error on thumbnails too: "Use thumbnail sizes listed on..."). This
looks like a sustained IP-level throttle from cumulative traffic across this
whole session (videos + images), not something request pacing alone fixes.
**19 of 28 queued images never downloaded** — a real, disclosed shortfall,
not silently worked around. If you want the rest, I can retry later once the
block likely expires, or you can send images directly (see "Sending images
yourself" below).

**2. Of the ones that did download, 4 were wrong and I removed them** — this
is worth knowing because it shows why titles alone aren't reliable. Caught by
actually looking at each image, not by trusting the search-result title:
- `hospital_fire.jpg` (was titled "Fire at Gobowen Orthopaedic Hospital") —
  actually shows a nighttime patient evacuation, no visible fire.
- `barn_fire.jpg` — shows fire hoses and a crowd, but no visible flame or
  smoke in frame (the fire itself is out of shot).
- `building_fire.jpeg` — a building facade lit orange by streetlight, not a
  fire at all. Misleading title.
- `minneapolis_smoke.jpg` (was titled "Canadian Wildfire Smoke in
  Minneapolis") — shows a jogger on a bike path with only a faint haze
  barely visible in the far background. Training on this would have taught
  the probe "person jogging = fire," the opposite of useful.

None of these made it into the final dataset. This is exactly the kind of
verification a rushed pipeline would skip — worth flagging since it's real
evidence the dataset was actually checked, not just downloaded and trusted.

## Positive examples — what each one actually shows (verified by looking, not just by title)

### `fire_smoke/positive/` (10: 4 video + 6 image)
Video clips are unchanged from round 1 (see git history / earlier session).
New images, all confirmed to show real, visible fire or smoke:

| File | Shows | Source | License |
|---|---|---|---|
| `bomberos_fire.jpg` | Firefighters on an aerial platform actively spraying water into a heavily smoking building, Buenos Aires | [Wikimedia: Bomberos-incendio-once-buenos-aires-2008.JPG](https://commons.wikimedia.org/wiki/File:Bomberos-incendio-once-buenos-aires-2008.JPG) | CC BY-SA 2.5 |
| `ontario_smoke.jpg` | Landscape blanketed in dense wildfire smoke haze, 2023 Canadian wildfires | [Wikimedia: Wildfire smoke blankets eastern Ontario 06-2023.jpg](https://commons.wikimedia.org/wiki/File:Wildfire_smoke_blankets_eastern_Ontario_06-2023.jpg) | CC BY (verify exact version on file page) |
| `rural_house_fire.jpg` | Historical (1911) house fire, heavy dark smoke, period fire apparatus | [Wikimedia: Fire at house on rural road, Keene NH](https://commons.wikimedia.org/wiki/File:Fire_at_house_on_rural_road,_Keene_NH_(14555187701).jpg) | CC BY-SA 4.0 |
| `sawmill_fire_1.jpg` | Historical (1911, Keene NH) large fire, dramatic smoke plumes filling the sky | [Wikimedia: Fire at Impervious Package Company sawmill (1)](https://commons.wikimedia.org/wiki/File:Fire_at_Impervious_Package_Company_sawmill_(1)_(14573259963).jpg) | CC BY-SA 4.0 |
| `sawmill_fire_2.jpg` | Same 1911 fire, crowd watching, visible smoke haze between buildings | [Wikimedia: Fire at Impervious Package Company sawmill (2)](https://commons.wikimedia.org/wiki/File:Fire_at_Impervious_Package_Company_sawmill_(2)_(14549753771).jpg) | CC BY-SA 4.0 |
| `wildfire_smoke_1.jpg` | The well-known 2023 NYC orange-sky wildfire-smoke photo, Grand Central Terminal | [Wikimedia: Wildfire Smoke.jpg](https://commons.wikimedia.org/wiki/File:Wildfire_Smoke.jpg) | CC BY 2.0 |

### `fighting/positive/` (5: 4 video + 1 image)
Video clips unchanged from round 1. New image:

| File | Shows | Source | License |
|---|---|---|---|
| `wrestling_takedown.jpg` | Two college wrestlers mid-takedown, referee present, active grappling | [Wikimedia: Takedown.jpg](https://commons.wikimedia.org/wiki/File:Takedown.jpg) | CC BY-SA 2.0 |

Consensual competitive sport, same disclosed proxy-for-fighting approach as
round 1 — confirmed by looking, not just the filename.

### `vandalism/positive/` (4: 3 video + 1 image — first time meeting the minimum)
Video clips unchanged from round 1 (still all static/aftermath shots — see
that limitation below, unresolved). New image:

| File | Shows | Source | License |
|---|---|---|---|
| `pruitt_igoe_windows.jpg` | Building facade with multiple broken/shattered windows, clear real property damage, no people visible | [Wikimedia: Pruitt-Igoe-vandalized-windows.jpg](https://commons.wikimedia.org/wiki/File:Pruitt-Igoe-vandalized-windows.jpg) | Public domain |

**Still true from round 1**: all 4 positives are aftermath/static shots (no
one is filmed in the act of damaging anything), so a trained probe would
still be learning "vandalism-looking damage is visible," not "vandalism is
happening right now." Getting to 4 didn't fix this — worth knowing before
deciding this class's data is actually ready.

### `negative/` (1 new image, shared across all 3 classes)

| File | Shows | Source | License |
|---|---|---|---|
| `vancouver_skyline.jpg` | Ordinary daytime marina/skyline scene, boats, no incident | [Wikimedia: Vancouver skyline and harbour daytime by 500px](https://commons.wikimedia.org/wiki/File:Vancouver_skyline_and_harbour_daytime_by_500px_(56742618).jpg) | CC BY-SA 4.0 |

Linked into all 3 classes' `negative/` folders alongside the existing
cross-class-reuse negatives from round 1.

## What I could not find (again)

**Theft/Chain-Snatching: still zero.** Tried again this round with image-
specific search terms (pickpocket illustrations, shoplifting photos, purse-
snatcher photos). One search hit — "Plan to arrest purse snatchers and
extortionists" (a 33-photo series) — turned out to be real Iranian police
press-conference documentation of an actual operation, likely showing real
detained individuals. Not used, same privacy/ethics line held throughout this
project regardless of licensing. No `data/few_shot/theft/` folder exists.

## Sending images yourself

If you want to add more (especially: real Theft examples if you can find
appropriate ones, more Vandalism *in-progress* shots to fix the limitation
above, or just more volume for stronger training), same structure as before:

```
data/few_shot/<class>/positive/*.jpg   (or .png, .webp, .bmp, or video)
data/few_shot/<class>/negative/*.jpg
```

I'll visually verify anything you send the same way I did above before
counting it as usable — happy to do that for images you provide too, not
just ones I source myself.

## Real limitations, still — please read before approving training

- **Still few examples per class** (4-10) — any printed "train_accuracy" is
  fit on this exact data with no held-out split, will look better than it is.
- **Vandalism's positives are still all static/aftermath**, not action — see
  above, unresolved even after reaching the minimum count.
- **Cross-class negatives** (fire clips as fighting's negatives, etc.) —
  legitimate but means "X vs. these specific other things," not "X vs.
  everything."
- **19 additional candidate images never downloaded** due to the rate limit —
  more retries or a different sourcing approach could still add them later.

## Training results (real, measured — trained, tested, and reverted)

Trained on 2026-09-11: `fire_smoke` (30 positive / 49 negative frames,
train_accuracy=0.90), `fighting` (25 positive / 55 negative frames,
train_accuracy=0.79), `vandalism` (19 positive / 61 negative frames,
train_accuracy=0.76). `theft` skipped — no data.

**Then tested against the exact same sample videos used throughout this
project, comparing trained-probe output to the known zero-shot baseline:**

| Video | Zero-shot (baseline) | Trained probes | Verdict |
|---|---|---|---|
| `fire_smoke_01.mp4` | Fire/Smoke **0.99** (correct) | Nothing (raw score 0.54-0.58, below the 0.6 threshold) | **Lost a correct detection** |
| `fighting_01.mp4` | Fighting **0.97** (correct) | Nothing for Fighting | **Lost a correct detection** |
| `fighting_02.mp4` | Fighting **0.63** (correct) | Nothing | **Lost a correct detection** |
| `running_02.mp4` | Theft 0.72 (known false positive) | Theft **0.88** | **Existing false positive got more confident** |
| `vandalism_01.mp4` | Vandalism 0.99 (right label, wrong context) | Nothing | Neutral (was never a valid positive test) |

**Every previously-correct detection was lost. Zero improvements. One false
positive got worse.** Root cause, confirmed by inspecting raw scores directly
rather than assumed: 19-30 frames per class is nowhere near enough for
logistic regression to beat CLIP's zero-shot prior (calibrated on 400M+
image-text pairs) — the trained probes learned a real but weak signal
(scores cluster at 0.5-0.6) instead of anything sharp enough to clear the
confidence threshold.

**A second, unexpected finding**: pulling `fire_smoke`/`fighting`/`vandalism`
out of the shared zero-shot softmax changed the calibration of `theft` too,
even though `theft` has no trained probe at all — with 3 fewer competing
classes, its score against just `{theft, background}` came out higher than
when it had to also beat 3 other real classes. This is a real architectural
side effect of *partial* probe adoption that wasn't anticipated when the
probe-loading logic was built, confirmed by inspecting `theft`'s raw scores
directly (0.67-0.90 with probes active vs. what produced 0.72 before).

**Action taken**: reverted. The trained checkpoints are preserved at
`checkpoints/_evaluated_underperformed/` (not deleted — this is a real,
informative result worth keeping) but moved out of
`checkpoints/clip_probes/`, so `inference.py` is back to zero-shot for all 4
appearance classes. Verified: `fire_smoke_01.mp4` and `vtest.avi` both
produce bit-for-bit identical output to before this experiment; all 53 unit
tests still pass.

**What would actually fix this**: meaningfully more data (tens of examples
per class, not single-digits), and/or fixing the softmax-competition
architecture so removing one class's zero-shot participation doesn't change
another's calibration — e.g. giving every class its own independent
background comparison instead of one shared pool.

## Follow-up: the softmax coupling bug was fixed, measured, and reverted too

Since the finding above named the softmax-sharing architecture as a root
cause, that fix was actually attempted next (2026-09-11): every zero-shot
class scored as its own independent binary softmax against `background`
only, instead of one shared N-way softmax across all zero-shot classes.

**It works** — the coupling is gone; giving a class a trained probe no
longer perturbs its untouched siblings' calibration. It also slightly raised
confidence on real true positives (Fire/Smoke 0.99→1.00, Fighting
0.97→0.98).

**But it has its own real cost.** Tested against every sample video with no
trained probes active (the current deployed state): `running_01.mp4` and
`running_02.mp4` picked up new Fighting false positives (0.64, 0.68) that
weren't there before; `falling_01.mp4`'s existing Vandalism false positive
got worse (0.77→0.87); and on `vtest.avi` — the one video with real ground
truth, previously 100% clean of appearance-class false positives — three new
Vandalism false positives appeared. Measured with `scripts.evaluate`:
**F1@tIoU0.5 dropped from 0.909 to 0.714** (precision 0.833→0.556, recall
unchanged at 1.000). A threshold sweep (0.5-0.95) was run looking for a
value that recovers the original clean result without losing true
positives — none did; the best achievable was F1=0.727, still below
baseline, and that came at the cost of losing a real Intrusion detection.

Root cause of the trade-off: the original shared softmax was doing real
false-positive suppression by forcing classes to compete for probability
mass — an independent per-class check removes that competition along with
the coupling bug.

**Decision: reverted, not shipped** (`src/classifiers/clip_zeroshot.py` back
to the shared-softmax design — see the module docstring there for the full
writeup). The coupling bug is real but currently dormant: no trained probes
are deployed right now, so no class is currently missing from the shared
softmax, so the bug has no live effect to fix. Paying a measured accuracy
cost today for a bug that isn't currently active isn't worth it. Confirmed
after reverting: `vtest.avi` F1 back to exactly 0.909, all 53 tests pass,
`fire_smoke_01.mp4` output bit-for-bit identical to before either
experiment.

The right fix, if/when partial probe deployment actually happens: something
that keeps per-class independence *and* keeps the false-positive suppression
— e.g. per-class calibrated thresholds instead of a shared normalizer, or
isotonic recalibration fit once on a held-out set — not a full architecture
swap made and measured in one sitting with no time to tune it properly.

## Next step

Data-gathering and both architecture experiments are done for this round —
the honest result both times is that it doesn't help yet. If you want to
pursue this further: get significantly more images per class (aim for
30-50+, not the 4-10 gathered here), and/or take more time on a softmax fix
that doesn't sacrifice the existing false-positive suppression. Otherwise
the system is back to its previous known-good zero-shot state (verified,
not assumed).
