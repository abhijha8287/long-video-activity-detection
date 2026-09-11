# Sample video provenance

Sample videos for demo runs (`python inference.py --video data/samples/<name>.mp4`
or the Streamlit UI, `streamlit run app.py`). **None of these have hand-authored
ground truth** — they are demo/sample-run material, not part of the scored
evaluation in `report.md` (which is `vtest.avi` only, §3). Two categories
(Theft/Chain-Snatching, Abandoned Object) have no video here — see "What's
missing and why" below.

| File | Category (intended) | Source | License |
|---|---|---|---|
| `vtest.avi` | Crowd Formation, Intrusion (scored eval video) | OpenCV `samples/data/vtest.avi` | Public domain (BSD-licensed project sample) |
| `fire_smoke_01.mp4` | Fire / Smoke | [Wikimedia Commons: Table Rock Fire.webm](https://commons.wikimedia.org/wiki/File:Table_Rock_Fire.webm) | CC BY-SA 4.0, © Kencf0618 |
| `fighting_01.mp4` | Fighting / Physical Assault | [Wikimedia Commons: EVD-taekwondo-003.ogv](https://commons.wikimedia.org/wiki/File:EVD-taekwondo-003.ogv) | CC BY-SA 2.5 |
| `fighting_02.mp4` | Fighting / Physical Assault (second example) | [Wikimedia Commons: JudoVideo.net Okuri-ashi-barai.webm](https://commons.wikimedia.org/wiki/File:JudoVideo.net_Okuri-ashi-barai.webm), trimmed to a 9s segment (original offset 12-21s) | CC BY 3.0 |
| `falling_01.mp4` | Falling (person collapse) | [Internet Archive: "Down and Out"](https://archive.org/details/down_and_out) (1971 workplace safety film, A/V Geeks collection), trimmed to a 20s segment (original offset 65-85s) | Freely streamable/downloadable on Internet Archive; **rights status is not explicitly public-domain-tagged** — the item page directs broadcast/film licensing inquiries to A/V Geeks. Used here for local testing only; re-verify before any redistribution. |
| `running_01.mp4` | Running (in a non-running-permitted zone) | [Wikimedia Commons: 2019 Montevideo Sprint Triathlon American Cup — Elite men's race - 08.webm](https://commons.wikimedia.org/wiki/File:2019_Montevideo_Sprint_Triathlon_American_Cup_-_Elite_men%27s_race_-_08.webm) | CC BY-SA 4.0 |
| `running_02.mp4` | Running (second example, cleaner tracking) | [Wikimedia Commons: 2019 Montevideo Sprint Triathlon American Cup — Elite women's race - 17.webm](https://commons.wikimedia.org/wiki/File:2019_Montevideo_Sprint_Triathlon_American_Cup_-_Elite_women%27s_race_-_17.webm) | CC BY-SA 4.0 |
| `vandalism_01.mp4` | Vandalism | [Wikimedia Commons: Graffiti in Augsburg 5181 die bunten & brnzn.webm](https://commons.wikimedia.org/wiki/File:Graffiti_in_Augsburg_5181_die_bunten_%26_brnzn.webm) | CC BY-SA 4.0 |

Zone configs exist for the two Running clips: `configs/zones/running_01.yaml`,
`configs/zones/running_02.yaml` (both a full-frame `no_running` rectangle —
the same coarse-approximation approach as `configs/zones/vtest.yaml`).

## Important caveats — read before treating any result as a benchmark

- **`fighting_01.mp4` and `fighting_02.mp4` are competitive sparring (taekwondo,
  judo), not real violence.** Genuine assault footage isn't something to source
  or use, even where "freely licensed" versions exist (a web search for that
  turned up real violent-crime CCTV footage, which was discarded immediately,
  not downloaded). Sparring is used here as an honest, disclosed visual proxy —
  fast, aggressive-looking physical contact between two people — not a claim
  that this is what real fighting looks like in every respect.
- **`vandalism_01.mp4` shows a completed, legal mural** (b-roll from a Dutch
  news segment about a public art project), not an act of vandalism in
  progress. It was the best available third-person footage of graffiti-covered
  walls; running inference on it is actually an informative **negative test**
  for whether the appearance branch over-triggers on "graffiti visible in
  frame" regardless of legality — which it does (see below).
- **`falling_01.mp4`'s source item has ambiguous rights status** (see table).
  It is real, unstaged demonstration footage of a person falling — the most
  topically accurate clip in this set — but confirm licensing independently
  before using it beyond local testing.
- **`fighting_02.mp4` (the judo throw) was sourced to demonstrate Falling, not
  Fighting** — a person is visibly thrown from standing to lying on the mat,
  which is exactly the bbox-shape-change pattern the Falling rule targets.
  Tested at both 8fps and 15fps sampling: the geometry branch's Falling rule
  never fires on it (the practitioner doesn't hold post-throw stillness long
  enough in a repeated training rep), while CLIP correctly reads the grapple
  as Fighting. Kept and renamed rather than discarded, since it's a genuine,
  reproducible second data point for the appearance branch and an honest
  negative result for the geometry branch — not silently dropped because it
  didn't match the original intent.
- **Neither Running clip triggers the Running rule**, for two different,
  independently diagnosed reasons — real findings, not the same bug twice:
  - `running_01.mp4` (crowded triathlon transition area): extreme tracker ID
    churn from occlusion and a moving/panning camera — track IDs climb into
    the hundreds within 16s, so no track survives long enough to accumulate
    the required sustained-velocity duration. Confirmed via direct track-ID
    inspection, not assumed.
  - `running_02.mp4` (a single, cleanly-tracked runner, one track ID held for
    4.5+ continuous seconds): the tracker works fine here, but measured
    velocity — computed directly from track positions — peaked at 1.40
    bbox-heights/sec against a 2.0 threshold, across the best-tracked runner
    in the clip. This is real evidence the default `velocity_threshold_bbox_heights_per_s`
    (set from first-principles reasoning in the design doc, never empirically
    validated — no running-annotated video existed at the time) may be
    calibrated too high for real footage at 8fps sampling. **Not changed**:
    lowering it without a ground-truth video to check for new false positives
    on ordinary walking (e.g. in `vtest.avi`) would be exactly the kind of
    unvalidated threshold-chasing this project has otherwise avoided (see the
    `merge_gap_s` fix in `report.md` §8 for the validated version of this same
    kind of change). Named as an open, evidence-backed limitation instead.

## What's missing and why

**Theft/Chain-Snatching and Abandoned Object have no sample video.** Searched
repeatedly (Wikimedia Commons API and Internet Archive, multiple term
variations, across two separate sessions) and neither turned up appropriate,
freely-licensed, third-person footage — theft/mugging content on public video
platforms is either absent or is real crime-news footage (not usable here on
ethical grounds, independent of licensing), and staged "unattended luggage"
security-demo footage simply isn't common upload material; searches for
"suspicious package" surfaced real bomb-scare news footage, which was
discarded immediately, not downloaded. This is disclosed rather than worked
around with mislabeled substitute footage.

## Actual sample-run results (real, measured — see `outputs/predictions/*.json`)

| Video | Detected | Confidence | Note |
|---|---|---|---|
| `fire_smoke_01.mp4` | Fire / Smoke, 1.5-5.6s | 0.99 | Correct |
| `fighting_01.mp4` | Fighting / Physical Assault, 4.5-38.4s | 0.97 | Correct (sparring reads as fighting, as intended) |
| `fighting_01.mp4` | Falling (person collapse), 35.9-37.5s | 1.00 | Geometry branch also fired — a kick/throw likely produced a genuine bbox-shape flip; not verified frame-by-frame |
| `fighting_02.mp4` | Fighting / Physical Assault, 1.5-3.0s | 0.63 | Correct label; Falling rule did not fire despite an actual throw-to-mat (see caveats above) |
| `falling_01.mp4` | Vandalism, 18.0-20.0s | 0.77 | **False positive, and a miss on Falling** — old, grainy 1971 film footage likely degraded YOLO detection enough to break the bbox-shape-change rule; CLIP's background rejection didn't fully suppress a spurious Vandalism read either |
| `running_01.mp4` | (none) | — | Tracker ID churn prevents the velocity rule from accumulating (see caveats) |
| `running_02.mp4` | Theft / Chain Snatching, (full clip) | 0.72 | **False positive** — a fast-moving person near bikes/gear reads as "grabbing something and running away" to CLIP; Running rule did not fire (velocity below threshold, see caveats) |
| `vandalism_01.mp4` | Vandalism, 1.5-12.0s | 0.99 | "Correct" in label only — this is a legal, completed mural, not vandalism in progress. Real evidence that the appearance branch can't distinguish graffiti's *legality*, only its presence — a genuine, disclosed limitation of an appearance-only signal. |

(Segment end times are clamped to each video's actual duration —
`src/pipeline/run.py` — and `merge_gap_s` was raised from 1.0s to 2.0s after a
real, measured fix; see `report.md` §6/§8 for both.)

Timelines for each: `outputs/visualizations/<name>_timeline.png`.
