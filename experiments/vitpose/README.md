# ViTPose+ experiment: climbing self-occlusion

> Historical measurement report. On 8 September the user approved retaining
> this as an optional backend, with model choice before computation and a
> skeleton-only panel to the right of the video. References below to unchanged
> API/frontend and model-blind caching describe the measured experiment commit,
> not the subsequent selectable-model implementation. Accuracy conclusions and
> measurements are unchanged; this is still not a recommendation to make
> ViTPose the default.

Measured 7 September 2026; report completed 8 September. Base:
`ed0cd39508aa9d7e6b33c187409eebf4d442ffe3`. Experimental branch:
`codex/vitpose-experiment`. **Do not merge or change the default on the
strength of this experiment.**

## Verdict

ViTPose+ improves temporal consistency on this clip, particularly when the
right arm is hidden by the torso. It is **not a clear overall replacement**:
median full-frame CPU inference is **536.2 ms, 8.7x MediaPipe Heavy** (23.2x
Lite), and the existing refinement pass leaves worse hidden-joint p99 tails
in two of three windows. The model still guesses wrong limb positions.
This is evidence of better consistency and some visually more plausible
poses, not a measured hidden-joint accuracy improvement.

It is worth keeping as an **opt-in research baseline** if that CPU cost is
acceptable for occasional offline comparisons. Keep Heavy as the personal
instance's default. Before promoting it, label more footage, quantify
visible/reappearing-joint error and identity errors, and profile the detector
and per-person pose stages separately. Evaluate refinement as a separate
ablation rather than assuming it benefits both estimators equally. No such
pipeline or tracking changes were made here.

## What was implemented

- Optional stateless `VitPoseEstimator` and a `vitpose` factory dispatch.
- Full-frame [RT-DETR R18](https://huggingface.co/PekingU/rtdetr_r18vd) person
  detection followed by batched
  [ViTPose+ Base](https://huggingface.co/usyd-community/vitpose-plus-base).
  Both use Transformers, avoiding a second tracking framework or YOLO/ByteTrack
  integration. Person confidence threshold 0.3; max five people.
- 17 named COCO landmarks and 19 skeleton edges, full-frame normalized x/y.
  Both running modes are independent frame detections. CPU only.
- Lazy optional dependencies and pinned Hugging Face checkpoint revisions;
  the MediaPipe `.task` downloader is never used for ViTPose.
- Timing-window/thread metadata in the existing benchmark script, a local
  recording wrapper around the unchanged app, shared-schema tracker replay,
  comparison tooling and 31 additional tests.

The only production Python changes are the new estimator and factory
dispatch. Tracking, refinement, storage, API, frontend, queue, config defaults,
Dockerfile and compose settings are unchanged. The default Docker image does
not install the experiment's optional dependencies.

**Contract limits:** COCO supplies no depth, so z=0 explicitly means unavailable,
not inferred 3D depth. Heatmap scores populate visibility/presence as proxies,
not MediaPipe-equivalent probabilities. These limits must be addressed before
future metrics depend on those fields.

## Clip, setup and provenance

The available local file was `video_20260902_190236~2.mp4`, not the historical
31-second README fixture. Source SHA-256:

`207c844359ca261ca6acc844b1b87456281000949fde989cf759958afee5ad2c`

The real upload endpoint normalized it from 1080x1920 VP9 at approximately
29.97 fps to **720x1280, 30 fps, 1,033 frames / 34.433 seconds**.
Normalized SHA-256:

`b9e4d2744176611fe738a873409073fcda3cf498a4cdac9701418b8091660c2c`

Both full-clip jobs used that exact normalized file and video ID
`9a26c23fe9a64eceb62600f6928a481d`, selecting climber 0 at frame 240 (8 seconds).
Native seeds came from each model's actual picker. The experiment explicitly
refreshed the model-blind candidate cache between backends.

| Backend | Analysis run | Seed box x, y, width, height |
|---|---|---|
| ViTPose+ | `811d451c25fa4a08ab95041af940eedb` | 0.405698, 0.387233, 0.154730, 0.203962 |
| MP Heavy | `8f0bcf24ff9045efb9870841d2d7384e` | 0.408382, 0.380631, 0.160543, 0.229887 |

Native runs retain existing refinement (margin 0.55, agreement threshold 0.5)
and tracker settings (minimum IoU 0.3, unlimited gap recovery). Model revisions:

- RT-DETR: `ac77a11ff0170a41b771c03264987f8ce2b0d753`.
- ViTPose+: `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d`.

Machine: Intel Core i7-9700K, Docker Desktop/WSL2, eight exposed CPU cores;
Python 3.11.16, torch 2.7.1+cpu, torchvision 0.22.1+cpu, Transformers 4.57.6,
SciPy 1.15.3, NumPy 2.4.6, MediaPipe 1.0.1, OpenCV 5.0.0.
PyTorch, OMP and MKL used four threads; MediaPipe retained its own threading.
Models ran locally; no footage was sent to a hosted inference service.

## Comparison method

[Intervals](segments.json) were selected from source-frame contact sheets
before examining model outputs. They cover the folding left leg, the right
arm behind the upper body, and a tucked posture hiding the right arm/leg.
The foreground passer around 14–15 seconds is excluded from these
self-occlusion windows. End frames are exclusive.

Only left/right **shoulders, elbows, wrists, hips, knees and ankles** are
compared. Coordinates are converted to pixels to respect the portrait aspect
ratio. Jitter is the median across the selected joints of

`||p[t+1] - 2 p[t] + p[t-1]|| / shared_body_height[t]`.

This follows the README's frame-to-frame acceleration/body-height methodology,
with the exact aggregation now specified and tested. Units are body-heights
per frame squared, not per second squared. Percentiles use linear
interpolation across frame medians. No smoothing, confidence filtering,
gap interpolation or triples across interval edges. API rounding is retained.
Historical README whole-clip/33-joint values are **not directly comparable**.

A tracked frame means the tracker returned a pose, not that identity or every
hidden joint is correct. There is no annotated ground truth for hidden joint
locations. Low jitter can describe a consistently wrong or frozen pose.

### Shared-schema association and first-pass jitter

Using native 33-joint and 17-joint boxes for the main IoU comparison would
confound model performance with schema. Instead, all recorded full-frame
detections were projected to the shared 12 joints and replayed through the
**unchanged** `IouTracker`, with common seed box
`(0.39, 0.38, 0.19, 0.24)` at frame 240 and identical tracking configuration.
Both directions follow the existing seed-forward/pre-seed-backward method.
This diagnostic uses no new inference or refinement and does not replace
either application's stored result.

| Window | Model | Tracked | Match IoU | Jitter median | p90 | p99 | Samples |
|---|---|---:|---:|---:|---:|---:|---:|
| Folding leg · 11.5–12.5 s | MP Heavy | 90.0% | 0.865 | 0.0715 | 0.1183 | 0.1420 | 23 |
| Folding leg · 11.5–12.5 s | ViTPose+ | 100.0% | 0.953 | 0.0151 | 0.0287 | 0.0353 | 28 |
| Hidden arm · 16–18.5 s | MP Heavy | 100.0% | 0.948 | 0.0198 | 0.0389 | 0.0460 | 73 |
| Hidden arm · 16–18.5 s | ViTPose+ | 100.0% | 0.980 | 0.0078 | 0.0169 | 0.0220 | 73 |
| Tucked posture · 21.5–25.5 s | MP Heavy | 99.2% | 0.952 | 0.0373 | 0.0577 | 0.2067 | 115 |
| Tucked posture · 21.5–25.5 s | ViTPose+ | 100.0% | 0.972 | 0.0100 | 0.0207 | 0.0352 | 118 |

Jitter sample counts exclude interval edges and gaps. Recomputing on the
same valid centers in both models preserves the conclusion: ViTPose paired
median jitter is 0.0160, 0.0078 and 0.0099 respectively, versus Heavy's
0.0715, 0.0198 and 0.0373. All joint-level distributions and paired
hidden-joint distributions are included in the local detailed exports.

### Actual delivered results: existing refinement included

These are the raw, unsmoothed API artifacts actually displayed by the app,
projected to the shared 12 joints **for jitter only**. Native tracking still
uses its original model schema; its match scores are recorded separately in
[results.json](results.json), not presented as schema-equivalent here.

| Window | Model | Tracked | Jitter median | p90 | p99 | Samples |
|---|---|---:|---:|---:|---:|---:|
| Folding leg · 11.5–12.5 s | MP Heavy | 90.0% | 0.0292 | 0.0491 | 0.0612 | 23 |
| Folding leg · 11.5–12.5 s | ViTPose+ | 100.0% | 0.0241 | 0.0340 | 0.0566 | 28 |
| Hidden arm · 16–18.5 s | MP Heavy | 100.0% | 0.0143 | 0.0234 | 0.0338 | 73 |
| Hidden arm · 16–18.5 s | ViTPose+ | 100.0% | 0.0093 | 0.0186 | 0.0267 | 73 |
| Tucked posture · 21.5–25.5 s | MP Heavy | 99.2% | 0.0278 | 0.0532 | 0.1615 | 115 |
| Tucked posture · 21.5–25.5 s | ViTPose+ | 100.0% | 0.0133 | 0.0289 | 0.0442 | 118 |

The whole-body median can mask a bad hidden wrist or ankle. Below are the
labelled **hidden joints only**, evaluated on the **same valid three-frame
centers in both runs**:

| Window | Model | Tracked | Jitter median | p90 | p99 | Samples |
|---|---|---:|---:|---:|---:|---:|
| Folding leg · 11.5–12.5 s | MP Heavy | 90.0% | 0.1188 | 0.4168 | 0.5989 | 23 |
| Folding leg · 11.5–12.5 s | ViTPose+ | 100.0% | 0.0828 | 0.5553 | 0.8260 | 23 |
| Hidden arm · 16–18.5 s | MP Heavy | 100.0% | 0.0503 | 0.2433 | 0.4096 | 73 |
| Hidden arm · 16–18.5 s | ViTPose+ | 100.0% | 0.0168 | 0.0544 | 0.1048 | 73 |
| Tucked posture · 21.5–25.5 s | MP Heavy | 99.2% | 0.0546 | 0.1412 | 0.1982 | 115 |
| Tucked posture · 21.5–25.5 s | ViTPose+ | 100.0% | 0.0345 | 0.1005 | 0.2473 | 115 |

The arm result is the strongest: hidden-arm median is about 66% lower and p99
about 74% lower. But folding-leg p99 is approximately **38% worse**, and tucked
hidden-limb p99 approximately **25% worse**, despite better medians. Refinement
also increases ViTPose's shared-body jitter relative to its own first pass
in each window. This was measured, not fixed by changing refinement unasked.

Visual review of source / Heavy / ViTPose panels at frames 345, 360, 374,
480, 510, 554, 645, 705 and 750 supports a narrow conclusion. ViTPose avoids
some conspicuous arm misplacements (for example, Heavy pulls the right wrist
toward the leg at frame 480), but both models invent positions for hidden
limbs. ViTPose's folded-leg pose also jumps; a stable torso cannot establish
that an invisible ankle is correct. The panels draw only shared joints at
full opacity; the actual UI retains its existing confidence opacity.

## CPU cost

The existing benchmark ran all four backends **serially**, after analysis and
tests had completed, on frames 645–764 (the 21.5–25.5 s window), at the same
resolution, five-person cap and VIDEO mode. Each had ten discarded warmup
frames and three repeats. The script's existing fastest-median-repeat
convention is preserved and all repeat medians are shown.

This includes detector + pose inference per full frame, but excludes model
load/download, decode and the second refinement pass. It is not the cost of
posing one oracle crop.

| Backend | Mean ms/frame | Median | p90 | p99 | Repeat medians (ms) |
|---|---:|---:|---:|---:|---|
| lite | 23.0 | 23.1 | 24.0 | 24.7 | 23.1 / 23.2 / 23.3 |
| full | 28.1 | 27.9 | 29.7 | 31.6 | 27.9 / 27.9 / 28.0 |
| heavy | 62.0 | 61.5 | 63.8 | 69.6 | 61.5 / 65.1 / 64.8 |
| vitpose | 548.5 | 536.2 | 604.4 | 717.0 | 555.2 / 536.2 / 564.3 |

ViTPose is **8.7x Heavy by median / 8.9x by mean**, and **23.2x Lite by median**.
Including decode, throughput was 1.80 fps versus Heavy's 14.90 fps. The
measurement window contained about one detected person per frame; more people
can increase ViTPose cost because each retained box needs a pose.

Observed real API jobs, including refinement and artifact publication, took
**17m40s for ViTPose** and **2m39s for Heavy** on this 34.4-second clip. These
are operational observations, not a second controlled benchmark: some tests
and preparation overlapped the ViTPose run. Load/download costs were not
included in the isolated steady-state timing table.

Full timings and environment: [timing.json](timing.json).

## End-to-end verification and tests

Actually executed:

- Real clip upload and async normalization into an isolated local data store.
- Real candidate detection, analysis jobs and run-specific Parquet/API
  retrieval with both backends.
- ViTPose full-frame recording contained multiple people in **126 frames**,
  with up to five detections. This establishes a real multi-person path,
  not perfect detector recall/precision.
- Opened the completed ViTPose run in the existing frontend, disabled
  smoothing, played the video and visually verified the 17-joint overlay.
  Run selection, metadata and coverage/confidence charts rendered.
- Complete suite: **161 passed** (130 existing + 31 added), with clean
  `pyflakes app tests scripts`, under Python 3.11 / FFmpeg 7.1.5.
- The deployment-image test attempt produced **159 passed, 2 failed**:
  both failures were fixture construction using `-display_rotation`, which
  the image's FFmpeg 5.1.9 does not support. The same unchanged tests passed
  in the newer-FFmpeg environment. No application fix or weakened test was
  used to hide this environment mismatch.
- Two dependency deprecation warnings concerned Starlette/httpx/AnyIO.
  No GitHub CI or merge is claimed for this local experiment.

New tests cover multi-person conversion, offset/non-square full-frame
normalization, BGR/RGB, labels, score proxies, zero depth, stable 17-point
schema, invalid output, empty detections, stateless calls, close, optional
dependency dispatch/default preservation, benchmark timestamps, shared-joint
mapping and jitter mathematics, gaps, paired windows, incompatible inputs,
and replay's original greedy matching/configuration.

## Remaining issues and reproduction

[Setup and commands](../../docs/VITPOSE_EXPERIMENT.md) document optional CPU
installation, local serving, benchmark, replay and comparison. Aggregate
metrics and immutable run provenance are in [results.json](results.json).

Raw clip, Parquet, raw detections and diagnostic images remain **local,
ignored data**, not committed footage:
`data/vitpose-experiment/` and `data/vitpose-experiment-evidence/`.
The latter contains `native-{leg,arm,tuck}.jpg`, shared-first-pass panels,
full comparison JSON, run-specific exports and test XML.

Important limits before any adoption:

1. One clip and three short windows cannot establish general climbing
   accuracy. Hidden-joint and identity ground truth are missing.
2. This compares two complete detection/pose backends; it does not isolate
   the pose network from detector quality.
3. The current refinement pass may return multiple people on a crop and
   chooses the first prediction. This experiment leaves that rule intact;
   model-specific behavior needs a separately scoped evaluation.
4. AnalysisRun captures the variant, not individual detector/checkpoint
   revisions or dependency versions. Pins and this manifest supply those
   externally; the storage schema was deliberately not changed.
5. Candidate caches are model-blind. Refresh them explicitly when changing
   experimental backends on the same data directory.
6. COCO confidence and unavailable depth need an explicit semantic contract
   before consumers treat them as MediaPipe-equivalent measurements.
7. Optional large dependencies/checkpoints and a roughly nine-fold CPU cost
   are material operational costs, not just an implementation detail.

The experiment does not start a public tunnel or alter the ordinary server
on port 8000. The isolated experiment container is currently stopped; saved
results remain on disk. No new default, tracking algorithm, climbing metric,
API or frontend behavior has been introduced.
