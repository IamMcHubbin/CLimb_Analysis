# Optional ViTPose experiment

This is an opt-in CPU experiment, not the default estimator or a deployment
recommendation. See [the measured experiment](../experiments/vitpose/README.md).
Tracking and refinement remain unchanged. The app now supports per-run model
selection and synchronized video/skeleton-only panels. Default dependencies
remain MediaPipe-only; ViTPose installation is optional.

## Backend and limits

`create_pose_estimator(variant="vitpose")` lazily loads:

- [RT-DETR R18](https://huggingface.co/PekingU/rtdetr_r18vd), revision
  `ac77a11ff0170a41b771c03264987f8ce2b0d753`: full-frame person detector.
  COCO person class 0, confidence >= 0.3, descending confidence, up to the
  existing `num_poses` limit. Boxes are clipped and converted to COCO xywh.
- [ViTPose+ Base](https://huggingface.co/usyd-community/vitpose-plus-base), revision
  `92be54d7a29e42fad47b6e2ca01dd9e685a61e0d`: poses every retained box in a batch,
  using COCO expert 0 for each person. No single-person shortcut or ByteTrack.

Both are supported by the same Transformers dependency; this keeps the adapter
small without introducing another tracking system. Safetensors weights are
pinned and loaded through `from_pretrained` (roughly 583 MB combined), using
Hugging Face's cache, never the MediaPipe `.task` downloader. They run locally
on CPU; footage is not uploaded to Hugging Face.

Outputs contain 17 named COCO landmarks and skeleton edges. Postprocessed
full-frame pixel coordinates become normalized, clipped x/y in [0, 1]. COCO
has **no depth**: z is zero as an explicit unavailable-depth placeholder, not
a reconstructed hip-relative 3D estimate. Visibility and presence both carry
clipped heatmap confidence, **not** MediaPipe's calibrated visibility/presence
semantics. Low-confidence joints remain in place; never infer actual visibility
from equal confidence thresholds across models.

IMAGE and VIDEO calls are stateless. The app's existing greedy IoU tracker
retains all identity and gap handling. The existing crop-refinement pass also
runs unchanged (including its first-person and agreement rules). This is an
end-to-end backend comparison, not a controlled pose-only comparison with
identical oracle detector boxes.

## Install and run locally

Use Python 3.11 and the existing system dependencies (FFmpeg and, on Linux,
the GL libraries listed in README). In a separate virtual environment, these
commands are identical in PowerShell and Linux/macOS shells; `python` must
refer to that environment's interpreter:

```text
python -m pip install -r requirements-dev.txt
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-vitpose.txt
python scripts/serve_pose_experiment.py --model vitpose --data-dir data/vitpose-experiment --records-dir data/vitpose-recordings --port 8001
```

Open http://localhost:8001, upload the clip, choose the model, click **Detect
people**, select the climber and analyse. Opening the picker or changing its
model does not run inference until **Detect people** is pressed.
The default server on port 8000 need not be stopped. **Use a separate data
directory**: this wrapper changes the selected model only for this process
and records detections for comparison. Ctrl+C stops it; the same command
restarts it with the existing experiment data. It starts no public tunnel.
Allow several minutes for the first model download and CPU analysis.

To compare MediaPipe, choose **New analysis / change model** and select Heavy
in the same server. No restart or separate data directory is required per
model. Each run retains its own model, seed and artifact; the run selector
reopens completed results. The existing queue runs jobs one at a time.
Candidate caches are model-aware, and stale selection IDs are rejected.

For shell-based configuration without the recording wrapper:

| PowerShell | Linux/macOS |
|---|---|
| `$env:CLIMB_POSE_MODEL = 'vitpose'` | `export CLIMB_POSE_MODEL=vitpose` |
| `$env:CLIMB_DATA_DIR = 'data/vitpose-experiment'` | `export CLIMB_DATA_DIR=data/vitpose-experiment` |
| `$env:OMP_NUM_THREADS = '4'` | `export OMP_NUM_THREADS=4` |
| `$env:MKL_NUM_THREADS = '4'` | `export MKL_NUM_THREADS=4` |

Then `python -m uvicorn app.main:app --host 127.0.0.1 --port 8001` in either
shell. An ordinary Docker image leaves ViTPose visible but disabled in the
selector, with an installation explanation. To enable it, these commands work
in PowerShell and Linux/macOS (from the repository root):

```text
docker compose build --build-arg INSTALL_VITPOSE=true app
docker compose up -d --no-build
```

This installs CPU dependencies but leaves Heavy as the compose default.
First ViTPose use downloads the pinned checkpoints through Hugging Face.
The weight cache must be persisted separately if it should survive container
replacement. Rebuilding without the build argument returns to a MediaPipe-only
image; completed ViTPose runs remain viewable without its runtime installed.

## Reproduce measurements

Use the *same normalized file*, FPS and resolution for every model. Run timing
alone, not concurrently with analysis, tests or another benchmark. For example
(one line works in either shell):

```text
python scripts/benchmark_pose.py --video NORMALISED.mp4 --models lite,full,heavy,vitpose --num-poses 5 --start-frame 645 --max-frames 120 --warmup-frames 10 --repeats 3 --torch-threads 4 --json timing.json
```

This reports detector + pose inference per full frame, excluding load/download,
decode and the second refinement pass. Timing retains the existing script's
fastest-repeat convention and reports all repeat medians. MP retains its own
threading; the PyTorch thread count is explicit. Warmup is discarded and input
timestamps continue monotonically after seeking back to the measured window.

For native results, export the unchanged endpoint
`GET /videos/{video_id}/keypoints?analysis_run_id={run_id}` for each model.
The wrapper additionally writes one gzip JSONL recording per estimator
instance: a header, then all people's detections and inference times per call.
Use the full-frame VIDEO recording with exactly one row per source frame,
**not** the picker or variable-size refinement recording, for replay:

```text
python scripts/replay_pose_detections.py --recording FULL_PASS.jsonl.gz --seed-frame 240 --seed-box 0.39 0.38 0.19 0.24 --frame-count 1033 --fps 30 --video-id VIDEO_ID --json shared.json
python scripts/compare_pose_runs.py --run heavy=heavy-shared.json --run vitpose=vitpose-shared.json --segments experiments/vitpose/segments.json --width 720 --height 1280 --json shared-comparison.json
```

Replay projects every person's detections onto the same 12 named joints, then
uses the **unchanged** `IouTracker` with one common visually selected seed box,
IoU 0.3 and unlimited gaps, forwards and backwards from the seed. It does not
redetect, refine, interpolate, smooth or replace application results. This
separates schema-equivalent matching from native full-schema box differences.
Compare native API exports in a *separate* invocation for the actual delivered
refined overlay; do not mix them with replay exports.

Intervals are visually labelled on the source before reviewing model outputs;
end frames are exclusive. The 12 shared joints are left/right shoulders,
elbows, wrists, hips, knees and ankles. Jitter is the median across joints of
pixel-coordinate second differences divided by that frame's shared-joint body
height. Report median/p90/p99 across frame medians, without smoothing or
confidence filtering; no triples bridge gaps or interval edges. The JSON also
reports the labelled hidden-joint subset and paired valid frames, so a median
over stable visible joints cannot hide a bad wrist and gaps cannot look like
zero jitter. Native match IoU uses each model's own full-schema tracker box;
replay match IoU is shared-schema. Adjacent shared-box IoU is a separate
consistency diagnostic, **not** the tracker's match score.

Coverage and low jitter do not establish hidden-joint accuracy. Without
ground-truth joint locations, report visual observations and trade-offs, not
an accuracy percentage. No identity labels exist either: a non-null track
is not proof that the correct climber was followed.

## Reproducibility limits

AnalysisRun records the backend variant and existing configurations, but not
the separate detector/checkpoint revisions, dependencies, CPU threads or clip
hash. The experiment report and pinned constants supply those separately;
changing the provenance schema is out of scope. Candidate caches are not
keyed by model in the original measurement; current candidate sets include
model and selection identity. Confidence and depth semantics need a richer contract before
cross-model metrics could rely on them. Those are follow-up issues, not reasons
to alter tracking, storage or the API in this experiment.
