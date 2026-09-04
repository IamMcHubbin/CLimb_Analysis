# Climb Analysis

[![CI](https://github.com/IamMcHubbin/CLimb_Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/IamMcHubbin/CLimb_Analysis/actions/workflows/ci.yml)

Proof of concept: upload a climbing video, pick the climber, run 2D pose
estimation over the clip, play it back with a skeleton overlay. The question
this is meant to answer is whether tracking quality holds up on real climbing
footage — not to produce metrics or coaching feedback.

**Current state: the loop closes.** Drop a clip in, pick a person out of a
frame, watch the job run, then scrub the video with the skeleton drawn on top
and a coverage chart showing exactly where tracking held and where it did not.
Footage is deleted once it is no longer needed; the keypoints outlive it. No
metrics, no hold detection; smoothing is a display toggle, and the stored
keypoints stay raw.

---

## Quick start

Local, without Docker (needs `ffmpeg` and `ffprobe` on `PATH`):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_models.py lite heavy
.venv/bin/uvicorn app.main:app --reload
```

Then open <http://localhost:8000> to upload a clip and see what normalisation
did to it.

With Docker:

```bash
docker compose up --build
```

Deploying it somewhere: [docs/DEPLOY.md](docs/DEPLOY.md) for Fly.io,
[docs/SELF_HOST.md](docs/SELF_HOST.md) for your own machine behind a
Cloudflare tunnel. The app has no authentication, so read the section on that
in either before giving it a public address.

Run the tests:

```bash
.venv/bin/python -m pytest tests -q
```

CI runs the same lint and tests on every pull request and on pushes to
`main`, and separately
builds the Docker image and checks the container answers `/healthz` - which is
the only thing that actually exercises the Dockerfile.

## How slow is this going to be?

Measured on a 4-core Xeon @ 2.10GHz, CPU only, MediaPipe Tasks 1.0.1, on a
1280x720 clip. Figures are the fastest of several repeats; `p50` is the median
frame's inference time.

| Model | 1 person | 5 people | 60s clip, 5 people |
|-------|----------|----------|--------------------|
| lite  | 10.9 ms/frame (2.8x realtime) | 22.6 ms/frame (1.5x realtime) | ~40 s |
| full  | 14.1 ms/frame (2.3x realtime) | 26.6 ms/frame (1.3x realtime) | ~45 s |
| heavy | 47.7 ms/frame (0.77x realtime) | 58.9 ms/frame (0.64x realtime) | ~93 s |

Decoding costs about 0.7 ms/frame at 720p, so inference dominates. Ingest
normalisation of a 10-second 1080x1920 phone clip took 5.4 s, which includes
the extra decode pass that counts frames exactly.

What this means for the build: with `lite` or `full`, a single background
worker analyses a clip faster than real time. Multi-person detection roughly
doubles the per-frame cost even when only one person is in frame, because the
detector runs more often, and landmark refinement adds a second pass - so
budget about four times the single-pass figures above.

`heavy` is slower than real time here and is still the right choice on a
machine with CPU to spare: see the jitter section below for why. A 31-second
clip took 156 seconds end to end on this 4-core VM, against 75 with `lite`.
A desktop chip will be several times quicker than either.

Caveats worth knowing before trusting these numbers:

- Measured on a synthetic clip (a photo of one person panned and zoomed), not
  real climbing footage. Wide shots with several people in frame will be
  slower, and the detection rate reported by the script is meaningless here.
- The machine was noisy: run-to-run median inference varied by up to 3x. The
  benchmark now takes the fastest of N repeats and prints every repeat's median
  so the noise floor is visible. Re-run it on the target hardware.

Reproduce, with your own clip or a generated one:

```bash
# generate a clip that behaves like phone footage
.venv/bin/python scripts/make_sample_clip.py \
    --image person.jpg --out clip.mp4 --seconds 10 --rotate 90 --vfr

.venv/bin/python scripts/benchmark_pose.py \
    --video clip.mp4 --normalise --models lite,full,heavy --num-poses 5 \
    --repeats 3 --json results.json
```

## How it works

```
POST /videos            take the bytes, queue normalisation -> pending video
GET  /videos/{id}/candidates    pose on one frame -> numbered boxes + a JPEG
POST /videos/{id}/analyse       queue a job for the chosen box -> job id
GET  /jobs/{id}                 status and percent complete
GET  /videos/{id}/keypoints     the finished track, index-aligned with frames
```

Six decisions are load-bearing:

**Submitting an analysis freezes everything it depends on.** A run records the
seed box, the candidate index and frame it came from, the pose model, and the
tracking and refinement settings, at the moment it is queued. The worker reads
the run, never live settings or the current candidate set. Without this,
re-picking a person or changing an environment variable would silently alter a
job already in the queue, and the result would no longer describe anything you
could point at. Each run also owns its own keypoint file, so two runs over one
clip can be compared instead of overwriting each other.

**Candidates are stored, not recomputed.** The index a user picks only means
something against the exact detection run that produced it, so re-detecting
would renumber the options underneath someone mid-choice. The chosen box is
also what seeds tracking.

**The track is grown outwards from the seed frame.** The climber is picked in
the middle of the clip, but the track has to cover all of it, and where that
person stood at frame 0 is unknown. So the clip is decoded once, forwards:
frames from the seed onward are tracked as they are decoded, and detections
before the seed are buffered and tracked backwards afterwards. One inference
per frame in this pass; the refinement pass below adds a second. The buffer
holds the first part of the clip, which is what bounds how long a video this
handles comfortably.

**An unmatched frame is a gap, and stays one.** Nothing is interpolated across
it and the tracker never falls back to the nearest box. On a wall the wrong
person is often the closest one - a spotter directly below, a queue at the base
- so snapping would produce a track that looks plausible and is wrong. Gaps are
absent rows in the parquet file and nulls in the API response, and the overlay
draws nothing for them.

**Upload does not wait for ffmpeg.** Normalising a clip is a transcode of the
whole thing, and holding an HTTP request open for it invites a proxy timeout -
Cloudflare gives up on an origin after about 100 seconds. So the request only
takes delivery of the bytes and queues the work; the video is `pending` until
the worker has normalised it, and the client polls. ffmpeg reports its own
progress, so the wait has a number on it rather than a spinner.

**A job is committed before its id is queued.** The worker is another thread
with its own session; an id published inside an open transaction points at a
job that thread cannot see. This was a real bug, found by running the thing
rather than by testing it.

## Ingest

Every upload is normalised with ffmpeg before anything else touches it, and the
original is deleted. Three guarantees hold for every stored file:

| Guarantee | Why |
|-----------|-----|
| Constant 30 fps | Phone video is variable frame rate. Without this, `frame_index = round(time * fps)` is wrong, and the overlay drifts out of sync with the video. |
| Rotation baked into the pixels | OpenCV ignores the rotation flag in the container, the browser does not. Without this the analysed frames and the played-back frames disagree about which way is up. |
| Long edge capped at 1280 | Bounds inference cost. Raising it does not improve landmark quality - the crop the model gets is the constraint, not the frame. |

`normalise_video` verifies its own output — orientation, dimensions and frame
rate — and raises rather than storing a file that violates the contract.
Frame count comes from an actual decode pass, not the container header, because
the frontend maps playback time to a frame index against it.

What the source looked like (dimensions, rotation flag, whether it was variable
frame rate, codec) is recorded alongside the normalised metadata. Once
normalisation has happened, that evidence is otherwise unrecoverable, and it is
the first thing worth looking at when the overlay is misaligned.

## Layout

```
app/
  config.py            settings from the environment
  main.py              FastAPI app and lifespan
  geometry.py          bounding boxes and IoU
  frames.py            frame reading, shared by the picker and the worker
  candidates.py        detection on one frame, for the picker
  tracking.py          following one person; gaps
  keypoints.py         keypoint storage interface + parquet implementation
  analysis.py          the analysis job itself
  retention.py         deleting footage once it is no longer needed
  api/                 routes, response schemas, dependency wiring
  db/                  repositories: videos, jobs, candidates, analysis runs
  ingest/              ffprobe, ffmpeg normalisation, upload streaming
  jobs/                queue interface, worker thread, submission
  pose/                pose estimator interface, MediaPipe, crop refinement
scripts/
  benchmark_pose.py    times pose estimation over a clip
  make_sample_clip.py  builds a phone-like test clip from a photo
  download_models.py   fetches model files
static/index.html      markup and styles
static/app.js          the whole front end; no framework, no build step
```

Five boundaries are deliberate, because what sits behind them is expected to
change:

- **`app.pose.base.PoseEstimator`** — the pose model will be swapped.
  `create_pose_estimator()` is the only place naming MediaPipe. Coordinates are
  normalised 0-1, never pixels.
- **`app.tracking.Tracker`** — IoU is the simplest thing that could work.
- **`app.keypoints.KeypointStore`** — parquet today; long-format, so a model
  with a different number of joints still fits.
- **`app.jobs.base.JobQueue`** — one thread today, a broker later. It carries
  job ids and nothing else, which is what keeps that swap cheap.
- **`app.db.repository`** — routes and workers deal in `VideoRecord`,
  `JobRecord`, `CandidateSet` and `AnalysisRun`, never ORM objects.

`app.ingest` holds everything that knows about ffmpeg.

## API

| Endpoint | Purpose |
|----------|---------|
| `POST /videos` | Upload, normalise, return metadata |
| `GET /videos` | List uploaded videos |
| `GET /videos/{id}` | Metadata for one video |
| `GET /videos/{id}/file` | Serve the normalised file (range requests, so it seeks) |
| `GET /videos/{id}/candidates` | People detected in one frame, plus a JPEG of it |
| `GET /videos/{id}/candidates/frame.jpg` | That frame, to draw the boxes on |
| `POST /videos/{id}/analyse` | Queue analysis of a chosen candidate |
| `GET /jobs/{id}` | Job status and percent complete |
| `GET /videos/{id}/jobs` | Jobs for one video, newest first |
| `GET /videos/{id}/keypoints` | A finished track — latest, or `?analysis_run_id=` |
| `DELETE /videos/{id}/footage` | Delete the clip now, keep the keypoints |
| `GET /healthz` | Liveness |

Upload returns as soon as the bytes are on disk; normalisation is queued, and
the video stays `pending` until a worker has finished with it. Candidate
detection costs a second or two, and is cached after the first call. Analysis
is queued and polled.

`/keypoints` serves the video's most recently completed track by default. Pass
`?analysis_run_id=` to ask for a specific run's, which is how two runs over the
same clip can be compared.

`/keypoints` returns `frames` index-aligned with the video — entry N is frame
N, `null` is a gap — plus `match_iou` per frame, the tracker's confidence in
that frame's match. Coordinates are normalised 0-1, and can fall slightly
outside that range where the model extrapolates an occluded joint.

## Footage retention

The clip is the bulky, sensitive part, and it stops being useful once the track
has been extracted. So footage is deleted and the keypoints are kept - they are
small, they are what the analysis was for, and they are not video of anybody.

Two windows, because the cases differ. An **analysed** video is counted from
when its analysis finished; the delay exists only so the overlay has something
to draw on while somebody reviews the result. Set
`CLIMB_RETAIN_ANALYSED_SECONDS=0` to delete the moment a job completes, and
accept that there is then nothing to play back. An **unanalysed** upload is
abandoned, and counted from when it arrived.

A janitor thread sweeps on a timer rather than checking on access: the promise
is that footage is deleted, not that it is hidden from whoever asks next.
Nobody may ever ask again, and it still has to go. `DELETE
/videos/{id}/footage` removes a clip immediately.

The same janitor sweeps orphaned keypoint files — artifacts no run and no
video refers to any more, left behind by a job that died mid-write or by rows
since deleted. Anything still referenced is never touched, and a one-hour grace
period covers the window between a worker writing its file and recording the
path.

Once footage is gone, `GET /videos/{id}/file` returns 410 and the row reports
`has_footage: false`. Everything else about the video still works - the
keypoints, the coverage chart, the stats. The analysis outlives the footage,
which is the point.

## Jobs

One background thread, pulling ids off an in-process queue, running jobs one at
a time. Job state lives in SQLite, not in the queue, so a page refresh sees the
truth. `JobQueue` is three methods; swapping in a broker means writing another
implementation of it and changing what `build_queue` returns.

Jobs run one at a time because pose estimation already saturates the CPU — a
second worker would make both slower rather than finishing anything sooner.

On startup, jobs left `running` by a previous process are marked failed (the
process that owned them is gone, and leaving the row at `running` means a
status that never changes again) and jobs left `queued` are put back on the
queue.

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLIMB_DATA_DIR` | `./data` | Videos, keypoints, SQLite database |
| `CLIMB_MODEL_DIR` | `./models` | Pose model files |
| `CLIMB_DATABASE_URL` | `sqlite:///<data>/climb.db` | |
| `CLIMB_TARGET_FPS` | `30` | Normalisation frame rate |
| `CLIMB_MAX_LONG_EDGE` | `1280` | Normalisation size cap |
| `CLIMB_MAX_UPLOAD_BYTES` | `52428800` | Upload limit (50 MB) |
| `CLIMB_RETAIN_ANALYSED_SECONDS` | `3600` | Keep footage this long after analysis |
| `CLIMB_RETAIN_UNANALYSED_SECONDS` | `86400` | Keep never-analysed uploads this long |
| `CLIMB_RETENTION_SWEEP_SECONDS` | `60` | How often the janitor looks |
| `CLIMB_POSE_MODEL` | `lite` | `lite`, `full` or `heavy` |
| `CLIMB_MAX_PEOPLE` | `5` | Maximum people detected per frame |
| `CLIMB_REFINE_LANDMARKS` | `1` | Second pose pass on a crop around the tracked person |
| `CLIMB_REFINE_MARGIN` | `0.55` | How much context to leave around that crop |
| `CLIMB_FFMPEG` / `CLIMB_FFPROBE` | `ffmpeg` / `ffprobe` | Binary paths |

## What it does on real footage

Measured on a 31-second clip from a bouldering gym: 1080x1920 portrait,
variable frame rate, climber 10-26% of frame height, and somebody walking
through the foreground mid-climb.

| Phase | Frames | Tracked |
|-------|--------|---------|
| approach, not yet on the wall | 0-65 | 47% |
| climbing | 66-405 | 92% |
| person walks through the shot | 406-472 | 31% |
| climbing | 473-749 | 100% |
| coming off the wall | 750-822 | 44% |
| climber has left frame | 823-929 | 0% |
| **the climb itself** | **66-749** | **89%** |

Median match IoU 0.84. **Zero frames tracked the wrong person** — the walker
who fills two thirds of the frame was never picked up, and the track resumed
on the climber afterwards. That was the rule worth testing, and it held.

The gaps are honest ones: the climber genuinely is occluded, out of shot, or
moving too fast to detect. Nothing was invented to fill them.

Model choice barely matters here — lite, full and heavy each found the climber
in 12-13 of 16 sampled frames — so there is no reason to pay for a bigger one.

## Jitter, and what actually fixes it

The first build's skeleton visibly wobbled. The cause was not model noise,
which is what reaching for a smoothing filter would have assumed: on this
footage the climber is a **median of 150 pixels tall**, and MediaPipe's
landmark model works on a 256x256 crop of its subject, so it was upscaling
from far less detail than it wants. The signature says the same - jitter was
worst on feet, ankles and hands, lowest on ears and eyes.

Measured on the gym clip, as median frame-to-frame landmark acceleration
expressed as a fraction of body height:

| | median | p90 | p99 |
|---|---|---|---|
| lite, single pass | 0.0762 | 0.542 | 1.686 |
| lite + crop refinement | 0.0381 | 0.522 | 2.155 |
| **heavy + crop refinement** | **0.0172** | **0.168** | **1.317** |
| heavy + refinement + display smoothing | 0.0026 | 0.022 | 0.190 |

Three findings worth keeping:

**Raising the resolution cap does nothing.** 1280 to 1920 moved it from 0.0514
to 0.0504 on a test segment. The frame is not the constraint; the crop the
model gets is.

**Cropping to the tracked person halves it, and is free.** The model is
slightly faster on a smaller image, so the second pass costs about what the
first did. See `app/pose/refine.py`.

**heavy is worth its two-fold cost on this material.** Its raw output is
steadier than lite's, it fixes the p99 that refinement alone made worse, and
the two passes disagree on 30 frames rather than 81. `docker-compose.yml`
defaults to it; the code default stays `lite` for constrained hosts.

Smoothing is a One Euro filter applied **at display time only**, with a toggle.
The stored keypoints stay raw, so the filtered and unfiltered tracks can still
be compared - which was the whole reason for not shipping a filter until the
raw behaviour had been seen.

## What this still needs

**Thresholds are still guesses.** `TrackingConfig.min_iou` is 0.3 and gaps
never expire, so the tracker keeps trying to re-acquire indefinitely. One clip
is not enough to tune that. Footage with two climbers on the same wall is the
next useful input, since that is the case where re-acquiring the wrong person
is actually possible.

**Runs are recorded but not browsable.** Every analysis writes an immutable
run, and the API can serve any of them, but the front end only ever shows the
newest. Listing them is the obvious next piece of UI.

Known rough edges:

- The schema has no migrations. Changing a model means deleting
  `data/climb.db`; the data is disposable at this stage.
- Analysis buffers the detections before the seed frame in memory, which is
  what limits practical clip length. Fine for a few minutes.
- Uploads and analysis are unauthenticated and unbounded per user.

## Notes

- No authentication, single container, CPU only.
- Audio is stripped during normalisation; nothing here uses it.
- The Docker image is built and smoke-tested in CI rather than locally. Its
  system dependencies are `ffmpeg`, `libegl1`, `libgl1`, `libgles2` and
  `libglib2.0-0`: MediaPipe 1.0.1 fails at import without the GL libraries even
  though it runs on CPU.
