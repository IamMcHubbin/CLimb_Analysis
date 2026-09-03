# Climb Analysis

Proof of concept: upload a climbing video, pick the climber, run 2D pose
estimation over the clip, play it back with a skeleton overlay. The question
this is meant to answer is whether tracking quality holds up on real climbing
footage — not to produce metrics or coaching feedback.

**Current state: the loop closes.** Upload a clip, pick a person out of a
mid-clip frame, watch the job run, then scrub the video with the skeleton drawn
on top. No smoothing, no metrics, no hold detection.

---

## Quick start

Local, without Docker (needs `ffmpeg` and `ffprobe` on `PATH`):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/download_models.py lite full
.venv/bin/uvicorn app.main:app --reload
```

Then open <http://localhost:8000> to upload a clip and see what normalisation
did to it.

With Docker:

```bash
docker compose up --build
```

Run the tests:

```bash
.venv/bin/python -m pytest tests -q
```

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
worker analyses a clip faster than real time, so a two-minute climb finishes in
about a minute and a half. `heavy` does not keep up and would need the long
edge dropped below 1280 to be practical. Multi-person detection roughly doubles
the per-frame cost even when only one person is in frame, because the detector
runs more often.

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
POST /videos            ffmpeg normalise -> row in SQLite
GET  /videos/{id}/candidates    pose on one frame -> numbered boxes + a JPEG
POST /videos/{id}/analyse       queue a job for the chosen box -> job id
GET  /jobs/{id}                 status and percent complete
GET  /videos/{id}/keypoints     the finished track, index-aligned with frames
```

Four decisions are load-bearing:

**Candidates are stored, not recomputed.** The index a user picks only means
something against the exact detection run that produced it, so re-detecting
would renumber the options underneath someone mid-choice. The chosen box is
also what seeds tracking.

**The track is grown outwards from the seed frame.** The climber is picked in
the middle of the clip, but the track has to cover all of it, and where that
person stood at frame 0 is unknown. So the clip is decoded once, forwards:
frames from the seed onward are tracked as they are decoded, and detections
before the seed are buffered and tracked backwards afterwards. Every frame gets
exactly one pose inference. The buffer holds the first part of the clip, which
is what bounds how long a video this handles comfortably.

**An unmatched frame is a gap, and stays one.** Nothing is interpolated across
it and the tracker never falls back to the nearest box. On a wall the wrong
person is often the closest one - a spotter directly below, a queue at the base
- so snapping would produce a track that looks plausible and is wrong. Gaps are
absent rows in the parquet file and nulls in the API response, and the overlay
draws nothing for them.

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
| Long edge capped at 1280 | Bounds inference cost. |

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
  main.py              FastAPI app
  api/                 routes, response schemas, dependency wiring
  db/                  repository interface + SQLAlchemy implementation
  ingest/              ffprobe, ffmpeg normalisation, upload streaming
  pose/                pose estimator interface + MediaPipe implementation
scripts/
  benchmark_pose.py    times pose estimation over a clip
  make_sample_clip.py  builds a phone-like test clip from a photo
  download_models.py   fetches model files
static/index.html      upload page (placeholder for the real UI)
```

Three boundaries are deliberate, because what sits behind them is expected to
change:

- **`app.pose.base.PoseEstimator`** — the pose model will be swapped.
  `create_pose_estimator()` is the only place naming MediaPipe. Coordinates are
  normalised 0-1, never pixels.
- **`app.db.repository.VideoRepository`** — routes and workers deal in
  `VideoRecord`, never ORM objects, so the backing store can change without
  touching them.
- **`app.ingest`** — everything about ffmpeg lives here.

## API

Implemented:

| Endpoint | Purpose |
|----------|---------|
| `POST /videos` | Upload, normalise, return metadata |
| `GET /videos` | List uploaded videos |
| `GET /videos/{id}` | Metadata for one video |
| `GET /videos/{id}/file` | Serve the normalised file (range requests, so it seeks) |
| `GET /healthz` | Liveness |

Upload normalisation is synchronous, so a long clip holds the request open. It
is deliberate: nothing can be done with a video until it is normalised, and
deferring it would mean a video row that exists but is not yet usable.

Planned:

| Endpoint | Purpose |
|----------|---------|
| `GET /videos/{id}/candidates` | Detected people in a mid-clip frame, plus a JPEG of it |
| `POST /videos/{id}/analyse` | Enqueue analysis for a chosen candidate |
| `GET /jobs/{id}` | Job status and percent complete |
| `GET /videos/{id}/keypoints` | Finished keypoint data |

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLIMB_DATA_DIR` | `./data` | Videos, keypoints, SQLite database |
| `CLIMB_MODEL_DIR` | `./models` | Pose model files |
| `CLIMB_DATABASE_URL` | `sqlite:///<data>/climb.db` | |
| `CLIMB_TARGET_FPS` | `30` | Normalisation frame rate |
| `CLIMB_MAX_LONG_EDGE` | `1280` | Normalisation size cap |
| `CLIMB_MAX_UPLOAD_BYTES` | `1073741824` | Upload limit |
| `CLIMB_POSE_MODEL` | `lite` | `lite`, `full` or `heavy` |
| `CLIMB_MAX_PEOPLE` | `5` | Maximum people detected per frame |
| `CLIMB_FFMPEG` / `CLIMB_FFPROBE` | `ffmpeg` / `ffprobe` | Binary paths |

## Next

1. `GET /videos/{id}/candidates` — pose detection on one mid-clip frame,
   returning normalised bounding boxes and a JPEG to draw them on.
2. Job queue: a single background thread behind a small interface, job state in
   SQLite so status survives a refresh, percent complete updated as frames are
   processed.
3. IoU tracking of the chosen candidate, with unmatched frames recorded as gaps
   rather than snapped to whoever else is nearby.
4. Keypoint storage, one parquet file per video, path in the database row.
5. The overlay: `<canvas>` over `<video>`, frame index from `currentTime` on
   each `requestAnimationFrame`, gaps drawn as nothing.

No smoothing filter — the point is to see the raw jitter first.

## Notes

- No authentication, single container, CPU only.
- Audio is stripped during normalisation; nothing here uses it.
- The Docker image has not been built and run in this environment (no Docker
  daemon available), though its system dependencies match what the application
  was verified against here: `ffmpeg`, `libegl1`, `libgl1`, `libgles2`,
  `libglib2.0-0`. MediaPipe 1.0.1 fails at import without the GL libraries even
  though it runs on CPU.
