# Deploying to Fly.io

The repository is ready to deploy; `fly.toml` is committed and the Dockerfile
listens on `$PORT`. What follows is the whole procedure.

## Before you start: this app has no authentication

Deploying it with a public address puts an **open upload endpoint that runs
minutes of CPU per request** on the internet. Anyone who finds the URL can
queue work on your machine and store video on your volume. That is fine for a
private trial and not fine for a link you paste anywhere.

Two ways to handle it. Pick one before allocating a public IP.

**Private only (no code, recommended for a trial).** Skip the public address
entirely and reach the app over Fly's WireGuard network:

```bash
fly deploy --ha=false            # do NOT run `fly ips allocate-*`
fly proxy 8080:8080              # then open http://localhost:8080
```

The app is then reachable only by you, through the tunnel.

**Public with a front door.** Put Cloudflare Access, Tailscale, or an
authenticating proxy in front of it. Adding real auth to the app itself is a
change nobody has made yet - it is listed under "what this still needs".

## First deploy

```bash
# 1. Install flyctl and sign in
#    macOS/Linux:
curl -L https://fly.io/install.sh | sh
#    Windows (PowerShell):
#    powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"
fly auth login

# 2. Claim an app name. Edit `app = ` in fly.toml first if you want a
#    different one; names are global, so "climb-analysis" may be taken.
fly apps create climb-analysis

# 3. Create the volume the SQLite database and parquet files live on.
#    Must be in the same region as primary_region in fly.toml.
fly volumes create climb_data --size 3 --region lhr

# 4. Deploy. The first build is slow - the image installs MediaPipe, OpenCV
#    and ffmpeg, and bakes the pose models in, so expect ~1.5GB and several
#    minutes. Later deploys reuse the layers.
fly deploy --ha=false
```

`--ha=false` matters: without it Fly creates two machines, and the second one
cannot attach the volume.

## Checking it

```bash
fly status
fly logs
fly ssh console -C "df -h /data"     # volume mounted and sized
```

Then open the app (through `fly proxy`, or its public URL if you allocated
one) and upload a clip.

## Why the config looks the way it does

| Setting | Reason |
|---|---|
| `auto_stop_machines = false` | The analysis worker is a thread inside the web process. Fly's default stops a machine when traffic goes quiet - which is exactly when a long job is still running, and it would be killed. |
| `min_machines_running = 1` | Same reason from the other side: something must stay up to finish queued work. |
| `--ha=false`, never scale out | State is SQLite plus files on one volume, and the job queue is in-process. A volume attaches to one machine. Two machines would mean two queues and two databases disagreeing. |
| `shared-cpu-2x`, 2GB | Pose estimation is CPU-bound, and analysis buffers detections before the seed frame in memory. `shared-cpu-1x`/1GB is enough for short clips; this size is so a few minutes of footage is not OOM-killed halfway. |
| `PORT = "8080"` | Fly's convention. The Dockerfile falls back to 8000 locally. |
| No request-body limit | Fly does not impose one, so 50MB uploads work. Cloud Run's 32 MiB limit is why it does not suit this app. |

## Cost and sizing

`shared-cpu-2x` with 2GB and a 3GB volume runs roughly $12-15/month at the
time of writing. Two ways down:

- Drop to `shared-cpu-1x` / 1GB (about half) if your clips are short. Watch
  `fly logs` for OOM kills after a long analysis.
- Shrink the volume. Retention is what makes this safe: footage is deleted an
  hour after analysis, so storage stays roughly flat instead of growing with
  every upload. A 31-second clip leaves a 260KB parquet behind rather than 9MB
  of video. 3GB is generous.

Note that sustained CPU on a *shared* Fly VM is throttled after a burst. If
analysis is slower on Fly than the benchmark in the README suggests, that is
why; `performance-1x` gives a dedicated core for more money.

## Updating

```bash
fly deploy --ha=false
```

The volume survives deploys. The schema does not migrate, though - see the
README - so if a model changes you will need to remove the database:

```bash
fly ssh console -C "rm -f /data/climb.db /data/climb.db-wal /data/climb.db-shm"
```

## Configuration

Anything in the README's configuration table can be set with `fly secrets set`
or in the `[env]` block of `fly.toml`. The one most worth revisiting:

```bash
# Delete footage the moment analysis finishes, rather than an hour later.
# There is then nothing to play back - the keypoints and charts remain.
fly secrets set CLIMB_RETAIN_ANALYSED_SECONDS=0
```
