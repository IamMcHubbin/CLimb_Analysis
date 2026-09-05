# Running it on your own machine, behind Cloudflare

The shape that fits: the app runs in Docker on your PC, `cloudflared` dials out
to Cloudflare, and Cloudflare handles TLS and who is allowed in. Your router
never has a port open.

```
phone at the gym  →  climb.example.com
                  →  Cloudflare  (TLS, Access)
                  →  cloudflared tunnel  (outbound only)
                  →  Docker on your PC
```

## What Cloudflare does and does not do here

Worth stating plainly, because it is a common assumption: **Cloudflare cannot
run the pose estimation.** Workers are JS/WASM with hard CPU-time limits and no
filesystem - no ffmpeg, no MediaPipe. Cloudflare Stream transcodes and hosts
video but does no pose estimation, and it would keep your footage on their
servers, which is the opposite of what the retention policy here is for.

Your CPU does the analysis. Cloudflare's job is getting traffic to it safely,
and for that it is genuinely the right tool.

## Hardware

Analysis is CPU-bound and single-job-at-a-time. A modern 6-core desktop chip is
comfortably enough - expect a few times the throughput of the figures in the
README, which were measured on a slow 4-core cloud VM.

**A consumer GPU will not help.** MediaPipe's Python Tasks API has no usable
AMD path, and its GPU delegate is limited to specific platforms; an RX-series
card will sit idle. It can accelerate ffmpeg decode and scaling during
normalisation (VAAPI on Linux, AMF on Windows), but normalisation is a few
seconds per clip, so the win is small.

Measure yours rather than trusting an estimate:

```bash
python scripts/benchmark_pose.py --video your_clip.mp4 --normalise \
    --models lite,full --num-poses 5 --repeats 3
```

Linux gets more out of the same chip than Docker Desktop on Windows, which
runs through a WSL2 VM.

## Running the app

Same commands on every OS - run them from PowerShell, Command Prompt, a WSL2
shell, or a Linux/macOS terminal, whichever Docker Desktop (or Docker Engine)
is already using:

```bash
git clone https://github.com/IamMcHubbin/CLimb_Analysis.git
cd CLimb_Analysis
docker compose up -d --build
```

**Windows only:** open Docker Desktop's **Settings → General** and enable
**"Start Docker Desktop when you log in."** Without it, nothing in this doc
restarts automatically after a reboot, because the Docker daemon on Windows
only exists while Docker Desktop is running - see "Reboots and updates" below.

That serves on `http://localhost:8000`, storing everything under `./data`.

Settings worth changing for a personal instance, in `docker-compose.yml`:

```yaml
environment:
  # An hour suits a shared service. On your own machine you probably want to
  # keep clips long enough to come back to them.
  CLIMB_RETAIN_ANALYSED_SECONDS: "604800"    # a week
  # Raise the upload cap - but see the Cloudflare limit below before going
  # past 100MB.
  CLIMB_MAX_UPLOAD_BYTES: "104857600"        # 100MB
```

## The tunnel

Pick your OS below - each block is the complete sequence for that OS, so you
only need to follow one of them top to bottom.

### Windows

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create climb
cloudflared tunnel route dns climb climb.example.com
```

`cloudflared` already created a `.cloudflared` folder in your home directory
during `tunnel login`. Put the config there:
`%USERPROFILE%\.cloudflared\config.yml` (e.g.
`C:\Users\you\.cloudflared\config.yml`).

```yaml
tunnel: climb
credentials-file: C:/Users/you/.cloudflared/<tunnel-id>.json   # forward slashes are fine
ingress:
  - hostname: climb.example.com
    service: http://localhost:8000
    originRequest:
      # Normalisation and analysis are queued, so requests are short - but a
      # 100MB upload over a slow phone connection is not.
      connectTimeout: 30s
  - service: http_status:404
```

Run it by hand with `cloudflared tunnel run climb`, or install it as a
service so it comes back after a reboot - from an **elevated** (Administrator)
PowerShell or Command Prompt:

```powershell
cloudflared service install
```

This registers a service named `cloudflared`, visible in `services.msc`;
`cloudflared service uninstall` (also elevated) removes it.

### Linux / macOS

```bash
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
sudo install cloudflared /usr/local/bin/cloudflared
cloudflared tunnel login
cloudflared tunnel create climb
cloudflared tunnel route dns climb climb.example.com
```

`cloudflared` already created a `.cloudflared` folder in your home directory
during `tunnel login`. Put the config there: `~/.cloudflared/config.yml`.

```yaml
tunnel: climb
credentials-file: /home/you/.cloudflared/<tunnel-id>.json
ingress:
  - hostname: climb.example.com
    service: http://localhost:8000
    originRequest:
      # Normalisation and analysis are queued, so requests are short - but a
      # 100MB upload over a slow phone connection is not.
      connectTimeout: 30s
  - service: http_status:404
```

Run it by hand with `cloudflared tunnel run climb`, or install it as a
service so it comes back after a reboot:

```bash
sudo cloudflared service install
```

## Locking it down

**Do this before the hostname exists.** The app has no authentication of its
own: anyone who finds the URL can upload clips and spend your CPU on them.

Cloudflare Access is the least-effort fix and is free for small numbers of
users. In the Zero Trust dashboard: **Access → Applications → Add**, self-hosted,
pointed at `climb.example.com`, with a policy allowing your own email address
(and anyone you want to share it with). Cloudflare then demands a login before
any request reaches the tunnel.

## Two limits to know

**Cloudflare caps request bodies at 100MB** on Free and Pro plans. That is your
real upload ceiling regardless of what you set `CLIMB_MAX_UPLOAD_BYTES` to, and
it is worth knowing what phone footage actually weighs:

| Recording | Roughly |
|---|---|
| 1080p30 | 5-10 MB per 30s |
| 1080p60 | ~20 MB per 30s |
| 4K60 | ~100 MB per 30s |

Record at 1080p and 50MB is plenty for a couple of minutes. Trim rather than
re-encode if you need to cut a file down - re-encoding strips the variable
frame timing and rotation flags this pipeline exists to handle:

```bash
ffmpeg -ss 0 -t 30 -i IMG_1234.MOV -c copy clip.MOV
```

**Cloudflare gives up on an origin after about 100 seconds.** This is why
upload no longer waits for normalisation: the request returns as soon as the
bytes are on disk, and ffmpeg runs on the worker afterwards. Nothing in the
request path now takes minutes, so this limit should not be reachable. If you
add anything that does, queue it rather than doing it inline.

## Reboots and updates

`docker-compose.yml` sets `restart: unless-stopped`, and `cloudflared service
install` handles the tunnel, so the machine coming back up is enough to bring
the app back.

**Windows only:** that restart policy only takes effect once the Docker
daemon is running again, and the daemon lives inside Docker Desktop, which
does not start on its own unless you enabled "Start Docker Desktop when you
log in" (see "Running the app," above). Without that setting, the container
simply stays down after a reboot until you notice and open Docker Desktop by
hand.

To update:

```bash
git pull && docker compose up -d --build
```

`./data` survives. The schema does not migrate, so if a model changes, delete
`data/climb.db` - the videos and keypoints are separate files and the app will
simply not know about the old ones.
