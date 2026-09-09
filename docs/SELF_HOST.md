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

**Windows prerequisite:** Docker Desktop needs the current Store-serviced WSL2,
not the older Windows inbox command. Run `wsl --version` first. If that prints
the `wsl.exe` usage text instead of version numbers, run this from an elevated
PowerShell and reboot before starting Docker Desktop:

```powershell
wsl --install --no-distribution --web-download
```

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

## Temporary testing without a domain

A Quick Tunnel is useful for an occasional remote test before the app warrants
a domain. It creates a new random `*.trycloudflare.com` URL each time and exists
only while the command is running:

```powershell
# Windows: install once, then run the tunnel when it is needed
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

```bash
# Linux/macOS, once cloudflared is installed
cloudflared tunnel --url http://localhost:8000
```

The URL is public and a Quick Tunnel has no authentication. Anyone who obtains
it can use the app and see its retained videos, so use only non-sensitive test
footage and do not leave it running. Press **Ctrl+C** in that terminal to shut
off public access. Run the same command again to restart it; Cloudflare will
issue a different URL.

Quick Tunnels are for testing, have no uptime guarantee, and should not be
installed as a service. Use the named tunnel and Access setup below when the app
needs a stable, authenticated address.

## The tunnel

For a permanent authenticated address, pick your OS below. Each block is the
complete sequence for that OS, so you only need to follow one of them top to
bottom.

### Windows

```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel login
cloudflared tunnel create climb
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

Only after that Access application and its Allow policy are saved should the
hostname be published:

```bash
cloudflared tunnel route dns climb climb.example.com
```

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

Run these from the repository directory to control the local app:

```bash
docker compose stop                 # stop the app; keep its container and data
docker compose start                # start it again without rebuilding
docker compose restart              # restart the running container
docker compose up -d --build        # build changes and leave it running
docker compose down                 # remove the container; ./data still survives
```

On Windows, double-click the included `climb-server.bat` to open a Start,
Stop, Restart and Status menu. It also accepts command-line arguments:

```bat
climb-server.bat start
climb-server.bat stop
climb-server.bat restart
climb-server.bat status
climb-server.bat local
```

`start` builds any changed image, starts the app, opens a Cloudflare Quick
Tunnel, and **prints both addresses**:

```text
========================================
  On this machine:  http://localhost:8000
  From anywhere:    https://mounted-brave-tommy-strain.trycloudflare.com
========================================
```

The public address is randomly generated every time the tunnel starts and is
forgotten when it stops, so reading it out of cloudflared's log is the only
way to know it - which is what the launcher does. `restart` and `status`
reprint it, so clearing the window does not lose it.

`stop` closes the tunnel as well as the container, and keeps `./data`. Use
`local` to start without opening a public link at all.

The tunnel needs `cloudflared` installed (`winget install --id
Cloudflare.cloudflared`) but needs no domain or Cloudflare account. If it is
missing, `start` still starts the app and simply says there is no public link
rather than failing. The link is **unauthenticated**: the app has no login, so
anyone who has it can upload clips and spend your CPU. Prefer it for short
sessions rather than leaving it up. For a stable address with a login in front
of it, use the named tunnel and Cloudflare Access described above instead.

To make the experimental ViTPose model selectable on this machine, copy
`.env.example` to `.env` and set `INSTALL_VITPOSE=true` before choosing Start.
That local file is ignored by Git. Leave it false for the smaller MediaPipe-only
image.

For an interactive Quick Tunnel, **Ctrl+C** stops public access without stopping
Docker. Restart it with `cloudflared tunnel --url http://localhost:8000` and
share the newly generated URL. A permanent Windows tunnel installed as a
service can instead be controlled from an elevated PowerShell:

```powershell
Stop-Service cloudflared
Start-Service cloudflared
Restart-Service cloudflared
```

To update:

```bash
git pull && docker compose up -d --build
```

`./data` survives. The schema does not migrate, so if a model changes, delete
`data/climb.db` - the videos and keypoints are separate files and the app will
simply not know about the old ones.
