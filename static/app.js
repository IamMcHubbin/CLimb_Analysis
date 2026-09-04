'use strict';

/* Climb Analysis front end.
 *
 * One page, four steps: upload, pick a person, wait, look at the result.
 * No framework and no build step, so what is here is what runs.
 */

const $ = (id) => document.getElementById(id);
const CSS = getComputedStyle(document.documentElement);
const COLOUR = {
  track: CSS.getPropertyValue('--track').trim(),
  gap: CSS.getPropertyValue('--gap').trim(),
  line: CSS.getPropertyValue('--line').trim(),
  muted: CSS.getPropertyValue('--muted').trim(),
  accent: CSS.getPropertyValue('--accent').trim(),
};

const state = {
  config: null,
  video: null,       // the video record being worked on
  candidates: null,
  selected: null,
  keypoints: null,
  poll: null,
  retentionTick: null,
};

// ------------------------------------------------------------------ helpers

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = response.status === 204 ? null : await response.json();
  if (!response.ok) {
    throw new Error((payload && payload.detail) || `${path} failed (${response.status})`);
  }
  return payload;
}

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle('error', isError);
}

const show = (id, visible = true) => { $(id).hidden = !visible; };

function formatBytes(bytes) {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} kB`;
}

function formatDuration(seconds) {
  if (seconds <= 0) return 'any moment';
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

// ------------------------------------------------------------------- config

async function loadConfig() {
  state.config = await api('/config');
  $('drop-hint').textContent =
    `MP4, MOV or similar, up to ${formatBytes(state.config.max_upload_bytes)}. ` +
    `Normalised to ${state.config.target_fps}fps, long edge ${state.config.max_long_edge}px.`;
}

// ------------------------------------------------------------------- upload

const drop = $('drop');
drop.addEventListener('click', () => $('file').click());
drop.addEventListener('dragover', (event) => {
  event.preventDefault();
  drop.classList.add('over');
});
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (event) => {
  event.preventDefault();
  drop.classList.remove('over');
  if (event.dataTransfer.files.length) upload(event.dataTransfer.files[0]);
});
$('file').addEventListener('change', (event) => {
  if (event.target.files.length) upload(event.target.files[0]);
});

function upload(file) {
  const limit = state.config ? state.config.max_upload_bytes : Infinity;
  if (file.size > limit) {
    // Checked here as well as on the server: no reason to push 200MB up the
    // wire only to be told no at the far end.
    setStatus('upload-status',
      `${file.name} is ${formatBytes(file.size)}; the limit is ${formatBytes(limit)}. ` +
      `Trim it rather than re-encoding — re-encoding strips the frame timing and ` +
      `rotation flags this pipeline exists to handle.`, true);
    return;
  }

  ['s-pick', 's-progress', 's-play'].forEach((id) => show(id, false));
  const bar = $('up-bar');
  bar.hidden = false;
  bar.firstElementChild.style.width = '0%';
  setStatus('upload-status', `Uploading ${file.name}…`);

  const body = new FormData();
  body.append('file', file);

  // XHR rather than fetch: fetch cannot report upload progress, and a 50MB
  // clip on a slow connection is a long time to show nothing.
  const request = new XMLHttpRequest();
  request.open('POST', '/videos');
  request.upload.addEventListener('progress', (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((event.loaded / event.total) * 100);
    bar.firstElementChild.style.width = `${percent}%`;
    if (percent >= 100) setStatus('upload-status', 'Uploaded. Waiting for normalisation…');
  });
  request.addEventListener('load', async () => {
    bar.hidden = true;
    let payload;
    try {
      payload = JSON.parse(request.responseText);
    } catch {
      setStatus('upload-status', `Upload failed (${request.status})`, true);
      return;
    }
    if (request.status !== 202) {
      setStatus('upload-status', payload.detail || `Upload failed (${request.status})`, true);
      return;
    }
    // The bytes are in; ffmpeg has not run yet. The server answers straight
    // away so no proxy times out waiting on a transcode, which means the
    // client has to wait for "ready" itself.
    await waitForNormalisation(payload);
    refreshLibrary();
  });
  request.addEventListener('error', () => {
    bar.hidden = true;
    setStatus('upload-status', 'Upload failed: the connection dropped', true);
  });
  request.send(body);
}

async function waitForNormalisation(video) {
  const bar = $('up-bar');
  bar.hidden = false;
  bar.firstElementChild.style.width = '0%';
  setStatus('upload-status', 'Normalising — constant frame rate, rotation baked in…');

  while (true) {
    let current;
    try {
      current = await api(`/videos/${video.id}`);
    } catch (error) {
      setStatus('upload-status', error.message, true);
      bar.hidden = true;
      return;
    }
    if (current.status === 'ready') {
      bar.hidden = true;
      setStatus('upload-status', `Ready — ${current.frame_count} frames at ${current.fps}fps`);
      await openVideo(current);
      return;
    }
    if (current.status === 'failed') {
      bar.hidden = true;
      setStatus('upload-status', current.ingest_error || 'Normalisation failed', true);
      return;
    }
    // Progress comes from the ingest job, which reads it out of ffmpeg.
    const job = await ingestJobFor(video.id);
    if (job) {
      const percent = Math.round(job.progress * 100);
      bar.firstElementChild.style.width = `${percent}%`;
      setStatus('upload-status', `Normalising — ${percent}%`);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

async function ingestJobFor(videoId) {
  try {
    const jobs = await api(`/videos/${videoId}/jobs`);
    return jobs.find((job) => job.kind === 'ingest') || null;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ library

async function refreshLibrary() {
  let videos;
  try {
    videos = await api('/videos?limit=8');
  } catch {
    return;
  }
  show('s-library', videos.length > 0);
  $('lib').innerHTML = '';
  for (const video of videos) {
    const row = document.createElement('div');
    row.className = 'lib-item';
    const tags = [];
    if (video.status === 'pending') tags.push('<span class="tag">normalising…</span>');
    if (video.status === 'failed') tags.push('<span class="tag bad">failed</span>');
    if (video.duration_seconds) {
      tags.push(`<span class="tag">${video.duration_seconds.toFixed(0)}s</span>`);
    }
    if (video.has_keypoints) tags.push('<span class="tag ok">analysed</span>');
    if (!video.has_footage) tags.push('<span class="tag gone">footage deleted</span>');
    row.innerHTML =
      `<span class="name" title="${video.original_filename}">${video.original_filename}</span>` +
      `<span class="tags">${tags.join('')}</span>`;
    row.addEventListener('click', () => openVideo(video));
    $('lib').appendChild(row);
  }
}

// --------------------------------------------------------- opening a video

async function openVideo(video) {
  state.video = video;
  state.keypoints = null;
  clearInterval(state.poll);
  ['s-pick', 's-progress', 's-play'].forEach((id) => show(id, false));

  if (video.status === 'pending') {
    await waitForNormalisation(video);
    return;
  }
  if (video.status === 'failed') {
    setStatus('upload-status', video.ingest_error || 'This upload could not be normalised', true);
    return;
  }
  if (video.has_keypoints) {
    await showResult();
    return;
  }
  if (!video.has_footage) {
    setStatus('upload-status',
      'That clip was never analysed and its footage has been deleted.', true);
    return;
  }
  await loadCandidates();
}

// --------------------------------------------------------------- candidates

async function loadCandidates(frameIndex) {
  show('s-pick', true);
  setStatus('pick-status', 'Detecting people…');
  const query = frameIndex === undefined ? '' : `?frame_index=${frameIndex}&refresh=true`;
  try {
    state.candidates = await api(`/videos/${state.video.id}/candidates${query}`);
  } catch (error) {
    setStatus('pick-status', error.message, true);
    return;
  }
  state.selected = null;
  $('analyse').disabled = true;
  renderCandidates();
}

function renderCandidates() {
  const data = state.candidates;
  $('cand-frame').src = `${data.frame_url}?t=${Date.now()}`;

  const wrap = $('frame-wrap');
  wrap.querySelectorAll('.cand').forEach((el) => el.remove());
  data.candidates.forEach((candidate) => {
    const box = candidate.bounding_box;
    const el = document.createElement('div');
    el.className = 'cand';
    // Percentages, so the boxes stay aligned however the image is scaled.
    el.style.left = `${box.x * 100}%`;
    el.style.top = `${box.y * 100}%`;
    el.style.width = `${box.width * 100}%`;
    el.style.height = `${box.height * 100}%`;
    el.innerHTML = `<b>${candidate.index}</b>`;
    el.addEventListener('click', () => selectCandidate(candidate.index));
    wrap.appendChild(el);
  });

  const count = data.candidates.length;
  const at = `frame ${data.frame_index} (${data.timestamp_seconds.toFixed(1)}s)`;
  if (count === 0) {
    setStatus('pick-status', `Nobody detected, ${at}. Try another frame.`, true);
  } else {
    setStatus('pick-status',
      `${count} ${count === 1 ? 'person' : 'people'} in ${at}. ` +
      (count === 1 ? 'Selected.' : 'Click the climber.'));
    if (count === 1) selectCandidate(0);
  }
}

function selectCandidate(index) {
  state.selected = index;
  $('frame-wrap').querySelectorAll('.cand').forEach((el, position) => {
    el.classList.toggle('selected', position === index);
  });
  $('analyse').disabled = false;
}

$('other-frame').addEventListener('click', () => {
  const total = state.video.frame_count;
  const current = state.candidates ? state.candidates.frame_index : Math.floor(total / 2);
  const next = (current + Math.round(state.video.fps * 2)) % total;
  loadCandidates(next);
});

// ----------------------------------------------------------------- analysis

$('analyse').addEventListener('click', async () => {
  $('analyse').disabled = true;
  show('s-progress', true);
  show('s-play', false);
  setStatus('job-status', 'Queued…');
  $('job-bar').style.width = '0%';
  try {
    const job = await api(`/videos/${state.video.id}/analyse`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ candidate_index: state.selected }),
    });
    pollJob(job.id);
  } catch (error) {
    setStatus('job-status', error.message, true);
    $('analyse').disabled = false;
  }
});

function pollJob(jobId) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let job;
    try {
      job = await api(`/jobs/${jobId}`);
    } catch (error) {
      clearInterval(state.poll);
      setStatus('job-status', error.message, true);
      return;
    }
    const percent = Math.round(job.progress * 100);
    $('job-bar').style.width = `${percent}%`;
    setStatus('job-status', `${job.status} — ${percent}%`);

    if (job.status === 'done') {
      clearInterval(state.poll);
      $('analyse').disabled = false;
      state.video = await api(`/videos/${state.video.id}`);
      await showResult();
      refreshLibrary();
    } else if (job.status === 'failed') {
      clearInterval(state.poll);
      $('analyse').disabled = false;
      setStatus('job-status', `failed: ${job.error}`, true);
    }
  }, 1000);
}

// ------------------------------------------------------------------- result

async function showResult() {
  const video = state.video;
  state.keypoints = await api(`/videos/${video.id}/keypoints`);
  show('s-play', true);
  show('s-pick', false);

  $('hud-total').textContent = state.keypoints.frame_count;

  const player = $('player');
  const hasFootage = video.has_footage;
  show('stage', hasFootage);
  show('no-footage', !hasFootage);
  $('delete-footage').disabled = !hasFootage;
  // Re-picking needs the clip: both the candidate frame and the re-analysis
  // read from it.
  $('repick').disabled = !hasFootage;
  ['step-back', 'step-fwd'].forEach((id) => { $(id).disabled = !hasFootage; });

  if (hasFootage) {
    player.src = `/videos/${video.id}/file`;
    player.load();
  } else {
    player.removeAttribute('src');
    setStatus('no-footage',
      'The footage has been deleted. The track below outlived it — that is the point.');
  }

  renderMeta();
  drawCharts();
  updateRetentionNote();
  requestAnimationFrame(drawOverlay);
}

function renderMeta() {
  const video = state.video;
  const kp = state.keypoints;
  const source = video.source;
  const rows = [
    ['Normalised to', `${video.width}×${video.height}, ${video.fps} fps, ${video.frame_count} frames`],
    ['Source', `${source.width}×${source.height} ${source.codec}, ${formatBytes(video.size_bytes)}`],
    ['Source rotation flag', source.rotation
      ? `<span class="flag">${source.rotation}° — baked into the pixels</span>` : 'none'],
    ['Source frame timing', source.variable_frame_rate
      ? '<span class="flag">variable — retimed to constant</span>' : 'constant'],
    ['Pose model', `${kp.pose_model}, tracked at IoU ≥ ${kp.min_iou}`],
  ];
  $('meta').innerHTML = rows
    .map(([label, value]) => `<tr><td>${label}</td><td>${value}</td></tr>`).join('');
}

// -------------------------------------------------------------- the overlay

function sizeCanvas(canvas, cssWidth, cssHeight) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(cssWidth * ratio));
  const height = Math.max(1, Math.round(cssHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return ratio;
}

function currentFrameIndex() {
  const data = state.keypoints;
  if (!data) return 0;
  // Frame N covers [N/fps, (N+1)/fps), so floor - not round - is the frame
  // actually on screen.
  const index = Math.floor($('player').currentTime * data.fps);
  return Math.max(0, Math.min(data.frame_count - 1, index));
}

function drawOverlay() {
  requestAnimationFrame(drawOverlay);
  const data = state.keypoints;
  if (!data || !state.video || !state.video.has_footage) return;

  const player = $('player');
  const canvas = $('overlay');
  const scale = sizeCanvas(canvas, player.clientWidth, player.clientHeight);
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);

  const index = currentFrameIndex();
  updateHud(index);
  drawPlayhead(index);

  const landmarks = data.frames[index];
  if (!landmarks) return;   // A gap draws nothing. Never interpolated across.

  const toX = (x) => x * canvas.width;
  const toY = (y) => y * canvas.height;

  context.lineWidth = 2.5 * scale;
  context.lineCap = 'round';
  for (const [start, end] of data.landmark_connections) {
    const a = landmarks[start];
    const b = landmarks[end];
    if (!a || !b) continue;
    // Opacity carries the model's own confidence, so uncertainty is visible
    // rather than hidden behind a threshold.
    const confidence = Math.min(a[2], b[2]);
    context.strokeStyle = `rgba(91,156,246,${Math.max(0.15, confidence)})`;
    context.beginPath();
    context.moveTo(toX(a[0]), toY(a[1]));
    context.lineTo(toX(b[0]), toY(b[1]));
    context.stroke();
  }
  for (const landmark of landmarks) {
    context.fillStyle = `rgba(255,255,255,${Math.max(0.2, landmark[2])})`;
    context.beginPath();
    context.arc(toX(landmark[0]), toY(landmark[1]), 3 * scale, 0, Math.PI * 2);
    context.fill();
  }
}

let lastHudFrame = -1;
function updateHud(index) {
  if (index === lastHudFrame) return;
  lastHudFrame = index;
  const data = state.keypoints;
  $('hud-frame').textContent = index;
  const tracked = data.frames[index] !== null;
  const stateEl = $('hud-state');
  stateEl.textContent = tracked ? 'tracked' : 'gap — no match';
  stateEl.className = tracked ? 'st-ok' : 'st-gap';
  const iou = data.match_iou ? data.match_iou[index] : null;
  $('hud-iou').textContent = (iou === null || iou === undefined) ? '—' : iou.toFixed(2);
}

function stepFrame(delta) {
  const data = state.keypoints;
  if (!data) return;
  const player = $('player');
  player.pause();
  const next = Math.max(0, Math.min(data.frame_count - 1, currentFrameIndex() + delta));
  // Nudged into the middle of the frame's window so rounding cannot land it
  // on the neighbour.
  player.currentTime = (next + 0.5) / data.fps;
}
$('step-back').addEventListener('click', () => stepFrame(-1));
$('step-fwd').addEventListener('click', () => stepFrame(1));
document.addEventListener('keydown', (event) => {
  if ($('s-play').hidden) return;
  if (event.target.tagName === 'INPUT') return;
  if (event.key === 'ArrowLeft') { stepFrame(-1); event.preventDefault(); }
  if (event.key === 'ArrowRight') { stepFrame(1); event.preventDefault(); }
});

// -------------------------------------------------------------- the charts
//
// Two charts over one shared x axis (frame index), drawn as separate figures
// rather than stacked on one pair of axes. Coverage answers "where did it
// lose the climber", confidence answers "how sure was it where it did not".

function chartGeometry(canvas, cssHeight) {
  const wrap = canvas.parentElement;
  const width = wrap.clientWidth;
  const ratio = sizeCanvas(canvas, width, cssHeight);
  canvas.style.height = `${cssHeight}px`;
  return { context: canvas.getContext('2d'), ratio, width, height: cssHeight };
}

/** Per-pixel column, the frames it covers. */
function columnRange(x, width, frameCount) {
  const from = Math.floor((x / width) * frameCount);
  const to = Math.max(from + 1, Math.ceil(((x + 1) / width) * frameCount));
  return [from, Math.min(frameCount, to)];
}

function drawCoverage() {
  const data = state.keypoints;
  const canvas = $('coverage');
  const { context, ratio, width, height } = chartGeometry(canvas, 34);
  context.clearRect(0, 0, canvas.width, canvas.height);

  for (let x = 0; x < width; x += 1) {
    const [from, to] = columnRange(x, width, data.frame_count);
    let gaps = 0;
    for (let f = from; f < to; f += 1) if (data.frames[f] === null) gaps += 1;
    // A column showing any gap is drawn as a gap. At this zoom a single
    // dropped frame is a sub-pixel event, and rounding it away would hide
    // exactly the failures this chart exists to show.
    context.fillStyle = gaps > 0 ? COLOUR.gap : COLOUR.track;
    context.fillRect(x * ratio, 0, ratio + 0.5, height * ratio);
  }
}

function drawConfidence() {
  const data = state.keypoints;
  const canvas = $('iouchart');
  const { context, ratio, width, height } = chartGeometry(canvas, 72);
  context.clearRect(0, 0, canvas.width, canvas.height);

  const pad = 6 * ratio;
  const plotHeight = height * ratio - pad * 2;
  const toY = (value) => pad + (1 - value) * plotHeight;

  const tracked = data.match_iou.filter((v) => v !== null);
  const median = tracked.length
    ? tracked.slice().sort((a, b) => a - b)[Math.floor(tracked.length / 2)] : 0;

  // Recessive reference line at the median, labelled directly rather than
  // with a full axis - one number is the only one worth reading off.
  context.strokeStyle = COLOUR.line;
  context.lineWidth = 1 * ratio;
  context.setLineDash([3 * ratio, 3 * ratio]);
  context.beginPath();
  context.moveTo(0, toY(median));
  context.lineTo(canvas.width, toY(median));
  context.stroke();
  context.setLineDash([]);

  // The line breaks at gaps rather than spanning them: joining across a hole
  // would draw a confidence that was never measured.
  context.strokeStyle = COLOUR.track;
  context.lineWidth = 2 * ratio;
  context.lineJoin = 'round';
  context.beginPath();
  let drawing = false;
  for (let x = 0; x < width; x += 1) {
    const [from, to] = columnRange(x, width, data.frame_count);
    let sum = 0;
    let seen = 0;
    for (let f = from; f < to; f += 1) {
      const value = data.match_iou[f];
      if (value !== null && value !== undefined) { sum += value; seen += 1; }
    }
    if (seen === 0) { drawing = false; continue; }
    const y = toY(sum / seen);
    if (drawing) context.lineTo(x * ratio, y);
    else { context.moveTo(x * ratio, y); drawing = true; }
  }
  context.stroke();

  context.fillStyle = COLOUR.muted;
  context.font = `${11 * ratio}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  context.fillText(`median ${median.toFixed(2)}`, 4 * ratio, toY(median) - 5 * ratio);
}

function drawPlayhead(index) {
  const data = state.keypoints;
  for (const [id, cssHeight] of [['coverage', 34], ['iouchart', 72]]) {
    const canvas = $(id);
    if (!canvas.width) continue;
    const context = canvas.getContext('2d');
    const previous = canvas._playhead;
    if (previous !== undefined && previous === index) continue;
    canvas._playhead = index;
    if (id === 'coverage') drawCoverage(); else drawConfidence();
    const x = (index / data.frame_count) * canvas.width;
    context.strokeStyle = '#ffffff';
    context.lineWidth = Math.max(1, window.devicePixelRatio || 1);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, cssHeight * (window.devicePixelRatio || 1));
    context.stroke();
  }
}

function drawCharts() {
  const data = state.keypoints;
  $('coverage')._playhead = undefined;
  $('iouchart')._playhead = undefined;
  drawCoverage();
  drawConfidence();

  const pct = (100 * data.tracked_frame_count / data.frame_count).toFixed(0);
  $('cov-summary').textContent =
    `${data.tracked_frame_count} of ${data.frame_count} frames tracked (${pct}%), ` +
    `${data.gap_frame_count} in gaps`;

  const tracked = data.match_iou.filter((v) => v !== null).sort((a, b) => a - b);
  if (tracked.length) {
    $('iou-summary').textContent =
      `min ${tracked[0].toFixed(2)} · p10 ${tracked[Math.floor(tracked.length * 0.1)].toFixed(2)} ` +
      `· p90 ${tracked[Math.floor(tracked.length * 0.9)].toFixed(2)}`;
  } else {
    $('iou-summary').textContent = 'nothing was tracked';
  }
}

// Hover: an HTML chart is interactive, so both strips get a tooltip and both
// seek on click.
for (const [wrapId, canvasId] of [['cov-wrap', 'coverage'], ['iou-wrap', 'iouchart']]) {
  const wrap = $(wrapId);
  let tip = null;

  wrap.addEventListener('mousemove', (event) => {
    const data = state.keypoints;
    if (!data) return;
    const rect = $(canvasId).getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width - 1, event.clientX - rect.left));
    const [from, to] = columnRange(x, rect.width, data.frame_count);
    let gaps = 0;
    let sum = 0;
    let seen = 0;
    for (let f = from; f < to; f += 1) {
      if (data.frames[f] === null) gaps += 1;
      const value = data.match_iou[f];
      if (value !== null && value !== undefined) { sum += value; seen += 1; }
    }
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'tip';
      wrap.appendChild(tip);
    }
    const seconds = (from / data.fps).toFixed(1);
    const span = to - from > 1 ? `frames ${from}–${to - 1}` : `frame ${from}`;
    tip.innerHTML =
      `<span class="k">${span}</span> · ${seconds}s<br>` +
      (gaps === to - from
        ? '<span class="k">gap</span> — no match'
        : `IoU ${seen ? (sum / seen).toFixed(2) : '—'}` +
          (gaps ? ` · <span class="k">${gaps} gap frame${gaps > 1 ? 's' : ''}</span>` : ''));
    tip.style.left = `${x}px`;
    tip.style.top = `${rect.height}px`;
  });

  wrap.addEventListener('mouseleave', () => {
    if (tip) { tip.remove(); tip = null; }
  });

  wrap.addEventListener('click', (event) => {
    const data = state.keypoints;
    if (!data || !state.video.has_footage) return;
    const rect = $(canvasId).getBoundingClientRect();
    const fraction = (event.clientX - rect.left) / rect.width;
    $('player').currentTime = Math.max(0, fraction * data.frame_count / data.fps);
  });
}

window.addEventListener('resize', () => {
  if (state.keypoints) drawCharts();
});

// ---------------------------------------------------------------- retention

function updateRetentionNote() {
  clearInterval(state.retentionTick);
  const note = $('retention-note');
  const video = state.video;

  if (!video.has_footage) {
    note.textContent = 'Footage deleted. Keypoints and metadata are kept.';
    return;
  }
  if (!video.footage_expires_at) {
    note.textContent = '';
    return;
  }
  const render = () => {
    const remaining = (new Date(video.footage_expires_at) - new Date()) / 1000;
    note.textContent = remaining > 0
      ? `Footage is deleted automatically in ${formatDuration(remaining)}.`
      : 'Footage is due for deletion; the next sweep will remove it.';
  };
  render();
  state.retentionTick = setInterval(render, 10000);
}

$('repick').addEventListener('click', async () => {
  // Straight back to the picker on the frame the last choice came from, so a
  // track that followed the wrong person can be redone without re-uploading.
  const stored = state.candidates ? state.candidates.frame_index : undefined;
  show('s-play', false);
  await loadCandidates(stored);
  $('s-pick').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

$('delete-footage').addEventListener('click', async () => {
  if (!confirm('Delete the video file now? The keypoints and the charts stay.')) return;
  $('delete-footage').disabled = true;
  try {
    state.video = await api(`/videos/${state.video.id}/footage`, { method: 'DELETE' });
    await showResult();
    refreshLibrary();
  } catch (error) {
    setStatus('no-footage', error.message, true);
    $('delete-footage').disabled = false;
  }
});

// --------------------------------------------------------------------- boot

loadConfig().then(refreshLibrary).catch((error) => {
  setStatus('upload-status', `Could not reach the server: ${error.message}`, true);
});
