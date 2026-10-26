/* Shared helpers used by several pages. */
function fmtTs(sec) {
  sec = Math.max(0, sec || 0);
  const ms = Math.round(sec * 1000);
  const h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60, r = ms % 1000;
  const p = (n, w) => String(n).padStart(w, '0');
  return p(h, 2) + ':' + p(m, 2) + ':' + p(s, 2) + '.' + p(r, 3);
}

function parseTs(text) {
  const t = String(text).trim().replace(',', '.');
  if (!t) throw new Error('empty time');
  const parts = t.split(':');
  if (parts.length > 3) throw new Error('bad time');
  let total = 0;
  for (const p of parts) {
    const n = Number(p);
    if (!isFinite(n) || n < 0 || p.trim() === '') throw new Error('bad time');
    total = total * 60 + n;
  }
  return total;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
}

/* Simple dependency-free line chart.
   opts: {duration, markers: [t], bands: [[t0,t1]], cursor: t, color} */
function drawLineChart(canvas, points, opts) {
  opts = opts || {};
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 600, h = parseInt(canvas.getAttribute('height') || '140', 10);
  canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const padL = 34, padR = 8, padT = 8, padB = 18;
  const dur = opts.duration || (points.length ? points[points.length - 1][0] : 1) || 1;
  const maxV = Math.max(0.5, ...points.map(p => p[1]));
  const X = t => padL + (t / dur) * (w - padL - padR);
  const Y = v => h - padB - (v / maxV) * (h - padT - padB);
  (opts.bands || []).forEach(b => { ctx.fillStyle = 'rgba(108,117,125,.25)'; ctx.fillRect(X(b[0]), padT, X(b[1]) - X(b[0]), h - padT - padB); });
  ctx.strokeStyle = '#ced4da'; ctx.fillStyle = '#6c757d'; ctx.font = '11px sans-serif'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = (maxV * i) / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(1), 2, y + 4);
  }
  const ticks = Math.min(8, Math.max(2, Math.floor(w / 90)));
  for (let i = 0; i <= ticks; i++) { const t = (dur * i) / ticks; ctx.fillText(Math.round(t) + 's', X(t) - 8, h - 4); }
  (opts.markers || []).forEach(t => { ctx.strokeStyle = 'rgba(230,126,34,.9)'; ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), h - padB); ctx.stroke(); });
  ctx.strokeStyle = opts.color || '#1f6feb'; ctx.lineWidth = 1.6; ctx.beginPath();
  points.forEach((p, i) => { const x = X(p[0]), y = Y(p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  if (opts.cursor != null) { ctx.strokeStyle = '#dc3545'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(opts.cursor), padT); ctx.lineTo(X(opts.cursor), h - padB); ctx.stroke(); }
}

function activeSegmentIndex(segments, t) {
  for (let i = 0; i < segments.length; i++) if (t >= segments[i].start && t < segments[i].end) return i;
  return -1;
}
