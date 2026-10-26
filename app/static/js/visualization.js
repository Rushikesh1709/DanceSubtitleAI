/* Visualisation page: skeleton + hand overlay synced to the video, intensity chart, events list. */
(function () {
  const root = document.getElementById('viz');
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  const caption = document.getElementById('caption');
  const chart = document.getElementById('chart');
  const POSE = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28],[27,29],[29,31],[27,31],[28,30],[30,32],[28,32],[15,19],[16,20],[0,11],[0,12]];
  const HAND = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  let A = null, segments = [], samples = [];

  function nearest(t) {
    let lo = 0, hi = samples.length - 1;
    if (hi < 0) return null;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (samples[mid].t < t) lo = mid + 1; else hi = mid; }
    let best = lo;
    if (lo > 0 && Math.abs(samples[lo - 1].t - t) < Math.abs(samples[lo].t - t)) best = lo - 1;
    const maxGap = 1.2 / (A.meta.sample_fps || 5);
    return Math.abs(samples[best].t - t) <= maxGap ? samples[best] : null;
  }

  function fit() {
    // The <video> element letterboxes the picture; compute where the picture really is.
    const W = video.clientWidth, H = video.clientHeight;
    canvas.width = W; canvas.height = H; canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    const vw = video.videoWidth || A.meta.width, vh = video.videoHeight || A.meta.height;
    const scale = Math.min(W / vw, H / vh);
    return {ox: (W - vw * scale) / 2, oy: (H - vh * scale) / 2, w: vw * scale, h: vh * scale};
  }

  function draw() {
    if (!A) return;
    const box = fit();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const s = nearest(video.currentTime);
    if (s) {
      const P = (x, y) => [box.ox + x * box.w, box.oy + y * box.h];
      if (document.getElementById('show-pose').checked && s.pose) {
        ctx.lineWidth = 3; ctx.strokeStyle = '#2ecc71';
        POSE.forEach(([a, b]) => {
          const pa = s.pose[a], pb = s.pose[b];
          if (pa[2] < 0.5 || pb[2] < 0.5) return;
          const [x1, y1] = P(pa[0], pa[1]), [x2, y2] = P(pb[0], pb[1]);
          ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        });
        ctx.fillStyle = '#f1c40f';
        s.pose.forEach(p => { if (p[2] >= 0.5) { const [x, y] = P(p[0], p[1]); ctx.beginPath(); ctx.arc(x, y, 3.5, 0, 6.283); ctx.fill(); } });
      }
      if (document.getElementById('show-hands').checked && s.hands) {
        ['left', 'right'].forEach((side, k) => {
          const h = s.hands[side]; if (!h) return;
          ctx.strokeStyle = k ? '#3498db' : '#e74c3c'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
          HAND.forEach(([a, b]) => { const [x1, y1] = P(h[a][0], h[a][1]), [x2, y2] = P(h[b][0], h[b][1]); ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); });
          h.forEach(p => { const [x, y] = P(p[0], p[1]); ctx.beginPath(); ctx.arc(x, y, 2, 0, 6.283); ctx.fill(); });
        });
      }
    }
    const i = activeSegmentIndex(segments, video.currentTime);
    caption.textContent = i >= 0 ? segments[i].text : '\u00a0';
  }

  function drawChart() {
    if (!A) return;
    drawLineChart(chart, A.intensity, {duration: A.meta.duration, markers: segments.map(s => s.start),
      bands: A.intervals.filter(i => i.kind === 'no_pose').map(i => [i.start, i.end]), cursor: video.currentTime || 0});
  }

  function loop() { draw(); if (!video.paused) requestAnimationFrame(loop); }
  video.addEventListener('play', () => requestAnimationFrame(loop));
  ['seeked', 'pause', 'timeupdate', 'loadedmetadata'].forEach(ev => video.addEventListener(ev, () => { draw(); }));
  ['seeked', 'pause'].forEach(ev => video.addEventListener(ev, drawChart));
  window.addEventListener('resize', () => { draw(); drawChart(); });

  Promise.all([fetch(root.dataset.analysisUrl).then(r => r.json()), fetch(root.dataset.segmentsUrl).then(r => r.json())])
    .then(([a, s]) => {
      A = a; samples = a.samples; segments = s.segments;
      const tb = document.querySelector('#events tbody');
      a.events.filter(e => e.type === 'movement' || e.type === 'turn' || e.type === 'repeated_movement' || e.type === 'no_pose').forEach(e => {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td class="text-nowrap">' + fmtTs(e.start) + '</td><td>' + escapeHtml(e.label) + '</td>';
        tr.style.cursor = 'pointer'; tr.onclick = () => { video.currentTime = e.start + 0.01; };
        tb.appendChild(tr);
      });
      draw(); drawChart();
    });
})();
