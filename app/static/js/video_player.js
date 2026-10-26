/* Results page: subtitle toggle, caption panel, timeline, table, intensity chart. */
(function () {
  const root = document.getElementById('results');
  const video = document.getElementById('video');
  const caption = document.getElementById('caption');
  const timeline = document.getElementById('timeline');
  const tbody = document.querySelector('#seg-table tbody');
  const toggle = document.getElementById('toggle-subs');
  const duration = parseFloat(root.dataset.duration) || 0;
  let segments = [], analysis = null, lastActive = -2;

  function setTrackMode() {
    const mode = toggle.checked ? 'showing' : 'disabled';
    for (const tr of video.textTracks) tr.mode = mode;
    caption.style.visibility = toggle.checked ? 'visible' : 'hidden';
  }
  toggle.addEventListener('change', setTrackMode);
  video.addEventListener('loadedmetadata', setTrackMode);
  video.textTracks.addEventListener && video.textTracks.addEventListener('addtrack', setTrackMode);

  function render() {
    timeline.innerHTML = '<div class="cursor" id="cursor"></div>';
    tbody.innerHTML = '';
    segments.forEach((s, i) => {
      const b = document.createElement('div');
      b.className = 'blk' + (s.category === 'no_pose' ? ' noposs' : (s.category === 'uncertain' ? ' uncertain' : ''));
      b.style.left = (100 * s.start / duration) + '%';
      b.style.width = Math.max(0.4, 100 * (s.end - s.start) / duration) + '%';
      b.title = fmtTs(s.start) + ' – ' + s.text;
      b.dataset.i = i;
      b.onclick = () => { video.currentTime = s.start + 0.01; };
      timeline.appendChild(b);
      const tr = document.createElement('tr');
      tr.dataset.i = i;
      tr.innerHTML = '<td class="text-nowrap">' + fmtTs(s.start) + '</td><td class="text-nowrap">' + fmtTs(s.end) +
        '</td><td>' + escapeHtml(s.text) + '</td><td>' + (s.confidence == null ? '–' : s.confidence.toFixed(2)) + '</td>';
      tr.onclick = () => { video.currentTime = s.start + 0.01; };
      tbody.appendChild(tr);
    });
  }

  function drawChart() {
    if (!analysis) return;
    const bands = analysis.intervals.filter(i => i.kind === 'no_pose').map(i => [i.start, i.end]);
    drawLineChart(document.getElementById('chart'), analysis.intensity, {
      duration: duration, markers: segments.map(s => s.start), bands: bands, cursor: video.currentTime || 0 });
  }

  function tick() {
    const t = video.currentTime;
    const i = activeSegmentIndex(segments, t);
    if (i !== lastActive) {
      lastActive = i;
      caption.textContent = i >= 0 ? segments[i].text : '\u00a0';
      document.querySelectorAll('.timeline .blk.active, #seg-table tr.active').forEach(e => e.classList.remove('active'));
      if (i >= 0) {
        timeline.querySelector('.blk[data-i="' + i + '"]')?.classList.add('active');
        const row = tbody.querySelector('tr[data-i="' + i + '"]');
        if (row) { row.classList.add('active'); }
      }
    }
    const c = document.getElementById('cursor');
    if (c && duration) c.style.left = (100 * t / duration) + '%';
    if (!video.paused) requestAnimationFrame(tick);
  }
  video.addEventListener('play', () => requestAnimationFrame(tick));
  ['seeked', 'timeupdate', 'pause'].forEach(ev => video.addEventListener(ev, () => { tick(); }));
  video.addEventListener('seeked', drawChart);
  video.addEventListener('pause', drawChart);
  window.addEventListener('resize', drawChart);

  Promise.all([
    fetch(root.dataset.segmentsUrl).then(r => r.json()),
    fetch(root.dataset.analysisUrl).then(r => r.ok ? r.json() : null).catch(() => null)
  ]).then(([s, a]) => { segments = s.segments; analysis = a; render(); drawChart(); tick(); });
})();
