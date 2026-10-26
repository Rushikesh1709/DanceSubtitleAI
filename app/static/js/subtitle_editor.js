/* Subtitle editor: edit text/times, add, delete, merge, split, save, live preview. */
(function () {
  const root = document.getElementById('editor');
  const rowsEl = document.getElementById('rows');
  const video = document.getElementById('video');
  const caption = document.getElementById('caption');
  const msgEl = document.getElementById('messages');
  let segments = [], dirty = false;

  function say(kind, lines) {
    lines = Array.isArray(lines) ? lines : [lines];
    msgEl.innerHTML = '<div class="alert alert-' + kind + ' py-2 mb-0">' + lines.map(escapeHtml).join('<br>') + '</div>';
  }
  function markDirty() { dirty = true; }
  window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

  function collect() {
    // Read the inputs into `segments`; returns an array of error strings (empty if all parse).
    const errors = [];
    const rows = rowsEl.querySelectorAll('.seg-row');
    rows.forEach((row, i) => {
      const s = segments[i];
      s.text = row.querySelector('textarea').value;
      row.classList.remove('invalid');
      try {
        s.start = parseTs(row.querySelector('.in-start').value);
        s.end = parseTs(row.querySelector('.in-end').value);
      } catch (e) { errors.push('Subtitle ' + (i + 1) + ': invalid time.'); row.classList.add('invalid'); }
    });
    return errors;
  }

  function render() {
    rowsEl.innerHTML = '';
    segments.forEach((s, i) => {
      const row = document.createElement('div');
      row.className = 'seg-row';
      row.innerHTML =
        '<div class="d-flex flex-wrap gap-2 align-items-center mb-2">' +
        '<span class="badge text-bg-secondary">' + (i + 1) + '</span>' +
        '<input class="form-control form-control-sm in-start" style="width:9.5rem" value="' + fmtTs(s.start) + '">' +
        '<span>→</span>' +
        '<input class="form-control form-control-sm in-end" style="width:9.5rem" value="' + fmtTs(s.end) + '">' +
        '<button class="btn btn-sm btn-outline-secondary ms-auto b-go" type="button">Go to</button>' +
        '</div>' +
        '<textarea class="form-control form-control-sm" rows="2" maxlength="500"></textarea>' +
        '<div class="mt-2 d-flex flex-wrap gap-2">' +
        '<button class="btn btn-sm btn-outline-primary b-split" type="button">Split</button>' +
        '<button class="btn btn-sm btn-outline-primary b-merge" type="button"' + (i === segments.length - 1 ? ' disabled' : '') + '>Merge with next</button>' +
        '<button class="btn btn-sm btn-outline-danger b-del" type="button">Delete</button></div>';
      row.querySelector('textarea').value = s.text;
      row.querySelectorAll('input,textarea').forEach(el => el.addEventListener('input', () => { markDirty(); collect(); updateCaption(); }));
      row.querySelector('.b-go').onclick = () => { collect(); video.currentTime = Math.max(0, segments[i].start + 0.01); video.play(); };
      row.querySelector('.b-del').onclick = () => { if (collect().length) return say('danger', 'Fix invalid times first.'); segments.splice(i, 1); markDirty(); render(); };
      row.querySelector('.b-merge').onclick = () => transform('merge', i);
      row.querySelector('.b-split').onclick = () => {
        if (collect().length) return say('danger', 'Fix invalid times first.');
        const s0 = segments[i];
        let at = video.currentTime;
        if (!(at > s0.start + 0.3 && at < s0.end - 0.3)) at = (s0.start + s0.end) / 2;
        transform('split', i, at);
      };
      rowsEl.appendChild(row);
    });
    if (!segments.length) rowsEl.innerHTML = '<p class="text-muted">No subtitles yet. Use “Add subtitle at current time”.</p>';
    updateCaption();
  }

  async function transform(op, index, at) {
    const errs = collect();
    if (errs.length) return say('danger', errs);
    const r = await fetch(root.dataset.transformUrl, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({op: op, segments: segments, index: index, at: at})});
    const d = await r.json();
    if (!r.ok) return say('danger', d.errors || 'Could not do that.');
    segments = d.segments; markDirty(); render(); say('info', op === 'merge' ? 'Merged.' : 'Split. Remember to save.');
  }

  function updateCaption() {
    const t = video.currentTime;
    const i = activeSegmentIndex(segments, t);
    caption.textContent = i >= 0 && segments[i].text ? segments[i].text : '\u00a0';
    rowsEl.querySelectorAll('.seg-row').forEach((r, k) => r.classList.toggle('active', k === i));
  }
  ['timeupdate', 'seeked', 'play', 'pause'].forEach(ev => video.addEventListener(ev, updateCaption));

  document.getElementById('add-btn').onclick = () => {
    if (collect().length) return say('danger', 'Fix invalid times first.');
    const dur = parseFloat(root.dataset.duration) || 0;
    let start = video.currentTime || 0;
    const clash = segments.find(s => start >= s.start && start < s.end);
    if (clash) start = clash.end + 0.1;
    let end = Math.min(start + 2, dur || start + 2);
    const next = segments.find(s => s.start >= start);
    if (next) end = Math.min(end, next.start - 0.05);
    if (end - start < 0.2) return say('danger', 'There is no free time at the current position. Move the video or shorten a neighbouring subtitle.');
    segments.push({start: start, end: end, text: 'New subtitle', category: null, confidence: null});
    segments.sort((a, b) => a.start - b.start);
    markDirty(); render();
  };

  document.getElementById('save-btn').onclick = async () => {
    const errs = collect();
    if (errs.length) return say('danger', errs);
    const r = await fetch(root.dataset.segmentsUrl, {method: 'PUT', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({segments: segments})});
    const d = await r.json();
    if (!r.ok) return say('danger', d.errors || 'Could not save.');
    dirty = false; say('success', 'Saved ' + d.count + ' subtitles. The downloads now contain your edits.');
  };

  fetch(root.dataset.segmentsUrl).then(r => r.json()).then(d => { segments = d.segments; render(); });
})();
