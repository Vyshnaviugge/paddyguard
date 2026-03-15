// PaddyGuard AI — Main JS

// ── Particles ─────────────────────────────────────────────────────────────────
function initParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.cssText = `
      left:${Math.random()*100}%;
      width:${2+Math.random()*4}px;
      height:${2+Math.random()*4}px;
      animation-duration:${8+Math.random()*14}s;
      animation-delay:${Math.random()*10}s;
      opacity:${.1+Math.random()*.3};
    `;
    container.appendChild(p);
  }
}
initParticles();

// ── State ─────────────────────────────────────────────────────────────────────
let currentFile = null;

// ── Drag & Drop ───────────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

if (dropZone) {
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
  dropZone.addEventListener('click', e => {
    if (e.target === dropZone || e.target.closest('.upload-idle')) {
      if (document.getElementById('uploadIdle').style.display !== 'none') fileInput.click();
    }
  });
}

if (fileInput) {
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/webp'];
  if (!allowed.includes(file.type)) {
    alert('Please upload a JPG, PNG, or WEBP image.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    alert('File too large. Max 10MB.');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('previewImg').src = e.target.result;
    document.getElementById('uploadIdle').style.display = 'none';
    document.getElementById('uploadPreview').style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  currentFile = null;
  fileInput.value = '';
  document.getElementById('uploadIdle').style.display = 'block';
  document.getElementById('uploadPreview').style.display = 'none';
}

// ── Loading Steps ─────────────────────────────────────────────────────────────
const steps = [
  'Running CNN model...',
  'Analyzing leaf patterns...',
  'Detecting disease markers...',
  'Generating recommendations...',
  'Preparing your report...'
];
let stepTimer = null;

function startLoading() {
  let i = 0;
  const el = document.getElementById('loadingStep');
  if (el) el.textContent = steps[0];
  stepTimer = setInterval(() => {
    i = (i + 1) % steps.length;
    if (el) el.textContent = steps[i];
  }, 950);
}

function stopLoading() {
  clearInterval(stepTimer);
}

// ── Analyze ───────────────────────────────────────────────────────────────────
async function analyzeImage() {
  if (!currentFile) return;
  document.getElementById('loadingOverlay').style.display = 'flex';
  document.getElementById('resultsSection').style.display = 'none';
  startLoading();
  const fd = new FormData();
  fd.append('image', currentFile);
  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResults(data);
  } catch(e) {
    alert('Something went wrong. Please try again.');
    console.error(e);
  } finally {
    stopLoading();
    document.getElementById('loadingOverlay').style.display = 'none';
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
  // Image
  document.getElementById('resultImg').src = data.image_url;

  // Severity badge
  const sev = document.getElementById('resultSev');
  sev.textContent = data.severity_icon + '  ' + data.severity_label;
  sev.style.background   = data.severity_bg;
  sev.style.color        = data.severity_color;
  sev.style.borderColor  = data.severity_border;

  // Urgency
  const urg = document.getElementById('resultUrg');
  urg.textContent = data.severity_level > 0 ? '⏱ Act ' + data.urgency : '';

  // Disease info
  document.getElementById('resultName').textContent     = data.display_name;
  document.getElementById('resultDescEl').textContent   = data.description;
  document.getElementById('resultActionEl').textContent = data.action;

  // Confidence ring
  const pct = data.confidence;
  document.getElementById('confPct').textContent = pct + '%';
  const circle = document.getElementById('confCircle');
  const circumference = 2 * Math.PI * 50;
  const offset = circumference - (pct / 100) * circumference;
  let color = '#16a34a';
  if (pct < 50) color = '#ef4444';
  else if (pct < 75) color = '#f97316';
  circle.style.stroke = color;
  setTimeout(() => {
    circle.style.transition = 'stroke-dashoffset 1.2s ease';
    circle.setAttribute('stroke-dashoffset', offset);
  }, 120);

  // Top 3
  document.getElementById('top3El').innerHTML = data.top3.map((item, i) => `
    <div class="top3-row ${i === 0 ? 'top1' : ''}">
      <span class="top3-n">${item.display}</span>
      <span class="top3-p">${item.prob}%</span>
    </div>`).join('');

  // Low confidence warning
  document.getElementById('lowconfWarn').style.display = data.low_confidence ? 'flex' : 'none';

  // Pesticides
  renderPest('chemGrid', data.chemical, 'chem', 'Chemical');
  renderPest('orgGrid', data.organic, 'org', 'Organic');
  switchTab('chem');

  // Tips
  document.getElementById('tipsGrid').innerHTML = (data.tips || []).map((t, i) => `
    <div class="tip-card">
      <span class="tip-n">${String(i + 1).padStart(2, '0')}</span>
      <span>${t}</span>
    </div>`).join('');

  // Show results
  document.getElementById('resultsSection').style.display = 'block';
  setTimeout(() => {
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function renderPest(panelId, list, cls, label) {
  const panel = document.getElementById(panelId);
  if (!list || list.length === 0) {
    panel.innerHTML = `<div class="no-treat">No ${label.toLowerCase()} treatment required.</div>`;
    return;
  }
  panel.innerHTML = list.map(p => `
    <div class="pcard ${cls}">
      <div class="pb ${cls}">${label}</div>
      <div class="pname">${p.name}</div>
      <div class="prows">
        <div class="prow"><span class="pk">Dosage</span><span class="pv">${p.dosage}</span></div>
        <div class="prow"><span class="pk">Frequency</span><span class="pv">${p.frequency}</span></div>
      </div>
    </div>`).join('');
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.getElementById('chemGrid').style.display = tab === 'chem' ? 'grid' : 'none';
  document.getElementById('orgGrid').style.display  = tab === 'org'  ? 'grid' : 'none';
  document.getElementById('tabChem').classList.toggle('active', tab === 'chem');
  document.getElementById('tabOrg').classList.toggle('active', tab === 'org');
}

// ── Reset All ─────────────────────────────────────────────────────────────────
function resetAll() {
  resetUpload();
  document.getElementById('resultsSection').style.display = 'none';
  const c = document.getElementById('confCircle');
  if (c) {
    c.style.transition = 'none';
    c.setAttribute('stroke-dashoffset', '314');
  }
  document.getElementById('upload').scrollIntoView({ behavior: 'smooth' });
}