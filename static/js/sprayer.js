/* ═══════════════════════════════════════════════════════════════
   PADDYGUARD — SPRAYER BOOKING JS
   Include this file in index.html AFTER main.js:
   <script src="/static/js/sprayer.js"></script>
   OR paste the contents at the bottom of main.js
═══════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────
let _sprayerBookingData = {
  sprayer_email: '',
  sprayer_name:  '',
  rate:          '',
  disease:       '',   // set after prediction
  disease_label: '',
};

// ── Called from main.js after prediction succeeds ─────────────
// Call this from your existing displayResults() function:
// e.g. after rendering pesticides → loadSprayerSection(data.display_name, data.severity_level)
function loadSprayerSection(diseaseName, severityLevel) {
  const sec = document.getElementById('sprayerSec');
  if (!sec) return;

  _sprayerBookingData.disease       = diseaseName;
  _sprayerBookingData.disease_label = diseaseName;

  // Only show the section if disease is detected (severity > 0)
  if (severityLevel === 0) {
    sec.style.display = 'none';
    return;
  }

  sec.style.display = 'block';

  const title = document.getElementById('sprayerSecTitle');
  if (title) {
    title.textContent = severityLevel >= 3
      ? `Urgent: Book a Certified Sprayer for ${diseaseName}`
      : `Book a Certified Sprayer Near You`;
  }

  // Set min date to today on the date input
  const todayStr = new Date().toISOString().split('T')[0];
  const bkDate   = document.getElementById('bk_date');
  if (bkDate) { bkDate.min = todayStr; bkDate.value = todayStr; }

  // Pre-fill notes
  const bkNotes = document.getElementById('bk_notes');
  if (bkNotes) {
    bkNotes.value = `Disease detected: ${diseaseName}\nSeverity: ${getSeverityLabel(severityLevel)}\nPlease bring recommended pesticides for treatment.`;
  }

  // Try GPS first, fall back to no-geo
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => fetchSprayers(pos.coords.latitude, pos.coords.longitude),
      ()  => fetchSprayers(null, null),
      { timeout: 6000 }
    );
  } else {
    fetchSprayers(null, null);
  }
}

function getSeverityLabel(level) {
  return ['No Disease', 'Low', 'Moderate', 'High', 'Critical'][level] || 'Unknown';
}

// ── Fetch sprayers from API ───────────────────────────────────
function fetchSprayers(lat, lng) {
  const loading = document.getElementById('sprayerLoading');
  const grid    = document.getElementById('sprayerGrid');
  const empty   = document.getElementById('sprayerEmpty');

  if (loading) loading.style.display = 'flex';
  if (grid)    grid.innerHTML = '';
  if (empty)   empty.style.display = 'none';

  let url = '/api/sprayers';
  if (lat && lng) url += `?lat=${lat}&lng=${lng}&disease=${encodeURIComponent(_sprayerBookingData.disease)}`;

  fetch(url)
    .then(r => r.json())
    .then(data => {
      if (loading) loading.style.display = 'none';
      if (!data.sprayers || data.sprayers.length === 0) {
        if (empty) empty.style.display = 'flex';
        return;
      }
      renderSprayerCards(data.sprayers);
      loadMyBookings();
    })
    .catch(() => {
      if (loading) loading.style.display = 'none';
      if (empty)   empty.style.display = 'flex';
    });
}

// ── Called from empty state link ─────────────────────────────
function loadSprayersWithoutGeo() {
  fetchSprayers(null, null);
}

// ── Render sprayer cards ──────────────────────────────────────
function renderSprayerCards(sprayers) {
  const grid = document.getElementById('sprayerGrid');
  if (!grid) return;

  grid.innerHTML = sprayers.map((s, i) => {
    const isRec       = i === 0;
    const avatarColor = ['#dcfce7:14532d', '#dbeafe:1e40af', '#fef3c7:92400e'][i % 3];
    const [bg, fg]    = avatarColor.split(':');
    const availDot    = s.available
      ? '<span class="spr-avail-dot"></span>'
      : '<span class="spr-avail-dot busy"></span>';
    const tags = (s.diseases || []).slice(0, 3)
      .map(d => `<span class="spr-tag">${d}</span>`).join('');
    const rateStr = `₹${s.rate}/${s.rate_unit || 'acre'}`;

    return `
    <div class="spr-card ${isRec ? 'spr-recommended' : ''}">
      <div class="spr-card-top">
        <div class="spr-avatar" style="background:#${bg};color:#${fg};">${s.initials || '?'}</div>
        <div class="spr-name-row">
          <div class="spr-name">
            ${s.name}
            ${isRec ? '<span class="spr-rec-pill">Recommended</span>' : ''}
          </div>
          <div class="spr-sub">Certified · ${s.distance_fmt || 'Nearby'}</div>
        </div>
      </div>

      <div class="spr-stats-row">
        <div class="spr-stat">
          <span class="spr-stat-val">${s.rating ? s.rating.toFixed(1) : '—'}</span>
          <span class="spr-stat-lbl">Rating</span>
        </div>
        <div class="spr-stat">
          <span class="spr-stat-val">${s.jobs_done || 0}</span>
          <span class="spr-stat-lbl">Jobs done</span>
        </div>
        <div class="spr-stat">
          <span class="spr-stat-val">${rateStr}</span>
          <span class="spr-stat-lbl">Rate</span>
        </div>
      </div>

      <div class="spr-avail">
        ${availDot}
        ${s.available_from || (s.available ? 'Available' : 'Busy')}
      </div>

      <div class="spr-tags">${tags}</div>

      <div class="spr-card-actions">
        <button class="spr-btn-book" onclick="openSprayerModal('${s.email}','${s.name.replace(/'/g,"\\'")}','${rateStr}','${s.initials}','${bg}','${fg}')">
          Book Now
        </button>
        ${s.phone ? `<a href="tel:${s.phone}" class="spr-btn-call">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M2 3C2 2.45 2.45 2 3 2H5.5L6.5 5L5 6C5.5 7 7 8.5 8 9L9 7.5L12 8.5V11C12 11.55 11.55 12 11 12C6 12 2 8 2 3Z" stroke="currentColor" stroke-width="1.3"/>
          </svg>
          Call
        </a>` : ''}
      </div>
    </div>`;
  }).join('');
}

// ── Open booking modal ────────────────────────────────────────
function openSprayerModal(email, name, rate, initials, avatarBg, avatarFg) {
  _sprayerBookingData.sprayer_email = email;
  _sprayerBookingData.sprayer_name  = name;
  _sprayerBookingData.rate          = rate;

  // Reset modal state
  document.getElementById('sprayerBookingForm').style.display = 'block';
  document.getElementById('sprayerSuccess').style.display     = 'none';
  document.getElementById('smError').style.display            = 'none';

  // Fill header
  document.getElementById('smAvatar').textContent           = initials;
  document.getElementById('smAvatar').style.background      = `#${avatarBg}`;
  document.getElementById('smAvatar').style.color           = `#${avatarFg}`;
  document.getElementById('smName').textContent             = name;
  document.getElementById('smMeta').textContent             = `Certified · ${rate}`;

  // Disease note
  const note = document.getElementById('smDiseaseNote');
  if (_sprayerBookingData.disease_label && _sprayerBookingData.disease_label !== 'Healthy') {
    note.textContent = `Diagnosed disease: ${_sprayerBookingData.disease_label} — the sprayer will be informed automatically.`;
    note.classList.add('show');
  } else {
    note.classList.remove('show');
  }

  // Reset confirm button
  const btn = document.getElementById('smConfirmBtn');
  btn.disabled = false;
  document.getElementById('smBtnTxt').textContent = 'Confirm Booking';

  document.getElementById('sprayerModalOverlay').style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

// ── Close modal ───────────────────────────────────────────────
function closeSprayerModal(e) {
  if (e && e.target !== document.getElementById('sprayerModalOverlay')) return;
  document.getElementById('sprayerModalOverlay').style.display = 'none';
  document.body.style.overflow = '';
}

// ── Submit booking ────────────────────────────────────────────
function submitBooking() {
  const location = document.getElementById('bk_location').value.trim();
  const date     = document.getElementById('bk_date').value;
  const time     = document.getElementById('bk_time').value;
  const acres    = document.getElementById('bk_acres').value;
  const notes    = document.getElementById('bk_notes').value.trim();
  const errEl    = document.getElementById('smError');

  // Validate
  if (!location) { showModalError('Please enter your farm location.'); return; }
  if (!date)     { showModalError('Please select a date.'); return; }
  if (!acres || parseFloat(acres) <= 0) { showModalError('Please enter a valid acreage.'); return; }

  const btn = document.getElementById('smConfirmBtn');
  btn.disabled = true;
  document.getElementById('smBtnTxt').textContent = 'Confirming...';
  errEl.style.display = 'none';

  const payload = {
    sprayer_email: _sprayerBookingData.sprayer_email,
    sprayer_name:  _sprayerBookingData.sprayer_name,
    farm_location: location,
    date:          date,
    time_slot:     time,
    acres:         parseFloat(acres),
    disease:       _sprayerBookingData.disease_label,
    notes:         notes,
  };

  fetch('/api/book-sprayer', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(payload),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        showModalError(data.error);
        btn.disabled = false;
        document.getElementById('smBtnTxt').textContent = 'Confirm Booking';
        return;
      }
      // Show success
      document.getElementById('sprayerBookingForm').style.display = 'none';
      document.getElementById('sprayerSuccess').style.display     = 'block';
      document.getElementById('smSuccessSub').textContent =
        `${data.sprayer_name} has been notified. They will call you to confirm the visit.`;
      document.getElementById('smRef').textContent = data.ref;
      // Refresh bookings strip
      loadMyBookings();
    })
    .catch(() => {
      showModalError('Network error. Please try again.');
      btn.disabled = false;
      document.getElementById('smBtnTxt').textContent = 'Confirm Booking';
    });
}

function showModalError(msg) {
  const el = document.getElementById('smError');
  el.textContent    = msg;
  el.style.display  = 'block';
}

// ── My Bookings strip ─────────────────────────────────────────
function loadMyBookings() {
  fetch('/api/my-bookings')
    .then(r => r.json())
    .then(data => {
      if (!data.bookings || data.bookings.length === 0) return;
      renderMyBookings(data.bookings);
    })
    .catch(() => {});
}

function renderMyBookings(bookings) {
  // Remove existing strip if any
  const existing = document.getElementById('myBookingsStrip');
  if (existing) existing.remove();

  const strip = document.createElement('div');
  strip.id        = 'myBookingsStrip';
  strip.className = 'my-bookings-strip';
  strip.innerHTML = `
    <div class="mb-strip-title">Your Bookings</div>
    ${bookings.slice(0, 3).map(b => `
    <div class="mb-item">
      <div class="mb-item-left">
        <span class="mb-item-sprayer">${b.sprayer_name}</span>
        <span class="mb-item-meta">${b.date} · ${b.time_slot} · ${b.acres} acres · ${b.disease || 'General'}</span>
      </div>
      <span class="mb-status ${b.status}">${b.status.charAt(0).toUpperCase() + b.status.slice(1)}</span>
    </div>`).join('')}
  `;

  const sec = document.getElementById('sprayerSec');
  if (sec) sec.appendChild(strip);
}