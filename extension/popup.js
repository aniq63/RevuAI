const SERVERS = {
  cloud: { name: 'Cloud', url: 'http://13.212.212.174:8000' },
  local: { name: 'Localhost', url: 'http://localhost:8000' }
};

let currentServerKey = 'cloud'; // Default to deployed cloud backend

function getApiBase() {
  return SERVERS[currentServerKey].url;
}

const TIMEOUT_MS = 5 * 60 * 1000;
let appState = { status: 'idle', result: null, scrapingMethod: 'new_reviews', currentAppId: '' };
let loadingTimers = [];
let activeReviewTab = 'positive';
const $ = id => document.getElementById(id);

function detectActivePlayStoreApp() {
  if (!chrome.tabs || !chrome.tabs.query) {
    const statusEl = $('app-detection-status');
    if (statusEl) statusEl.textContent = 'Automatic detection unavailable; paste the link manually';
    return;
  }
  chrome.tabs.query({ active: true, lastFocusedWindow: true }, tabs => {
    if (chrome.runtime.lastError || !tabs || !tabs[0] || !tabs[0].url) {
      const statusEl = $('app-detection-status');
      if (statusEl) statusEl.textContent = 'Paste the app ID or Play Store link manually';
      return;
    }
    const url = tabs[0].url;
    const appId = extractAppId(url);
    if (!appId) {
      const statusEl = $('app-detection-status');
      if (statusEl) statusEl.textContent = 'Open a Play Store app page or paste a link manually';
      return;
    }
    const input = $('app-id-input');
    if (input) input.value = appId;
    const statusEl = $('app-detection-status');
    if (statusEl) statusEl.textContent = 'Automatically detected from active Play Store tab';
  });
}

function isPlayStoreUrl(raw) {
  try {
    const url = new URL(raw);
    return (url.hostname === 'play.google.com' || url.hostname === 'market.android.com') &&
      url.pathname.includes('/store/apps/details') && Boolean(url.searchParams.get('id'));
  } catch (e) {
    return false;
  }
}

async function checkBackendHealth() {
  const badge = $('server-toggle-btn');
  const nameEl = $('server-name');
  const warnEl = $('backend-warn');

  if (nameEl) nameEl.textContent = SERVERS[currentServerKey].name;
  if (badge) {
    badge.className = 'server-badge' + (currentServerKey === 'local' ? ' local' : '');
  }

  try {
    const r = await fetch(getApiBase() + '/health', { signal: AbortSignal.timeout(4000) });
    const d = await r.json();
    if (d.status === 'ok') {
      if (warnEl) warnEl.classList.add('hidden');
      if (badge) badge.classList.remove('offline');
    } else {
      if (warnEl) warnEl.classList.remove('hidden');
      if (badge) badge.classList.add('offline');
    }
  } catch (e) {
    if (warnEl) warnEl.classList.remove('hidden');
    if (badge) badge.classList.add('offline');
  }
}

// Load saved server preference & check health
if (chrome.storage && chrome.storage.local) {
  chrome.storage.local.get(['selected_server'], (res) => {
    const savedServer = res && res.selected_server;
    currentServerKey = 'cloud';

    if (savedServer === 'local') {
      chrome.storage.local.set({ selected_server: 'cloud' });
    }

    checkBackendHealth();
  });
} else {
  currentServerKey = 'cloud';
  checkBackendHealth();
}

// Toggle server on badge click
const toggleBtn = $('server-toggle-btn');
if (toggleBtn) {
  toggleBtn.addEventListener('click', () => {
    currentServerKey = currentServerKey === 'cloud' ? 'local' : 'cloud';
    if (chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ selected_server: currentServerKey });
    }
    checkBackendHealth();
  });
}

detectActivePlayStoreApp();
const detectBtn = $('detect-app-button');
if (detectBtn) detectBtn.addEventListener('click', detectActivePlayStoreApp);

document.querySelectorAll('#method-pills .pill').forEach(pill => {
  pill.addEventListener('click', () => {
    if ($('analyze-btn').disabled) return;
    document.querySelectorAll('#method-pills .pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    appState.scrapingMethod = pill.dataset.value;
  });
});

document.querySelectorAll('.filter-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    chip.classList.toggle('active');
    chip.classList.toggle('purple-outline');
  });
});

function extractAppId(raw) {
  const t = raw.trim();
  if (!t) return '';
  try {
    if (isPlayStoreUrl(t)) {
      const id = new URL(t).searchParams.get('id');
      if (id) return id;
    }
  } catch (e) { }
  return t.includes('.') && !t.includes('/') && !t.includes(' ') ? t : '';
}

$('analyze-form').addEventListener('submit', async e => {
  e.preventDefault();
  const appId = extractAppId($('app-id-input').value);
  if (!appId) { showFormError('Enter a Play Store app id or paste the app Play Store link.'); return; }
  if (!appId.includes('.')) { showFormError('That does not look like a valid package id (e.g. com.supercell.clashofclans).'); return; }
  hideFormError();
  await runAnalysis({ appId, scrapingMethod: appState.scrapingMethod, forceRefresh: $('force-refresh-cb').checked });
});

function showFormError(msg) { const e = $('form-error'); e.textContent = msg; e.classList.remove('hidden'); }
function hideFormError() { $('form-error').classList.add('hidden'); }

async function analyzeApp({ appId, scrapingMethod, forceRefresh }) {
  const base = getApiBase();
  try {
    const res = await fetch(base + '/inference/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ app_id: appId, scraping_method: scrapingMethod, force_refresh: forceRefresh }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || ('Server error ' + res.status));
    }
    return await res.json();
  } catch (err) {
    if (err.name === 'TypeError' || (err.message && err.message.includes('fetch'))) {
      throw new Error(`Cannot connect to RevuAI backend (${SERVERS[currentServerKey].name} at ${base}). Please check that the server is running or toggle to the other server.`);
    }
    throw err;
  }
}

async function runAnalysis({ appId, scrapingMethod, forceRefresh }) {
  appState.currentAppId = appId;
  appState.status = 'loading';
  renderView();
  try {
    appState.result = await analyzeApp({ appId, scrapingMethod, forceRefresh });
    appState.status = 'done';
  } catch (err) {
    appState.status = 'error';
    $('api-error-msg').textContent = (err && err.message) || 'Something went wrong while analyzing this app.';
  } finally {
    stopLoadingTimers();
    renderView();
  }
}

$('btn-new-analysis').addEventListener('click', () => {
  appState.status = 'idle'; appState.result = null;
  stopLoadingTimers(); renderView();
});
$('btn-force-refresh').addEventListener('click', () => {
  if (!appState.result) return;
  runAnalysis({ appId: appState.result.app_id, scrapingMethod: appState.result.scraping_method, forceRefresh: true });
});

function renderView() {
  const { status, result } = appState;
  $('analyze-form').classList.toggle('hidden', status === 'done');
  setFormDisabled(status === 'loading');
  $('loading-screen').classList.toggle('hidden', status !== 'loading');
  if (status === 'loading') { $('loading-app-id').textContent = appState.currentAppId; startLoadingAnimation(); }
  $('api-error-card').classList.toggle('hidden', status !== 'error');
  $('results-view').classList.toggle('hidden', status !== 'done');
  if (status === 'done' && result) renderResults(result);
}

function setFormDisabled(d) {
  $('app-id-input').disabled = d;
  $('analyze-btn').disabled = d;
  $('force-refresh-cb').disabled = d;
  document.querySelectorAll('#method-pills .pill').forEach(p => { p.style.pointerEvents = d ? 'none' : ''; });
  $('btn-spinner').classList.toggle('hidden', !d);
  $('btn-label').textContent = d ? 'Analyzing...' : 'Analyze Reviews';
}

const STAGES = [
  { key: 'scrape', label: 'Scraping Play Store reviews', weight: 12, logs: ['Connecting to Play Store...', 'Fetching review pages...', 'Collected batch of reviews'] },
  { key: 'predict', label: 'Running sentiment model', weight: 18, logs: ['Vectorizing review text (TF-IDF)...', 'Scoring reviews with Champion model...', 'Predictions complete'] },
  { key: 'summary', label: 'Summarizing sentiment distribution', weight: 5, logs: ['Computing sentiment percentages...', 'Selecting top reviews...'] },
  { key: 'cluster', label: 'Clustering topics from reviews', weight: 50, logs: ['Embedding reviews (MiniLM)...', 'Reducing dimensions (UMAP)...', 'Finding clusters (HDBSCAN)...', 'Extracting keywords (KeyBERT)...'] },
  { key: 'llm', label: 'Generating LLM insight', weight: 15, logs: ['Sending summary to LLM...', 'Writing recommendations...'] },
];
const TOTAL_W = STAGES.reduce((s, st) => s + st.weight, 0);
let si = 0, li = 0, prog = 2;

function stopLoadingTimers() { loadingTimers.forEach(clearTimeout); loadingTimers = []; }

function startLoadingAnimation() {
  stopLoadingTimers(); si = 0; li = 0; prog = 2; renderStages(); updateProg();
  const dur = STAGES.map(s => (s.weight / TOTAL_W) * 55000);
  function goToStage(idx) {
    if (idx >= STAGES.length) return;
    si = idx; li = 0; renderStages();
    const lt = setInterval(() => { li = Math.min(li + 1, STAGES[idx].logs.length - 1); renderStages(); }, 1800);
    loadingTimers.push(lt);
    const st = setTimeout(() => { clearInterval(lt); goToStage(idx + 1); }, dur[idx]);
    loadingTimers.push(st);
  }
  goToStage(0);
  const pt = setInterval(() => { prog = prog < 92 ? prog + (92 - prog) * 0.04 : prog; updateProg(); }, 400);
  loadingTimers.push(pt);
}

function updateProg() { $('progress-fill').style.width = prog + '%'; }

function renderStages() {
  const c = $('stage-list'); c.innerHTML = '';
  STAGES.forEach((stage, idx) => {
    const isDone = idx < si, isActive = idx === si;
    const el = document.createElement('div');
    el.className = 'stage' + (isActive ? ' active' : '') + (isDone ? ' done' : '');
    el.innerHTML = '<span class="stage-icon">' + (isDone ? '&#10003;' : '') + '</span>' + stage.label;
    c.appendChild(el);
    if (isActive) {
      const le = document.createElement('div');
      le.className = 'stage-log';
      le.textContent = '> ' + (stage.logs[li] || '');
      c.appendChild(le);
    }
  });
}

const METHOD_LABELS = { new_reviews: 'Newest reviews', relevant_reviews: 'Most relevant', all_reviews: 'All reviews' };

function renderResults(r) {
  $('res-app-title').textContent = r.app_id;
  const cacheBadge = r.cache_hit ? '<span class="cache-badge">Served from cache</span>' : '';
  $('res-meta-pills').innerHTML =
    '<span class="meta-pill">' + (METHOD_LABELS[r.scraping_method] || r.scraping_method) + '</span>' +
    '<span class="meta-pill">' + Number(r.review_count).toLocaleString() + ' reviews</span>' +
    '<span class="meta-pill">' + r.execution_time_seconds + 's</span>' +
    cacheBadge;
  renderSentiment(r.sentiment_percentages, r.review_count);
  renderTopicChart(r.topic_summary);
  renderInsight(r.llm_insight, r.insight_generated_at);
  renderTopReviews(r.top_reviews);
}

function renderSentiment(pcts, count) {
  $('sentiment-count').textContent = '\u00b7 ' + Number(count).toLocaleString() + ' reviews analyzed';
  const COLORS = { positive: '#22c55e', neutral: '#6b7280', negative: '#ef4444', unknown: '#f59e0b' };
  const LABELS = { positive: 'Positive', neutral: 'Neutral', negative: 'Negative', unknown: 'Unclassified' };
  const keys = ['positive', 'neutral', 'negative', 'unknown'].filter(k => k in pcts);
  $('sentiment-grid').innerHTML = keys.map(k =>
    '<div class="sentiment-card">' +
    '<div class="sent-pct" style="color:' + COLORS[k] + '">' + pcts[k] + '%</div>' +
    '<div class="sent-label">' + LABELS[k] + '</div>' +
    '<div class="bar"><div class="bar-fill" style="width:' + pcts[k] + '%;background:' + COLORS[k] + '"></div></div>' +
    '</div>'
  ).join('');
}

function renderTopicChart(ts) {
  const COLORS = { negative: '#EF4444', neutral: '#6B7280', positive: '#22C55E' };
  const ORDER = ['negative', 'neutral', 'positive'];
  const TITLES = { negative: 'Negative', neutral: 'Neutral', positive: 'Positive' };
  const shorten = t => t.length > 45 ? t.slice(0, 45) + '\u2026' : t;
  const hasData = ORDER.some(s => (ts[s] || []).length > 0);
  $('topic-chart-container').classList.toggle('hidden', !hasData);
  $('topic-no-data').classList.toggle('hidden', hasData);
  if (!hasData) return;
  const chart = document.createElement('div');
  chart.className = 'topic-chart';
  ORDER.forEach(sentiment => {
    const items = ts[sentiment] || [];
    const group = document.createElement('section');
    const title = document.createElement('div');
    title.className = 'topic-group-title';
    title.style.color = COLORS[sentiment];
    title.textContent = TITLES[sentiment];
    group.appendChild(title);
    const maxPct = Math.max(...items.map(item => Number(item.percentage) || 0), 1);
    items.forEach(item => {
      const row = document.createElement('div');
      row.className = 'topic-item';
      row.innerHTML = '<div class="topic-label"><span class="topic-name"></span><span class="topic-value"></span></div><div class="topic-track"><div class="topic-bar"></div></div>';
      row.querySelector('.topic-name').textContent = shorten(String(item.topic));
      row.querySelector('.topic-value').textContent = item.percentage + '% (' + item.count + ')';
      row.querySelector('.topic-bar').style.cssText = 'width:' + ((Number(item.percentage) || 0) / maxPct * 100) + '%;background:' + COLORS[sentiment];
      group.appendChild(row);
    });
    chart.appendChild(group);
  });
  $('topic-chart-container').replaceChildren(chart);
}

function simpleMarkdown(md) {
  return md
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>')
    .split(/\n\n+/)
    .map(b => b.startsWith('<') ? b : '<p>' + b.replace(/\n/g, '<br>') + '</p>')
    .join('\n');
}

function renderInsight(insight, generatedAt) {
  const sec = $('insight-section');
  if (!insight) { sec.classList.add('hidden'); return; }
  sec.classList.remove('hidden');
  $('insight-body').innerHTML = simpleMarkdown(insight);
  $('insight-ts').textContent = generatedAt ? 'Generated at ' + new Date(generatedAt).toLocaleString() : '';
}

function renderTopReviews(topReviews) {
  const sec = $('reviews-section');
  const COLORS = { positive: '#22c55e', neutral: '#6b7280', negative: '#ef4444' };
  const available = ['positive', 'neutral', 'negative'].filter(s => (topReviews[s] || []).length > 0);
  if (available.length === 0) { sec.classList.add('hidden'); return; }
  sec.classList.remove('hidden');
  if (!available.includes(activeReviewTab)) activeReviewTab = available[0];
  function renderTabs() {
    $('review-tabs').innerHTML = available.map(s => {
      const isActive = activeReviewTab === s;
      const style = isActive ? 'border-color:' + COLORS[s] : '';
      return '<div class="pill' + (isActive ? ' active' : '') + '" data-s="' + s + '" style="' + style + '">' + s[0].toUpperCase() + s.slice(1) + '</div>';
    }).join('');
    $('review-tabs').querySelectorAll('.pill').forEach(p => {
      p.addEventListener('click', () => { activeReviewTab = p.dataset.s; renderTabs(); renderReviewList(); });
    });
  }
  function renderReviewList() {
    $('review-list').innerHTML = (topReviews[activeReviewTab] || []).map(r =>
      '<div class="review-card">' +
      '<div>' + escHtml(r.content) + '</div>' +
      '<div class="thumb">&#128077; ' + r.thumbsUpCount + ' found this helpful</div>' +
      '</div>'
    ).join('');
  }
  renderTabs();
  renderReviewList();
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

renderView();
