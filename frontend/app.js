/* ═══════════════════════════════════════════════════════
   ChurnGuard — Dashboard Logic v3.0
   ═══════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const form = $('#predict-form');
const submitBtn = $('#submitBtn');
const resultCard = $('#result-card');
const errorCard = $('#error-card');
const emptyState = $('#empty-state');
const errorText = $('#error-text');
const historyList = $('#history-list');
const recsCard = $('#recs-card');
const recsList = $('#recs-list');

const HISTORY_KEY = 'churnguard_history';
let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
let batchData = null;
let batchResults = null;

// ══════════════════════════════════════════════════════════
// Health Check & Stats Polling
// ══════════════════════════════════════════════════════════

async function checkHealth() {
  const dot = $('#status-dot');
  const label = $('#api-status');
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      dot.className = 'status-dot online';
      label.textContent = `API Online · ${formatUptime(data.uptime_seconds)}`;
    } else {
      throw new Error();
    }
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'API Offline';
  }
}

function formatUptime(seconds) {
  if (seconds < 60) return `${Math.floor(seconds)}s uptime`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m uptime`;
  return `${Math.floor(seconds / 3600)}h uptime`;
}

async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    if (res.ok) {
      const data = await res.json();
      animateCounter('#stat-total', data.total_predictions);
      animateCounter('#stat-churn', data.churn_count);
      animateCounter('#stat-stay', data.stay_count);
      $('#stat-rate').textContent = `${data.churn_rate}%`;
      $('#stat-latency').textContent = `${data.avg_latency_ms}ms`;
    }
  } catch { /* silent */ }
}

async function fetchModelInfo() {
  try {
    const res = await fetch('/api/model-info');
    if (res.ok) {
      const data = await res.json();
      const container = $('#model-info-content');
      if (!data.loaded) {
        container.innerHTML = '<p class="text-muted">⚠️ Models not loaded — run training pipeline first</p>';
        return;
      }
      const rows = [];
      if (data.grid) {
        rows.push({ label: 'GridSearch RF', value: `${data.grid.n_estimators || '?'} trees` });
      }
      if (data.bayesian) {
        rows.push({ label: 'Bayesian RF', value: `${data.bayesian.n_estimators || '?'} trees` });
      }
      rows.push({ label: 'Features', value: data.feature_count || '—' });
      rows.push({ label: 'Status', value: '✅ Ready' });

      container.innerHTML = rows.map(r =>
        `<div class="model-info-row"><span class="label">${r.label}</span><span class="value">${r.value}</span></div>`
      ).join('');
    }
  } catch {
    $('#model-info-content').innerHTML = '<p class="text-muted">Unable to fetch model info</p>';
  }
}

function animateCounter(selector, target) {
  const el = $(selector);
  const current = parseInt(el.textContent) || 0;
  if (current === target) return;
  
  const diff = target - current;
  const steps = 20;
  const inc = diff / steps;
  let step = 0;

  function tick() {
    step++;
    if (step >= steps) {
      el.textContent = target;
      return;
    }
    el.textContent = Math.round(current + inc * step);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// Init
checkHealth();
fetchStats();
fetchModelInfo();
setInterval(checkHealth, 30000);
setInterval(fetchStats, 15000);

// ══════════════════════════════════════════════════════════
// Tab Switching
// ══════════════════════════════════════════════════════════

$('#tab-single').addEventListener('click', () => switchTab('single'));
$('#tab-batch').addEventListener('click', () => switchTab('batch'));

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  $(`#tab-${tab}`).classList.add('active');
  $('#single-panel').hidden = tab !== 'single';
  $('#batch-panel').hidden = tab !== 'batch';
}

// ══════════════════════════════════════════════════════════
// Build Prediction Payload
// ══════════════════════════════════════════════════════════

function buildPayload() {
  const record = {
    tenure: Number($('#tenure').value),
    MonthlyCharges: Number($('#monthly').value),
    TotalCharges: Number($('#total').value),
    SeniorCitizen: Number($('#senior').value),
    Partner: $('#partner').value,
    Dependents: $('#dependents').value,
    MultipleLines: $('#multilines').value,
    InternetService: $('#internet').value,
    OnlineSecurity: $('#security').value,
    OnlineBackup: $('#backup').value,
    DeviceProtection: $('#protection').value,
    TechSupport: $('#techsupport').value,
    StreamingTV: $('#tv').value,
    StreamingMovies: $('#movies').value,
    Contract: $('#contract').value,
    PaperlessBilling: $('#paperless').value,
    PaymentMethod: $('#payment').value,
  };
  return {
    model_type: $('#modelType').value,
    records: [record],
  };
}

// ══════════════════════════════════════════════════════════
// Gauge Animation
// ══════════════════════════════════════════════════════════

function updateGauge(probability) {
  const pct = probability * 100;
  const circumference = 2 * Math.PI * 52;
  const offset = circumference * (1 - probability);
  const fill = $('#gauge-fill');
  const value = $('#gauge-value');
  const label = $('#gauge-label');

  fill.style.strokeDashoffset = offset;
  value.textContent = `${pct.toFixed(1)}%`;

  let color;
  if (pct > 70) {
    color = 'var(--danger)';
    label.textContent = 'HIGH RISK';
    value.style.color = 'var(--danger)';
  } else if (pct > 40) {
    color = 'var(--warning)';
    label.textContent = 'MEDIUM RISK';
    value.style.color = 'var(--warning)';
  } else {
    color = 'var(--success)';
    label.textContent = 'LOW RISK';
    value.style.color = 'var(--success)';
  }
  fill.style.stroke = color;
}

// ══════════════════════════════════════════════════════════
// Show Results
// ══════════════════════════════════════════════════════════

function showResult(data) {
  const pred = data.predictions[0];
  const proba = data.churn_probability[0];
  const riskLevel = data.risk_levels ? data.risk_levels[0] : (proba > 0.7 ? 'High' : proba > 0.4 ? 'Medium' : 'Low');
  const recommendations = data.recommendations ? data.recommendations[0] : [];

  // Model badge
  $('#result-model').textContent = data.model_type === 'bayesian' ? 'Bayesian RF' : 'GridSearch RF';

  // Prediction label
  const predEl = $('#result-pred');
  predEl.textContent = pred === 1 ? 'Will Churn' : 'Will Stay';
  predEl.className = `risk-detail__value ${pred === 1 ? 'churn' : 'safe'}`;

  // Risk level
  const riskEl = $('#result-risk');
  const riskClass = riskLevel === 'High' ? 'risk-high' : riskLevel === 'Medium' ? 'risk-medium' : 'risk-low';
  riskEl.textContent = riskLevel;
  riskEl.className = `risk-detail__value ${riskClass}`;

  // Gauge
  updateGauge(proba);

  // Show cards
  emptyState.hidden = true;
  errorCard.hidden = true;
  resultCard.hidden = false;

  // Recommendations
  if (recommendations.length > 0) {
    recsList.innerHTML = recommendations.map(r => `<li>${r}</li>`).join('');
    recsCard.hidden = false;
  } else {
    recsCard.hidden = true;
  }

  // History
  addToHistory(pred, proba, data.model_type);

  // Refresh stats
  fetchStats();
}

// ══════════════════════════════════════════════════════════
// History Management
// ══════════════════════════════════════════════════════════

function addToHistory(pred, proba, modelType) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    pred,
    proba: (proba * 100).toFixed(1),
    model: modelType === 'bayesian' ? 'Bayes' : 'Grid',
  };
  history.unshift(entry);
  if (history.length > 30) history = history.slice(0, 30);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historyList.innerHTML = '<p class="history-empty">No predictions recorded yet.</p>';
    return;
  }
  historyList.innerHTML = history.map((h) => `
    <div class="history-item">
      <div class="history-item__left">
        <span class="history-item__dot ${h.pred === 1 ? 'churn' : 'safe'}"></span>
        <span>${h.pred === 1 ? 'Churn' : 'Stay'}</span>
        <span class="history-item__info">${h.model} · ${h.time}</span>
      </div>
      <span class="history-item__prob">${h.proba}%</span>
    </div>
  `).join('');
}

renderHistory();

// Clear history
$('#clear-history').addEventListener('click', () => {
  history = [];
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

// ══════════════════════════════════════════════════════════
// Form Submission
// ══════════════════════════════════════════════════════════

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorCard.hidden = true;
  resultCard.hidden = true;
  recsCard.hidden = true;

  submitBtn.disabled = true;
  submitBtn.querySelector('.btn-text').textContent = 'Predicting…';
  submitBtn.querySelector('.btn-icon').textContent = '⏳';

  try {
    const payload = buildPayload();
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Request failed');
    }

    const data = await res.json();
    showResult(data);
  } catch (err) {
    errorText.textContent = err.message;
    errorCard.hidden = false;
    emptyState.hidden = true;
  } finally {
    submitBtn.disabled = false;
    submitBtn.querySelector('.btn-text').textContent = 'Predict Churn';
    submitBtn.querySelector('.btn-icon').textContent = '🔮';
  }
});

// ══════════════════════════════════════════════════════════
// Batch CSV Upload
// ══════════════════════════════════════════════════════════

const dropzone = $('#csv-dropzone');
const csvInput = $('#csv-input');
const csvPreview = $('#csv-preview');
const batchControls = $('#batch-controls');

dropzone.addEventListener('click', () => csvInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith('.csv')) {
    handleCSVFile(file);
  }
});

csvInput.addEventListener('change', () => {
  if (csvInput.files[0]) handleCSVFile(csvInput.files[0]);
});

function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    if (values.length !== headers.length) continue;
    const record = {};
    headers.forEach((h, idx) => {
      const val = values[idx];
      const num = Number(val);
      record[h] = isNaN(num) || val === '' ? val : num;
    });
    records.push(record);
  }
  return records;
}

function handleCSVFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const records = parseCSV(e.target.result);
    if (records.length === 0) {
      alert('Could not parse CSV. Please check the format.');
      return;
    }
    batchData = records;
    $('#csv-filename').textContent = file.name;
    $('#csv-rowcount').textContent = `${records.length} records`;
    
    // Show preview table (first 5 rows)
    const headers = Object.keys(records[0]);
    const previewRows = records.slice(0, 5);
    const tableHTML = `
      <table>
        <thead><tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr></thead>
        <tbody>${previewRows.map(r => 
          `<tr>${headers.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`
        ).join('')}</tbody>
      </table>
    `;
    $('#csv-table-wrap').innerHTML = tableHTML;
    
    dropzone.hidden = true;
    csvPreview.hidden = false;
    batchControls.hidden = false;
    $('#batch-results').hidden = true;
  };
  reader.readAsText(file);
}

$('#csv-clear').addEventListener('click', () => {
  batchData = null;
  batchResults = null;
  csvInput.value = '';
  dropzone.hidden = false;
  csvPreview.hidden = true;
  batchControls.hidden = true;
  $('#batch-results').hidden = true;
});

// ══════════════════════════════════════════════════════════
// Batch Prediction
// ══════════════════════════════════════════════════════════

$('#batchSubmitBtn').addEventListener('click', async () => {
  if (!batchData || batchData.length === 0) return;

  const btn = $('#batchSubmitBtn');
  btn.disabled = true;
  btn.querySelector('.btn-text').textContent = 'Processing…';

  try {
    const payload = {
      model_type: $('#batchModelType').value,
      records: batchData,
    };

    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Batch prediction failed');
    }

    const data = await res.json();
    batchResults = data;
    showBatchResults(data);
    fetchStats();
  } catch (err) {
    alert(`Batch prediction error: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.querySelector('.btn-text').textContent = 'Run Batch Prediction';
  }
});

function showBatchResults(data) {
  const churnCount = data.predictions.filter(p => p === 1).length;
  const stayCount = data.count - churnCount;
  const avgProba = (data.churn_probability.reduce((a, b) => a + b, 0) / data.count * 100).toFixed(1);

  // Summary cards
  $('#batch-summary').innerHTML = `
    <div class="batch-summary-item"><span class="num">${data.count}</span><span class="lbl">Total</span></div>
    <div class="batch-summary-item"><span class="num" style="color:var(--danger)">${churnCount}</span><span class="lbl">Churn</span></div>
    <div class="batch-summary-item"><span class="num" style="color:var(--success)">${stayCount}</span><span class="lbl">Stay</span></div>
    <div class="batch-summary-item"><span class="num">${avgProba}%</span><span class="lbl">Avg Risk</span></div>
  `;

  // Results table
  const rows = data.predictions.map((pred, i) => {
    const prob = (data.churn_probability[i] * 100).toFixed(1);
    const risk = data.risk_levels ? data.risk_levels[i] : (data.churn_probability[i] > 0.7 ? 'High' : data.churn_probability[i] > 0.4 ? 'Medium' : 'Low');
    const badge = pred === 1 ? '<span class="churn-badge churn">Churn</span>' : '<span class="churn-badge safe">Stay</span>';
    const custId = batchData[i]?.customerID || batchData[i]?.CustomerID || `#${i + 1}`;
    return `<tr><td>${custId}</td><td>${badge}</td><td>${prob}%</td><td>${risk}</td></tr>`;
  });

  $('#batch-table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Customer</th><th>Prediction</th><th>Probability</th><th>Risk</th></tr></thead>
      <tbody>${rows.join('')}</tbody>
    </table>
  `;

  $('#batch-results').hidden = false;
}

// ══════════════════════════════════════════════════════════
// Export Results to CSV
// ══════════════════════════════════════════════════════════

$('#export-btn').addEventListener('click', () => {
  if (!batchResults || !batchData) return;

  const headers = ['Customer', 'Prediction', 'Churn_Probability', 'Risk_Level'];
  const rows = batchResults.predictions.map((pred, i) => {
    const custId = batchData[i]?.customerID || batchData[i]?.CustomerID || `customer_${i + 1}`;
    return [
      custId,
      pred === 1 ? 'Churn' : 'Stay',
      batchResults.churn_probability[i].toFixed(4),
      batchResults.risk_levels ? batchResults.risk_levels[i] : '—',
    ].join(',');
  });

  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `churn_predictions_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});
