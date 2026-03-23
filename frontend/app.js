/* ═══════════════════════════════════════════════════════
   Churn Prediction Dashboard — App Logic v2.0
   ═══════════════════════════════════════════════════════ */

const $ = (sel) => document.querySelector(sel);
const form = $('#predict-form');
const submitBtn = $('#submitBtn');
const resultCard = $('#result-card');
const errorCard = $('#error-card');
const emptyState = $('#empty-state');
const errorText = $('#error-text');
const historyList = $('#history-list');

const HISTORY_KEY = 'churn_history';
let history = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');

// ── Health Check ──────────────────────────────────────────

async function checkHealth() {
  const dot = $('.status-dot');
  const label = $('#api-status');
  try {
    const res = await fetch('/health');
    if (res.ok) {
      dot.className = 'status-dot online';
      label.textContent = 'API Online';
    } else {
      throw new Error();
    }
  } catch {
    dot.className = 'status-dot offline';
    label.textContent = 'API Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── Build Prediction Payload ──────────────────────────────

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

// ── Gauge Animation ───────────────────────────────────────

function updateGauge(probability) {
  const pct = probability * 100;
  const circumference = 2 * Math.PI * 52; // r=52
  const offset = circumference * (1 - probability);
  const fill = $('#gauge-fill');
  const value = $('#gauge-value');
  const label = $('#gauge-label');

  fill.style.strokeDashoffset = offset;
  value.textContent = `${pct.toFixed(1)}%`;

  if (pct > 70) {
    fill.style.stroke = 'var(--danger)';
    label.textContent = 'HIGH RISK';
  } else if (pct > 40) {
    fill.style.stroke = 'var(--warning)';
    label.textContent = 'MEDIUM RISK';
  } else {
    fill.style.stroke = 'var(--success)';
    label.textContent = 'LOW RISK';
  }
}

// ── Update Result UI ──────────────────────────────────────

function showResult(data) {
  const pred = data.predictions[0];
  const proba = data.churn_probability[0];
  const pct = (proba * 100).toFixed(1);

  // Model badge
  $('#result-model').textContent = data.model_type === 'bayesian' ? 'Bayesian RF' : 'GridSearch RF';

  // Prediction label
  const predEl = $('#result-pred');
  predEl.textContent = pred === 1 ? 'Will Churn' : 'Will Stay';
  predEl.className = `risk-detail__value ${pred === 1 ? 'churn' : 'safe'}`;

  // Risk level
  const riskEl = $('#result-risk');
  let riskLevel, riskClass;
  if (proba > 0.7) { riskLevel = 'High'; riskClass = 'risk-high'; }
  else if (proba > 0.4) { riskLevel = 'Medium'; riskClass = 'risk-medium'; }
  else { riskLevel = 'Low'; riskClass = 'risk-low'; }
  riskEl.textContent = riskLevel;
  riskEl.className = `risk-detail__value ${riskClass}`;

  // Gauge
  updateGauge(proba);

  // Show card
  emptyState.hidden = true;
  errorCard.hidden = true;
  resultCard.hidden = false;

  // Add to history
  addToHistory(pred, proba, data.model_type);
}

// ── History Management ────────────────────────────────────

function addToHistory(pred, proba, modelType) {
  const entry = {
    time: new Date().toLocaleTimeString(),
    pred,
    proba: (proba * 100).toFixed(1),
    model: modelType === 'bayesian' ? 'Bayes' : 'Grid',
  };
  history.unshift(entry);
  if (history.length > 20) history = history.slice(0, 20);
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

// ── Form Submission ───────────────────────────────────────

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errorCard.hidden = true;
  resultCard.hidden = true;

  submitBtn.disabled = true;
  submitBtn.innerHTML = '<span class="btn-icon">⏳</span> Predicting…';

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
    submitBtn.innerHTML = '<span class="btn-icon">🔮</span> Predict Churn';
  }
});
