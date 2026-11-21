// Intelligent Student Complaint Classification & Priority Prediction System
// Frontend Application JavaScript

// Cached Analytics Payload for dynamic theme switching
let cachedAnalyticsData = null;
let categoryChartInstance = null;
let priorityChartInstance = null;
let comparisonChartInstance = null;

// Theme Controller
function initTheme() {
  const savedTheme = localStorage.getItem("eduresolve_theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const initialTheme = savedTheme || (prefersDark ? "dark" : "light");

  applyTheme(initialTheme);

  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      applyTheme(newTheme);
      localStorage.setItem("eduresolve_theme", newTheme);

      // Re-render charts with updated theme palette if already loaded
      if (cachedAnalyticsData) {
        renderAllCharts(cachedAnalyticsData);
      }
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const themeIcon = document.getElementById("theme-icon");
  const toggleBtn = document.getElementById("theme-toggle");

  if (themeIcon) {
    themeIcon.textContent = theme === "dark" ? "☀️" : "🌙";
  }
  if (toggleBtn) {
    toggleBtn.setAttribute("title", theme === "dark" ? "Switch to Light Theme" : "Switch to Dark Theme");
  }
}

// Helpers for Chart Colors based on Theme
function getThemeChartColors() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    textColor: isDark ? "#9ca3af" : "#475569",
    gridColor: isDark ? "#1f2937" : "#f1f5f9",
    doughnutBorder: isDark ? "#111827" : "#ffffff"
  };
}

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Theme
  initTheme();

  // Navigation Tabs
  const navBtns = document.querySelectorAll(".nav-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  function switchTab(targetTab) {
    navBtns.forEach((b) => b.classList.remove("active"));
    tabPanes.forEach((p) => p.classList.remove("active"));

    const activeBtn = document.querySelector(`.nav-btn[data-tab="${targetTab}"]`);
    if (activeBtn) activeBtn.classList.add("active");

    const activePane = document.getElementById(targetTab);
    if (activePane) activePane.classList.add("active");

    if (targetTab === "tab-analytics") {
      loadAnalyticsData();
    }
  }

  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      switchTab(targetTab);
    });
  });

  // Clicking brand logo navigates to main analyzer
  const brandSection = document.querySelector(".brand-section");
  if (brandSection) {
    brandSection.addEventListener("click", () => {
      switchTab("tab-analyzer");
    });
  }

  // Backend Health Check
  checkBackendHealth();

  // Single Complaint Analyzer Elements
  const complaintForm = document.getElementById("complaint-form");
  const complaintInput = document.getElementById("complaint-input");
  const charCounter = document.getElementById("char-counter");
  const submitBtn = document.getElementById("submit-btn");
  const clearBtn = document.getElementById("clear-btn");
  const presetChips = document.querySelectorAll(".preset-chip");

  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsDisplay = document.getElementById("results-display");

  // Character Counter
  complaintInput.addEventListener("input", () => {
    const len = complaintInput.value.length;
    charCounter.textContent = `${len} / 1000 characters`;
  });

  // Preset Chips
  presetChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const q = chip.getAttribute("data-query");
      complaintInput.value = q;
      charCounter.textContent = `${q.length} / 1000 characters`;
      analyzeComplaint(q);
    });
  });

  // Clear Button
  clearBtn.addEventListener("click", () => {
    complaintInput.value = "";
    charCounter.textContent = "0 / 1000 characters";
    emptyState.style.display = "block";
    loadingState.style.display = "none";
    resultsDisplay.style.display = "none";
    document.getElementById("analysis-time").textContent = "Waiting for input";
  });

  // Form Submit
  complaintForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = complaintInput.value.trim();
    if (text) {
      analyzeComplaint(text);
    }
  });

  // Probability Breakdown Toggle
  const toggleProbsBtn = document.getElementById("toggle-probs-btn");
  const probBreakdownContent = document.getElementById("prob-breakdown-content");
  const probToggleIcon = document.getElementById("prob-toggle-icon");
  let probsVisible = true;

  if (toggleProbsBtn) {
    toggleProbsBtn.addEventListener("click", () => {
      probsVisible = !probsVisible;
      probBreakdownContent.style.display = probsVisible ? "flex" : "none";
      probToggleIcon.textContent = probsVisible ? "▼" : "▶";
    });
  }

  // Batch Prediction Elements
  const batchTextarea = document.getElementById("batch-textarea");
  const runBatchBtn = document.getElementById("run-batch-btn");
  const loadTestBatchBtn = document.getElementById("load-test-batch-btn");
  const batchStatusText = document.getElementById("batch-status-text");
  const batchResultsCard = document.getElementById("batch-results-card");
  const batchTbody = document.getElementById("batch-tbody");
  const batchCount = document.getElementById("batch-count");

  if (loadTestBatchBtn) {
    loadTestBatchBtn.addEventListener("click", loadSampleBatch);
  }

  if (runBatchBtn) {
    runBatchBtn.addEventListener("click", executeBatchPrediction);
  }
});

// API Base URL
const API_BASE = "";

// Check Backend Health
async function checkBackendHealth() {
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("api-status-text");

  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("Status code: " + res.status);
    const data = await res.json();
    if (data.status === "healthy") {
      statusDot.className = "status-dot online";
      statusText.textContent = `Backend: Connected (Vocab: ${data.vocabulary_size})`;
    } else {
      statusDot.className = "status-dot offline";
      statusText.textContent = "Backend: Degraded";
    }
  } catch (err) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Backend: Disconnected";
  }
}

// Single Complaint Prediction
async function analyzeComplaint(text) {
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsDisplay = document.getElementById("results-display");
  const submitBtn = document.getElementById("submit-btn");
  const analysisTime = document.getElementById("analysis-time");

  emptyState.style.display = "none";
  loadingState.style.display = "block";
  resultsDisplay.style.display = "none";
  submitBtn.disabled = true;

  const startTime = performance.now();

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ complaint: text })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned error ${res.status}`);
    }

    const data = await res.json();
    const duration = Math.round(performance.now() - startTime);
    analysisTime.textContent = `Inference completed in ${duration} ms`;

    renderSingleResult(data);
  } catch (err) {
    alert("Prediction Error: " + err.message);
    emptyState.style.display = "block";
  } finally {
    loadingState.style.display = "none";
    submitBtn.disabled = false;
  }
}

// Render Single Prediction Result
function renderSingleResult(data) {
  const resultsDisplay = document.getElementById("results-display");
  resultsDisplay.style.display = "flex";

  // Category
  document.getElementById("res-category").textContent = data.predicted_category;
  const catPct = Math.round(data.category_confidence * 100);
  document.getElementById("res-category-conf").textContent = `${catPct}%`;
  document.getElementById("res-category-bar").style.width = `${catPct}%`;

  // Priority Card
  const priorityCard = document.getElementById("priority-card");
  priorityCard.className = `badge-card priority-${data.predicted_priority}`;
  document.getElementById("res-priority").textContent = data.predicted_priority;

  const prioIcon = document.getElementById("priority-icon");
  if (data.predicted_priority === "High") prioIcon.textContent = "🔥 Urgent";
  else if (data.predicted_priority === "Medium") prioIcon.textContent = "⏳ Standard";
  else prioIcon.textContent = "ℹ️ Routine";

  const prioPct = Math.round(data.priority_confidence * 100);
  document.getElementById("res-priority-conf").textContent = `${prioPct}%`;
  const prioBar = document.getElementById("res-priority-bar");
  prioBar.className = `progress-fill priority-${data.predicted_priority}`;
  prioBar.style.width = `${prioPct}%`;

  // SLA & Action
  document.getElementById("res-sla").textContent = data.resolution_sla;
  document.getElementById("res-action").textContent = `Recommended Protocol: ${data.suggested_action}`;

  // Keywords
  const kwContainer = document.getElementById("res-keywords");
  kwContainer.innerHTML = "";
  if (data.contributing_keywords && data.contributing_keywords.length > 0) {
    data.contributing_keywords.forEach((kw) => {
      const pill = document.createElement("span");
      pill.className = "kw-pill";
      pill.textContent = `#${kw}`;
      kwContainer.appendChild(pill);
    });
  } else {
    kwContainer.innerHTML = `<span style="font-size: 0.8rem; color: var(--text-muted);">Standard lexical pattern</span>`;
  }

  // Explanation
  document.getElementById("res-explanation").textContent = data.explanation;

  // Probability Breakdown List
  const probList = document.getElementById("prob-breakdown-content");
  probList.innerHTML = "";
  if (data.priority_probabilities) {
    Object.entries(data.priority_probabilities).forEach(([cls, prob]) => {
      const pct = Math.round(prob * 100);
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span style="width: 75px; font-weight: 700;">${cls}:</span>
        <div class="prob-bar">
          <div class="prob-fill priority-${cls}" style="width: ${pct}%;"></div>
        </div>
        <span style="width: 45px; text-align: right; font-weight: 700;">${pct}%</span>
      `;
      probList.appendChild(row);
    });
  }
}

// Load Samples for Batch Tester
async function loadSampleBatch() {
  const samples = [
    "Internship deadline today, need NOC approval.",
    "My fee payment failed but amount deducted.",
    "I cannot submit my exam form and today is the last date.",
    "How to apply for revaluation?",
    "How can I apply for hostel change?",
    "Please update my contact number.",
    "WiFi password for campus?",
    "What are library timings?",
    "How to join sports club?",
    "Is there any AI workshop this month?"
  ];
  document.getElementById("batch-textarea").value = samples.join("\n");
  document.getElementById("batch-status-text").textContent = "10 queries loaded into batch queue.";
}

// Execute Batch Prediction
async function executeBatchPrediction() {
  const textarea = document.getElementById("batch-textarea");
  const runBtn = document.getElementById("run-batch-btn");
  const statusText = document.getElementById("batch-status-text");
  const resultsCard = document.getElementById("batch-results-card");
  const tbody = document.getElementById("batch-tbody");
  const countSpan = document.getElementById("batch-count");

  const lines = textarea.value
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    alert("Please enter at least one complaint to run batch prediction.");
    return;
  }

  runBtn.disabled = true;
  statusText.textContent = `Processing ${lines.length} complaints...`;

  try {
    const res = await fetch(`${API_BASE}/api/predict-batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ complaints: lines })
    });

    if (!res.ok) throw new Error("Batch request failed with status: " + res.status);
    const data = await res.json();

    countSpan.textContent = data.total;
    tbody.innerHTML = "";

    data.predictions.forEach((item, idx) => {
      const tr = document.createElement("tr");

      let prioClass = "priority-Low";
      if (item.predicted_priority === "High") prioClass = "priority-High";
      else if (item.predicted_priority === "Medium") prioClass = "priority-Medium";

      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td style="max-width: 320px; font-weight: 500;">${item.complaint}</td>
        <td><span style="color: var(--primary); font-weight: 700;">${item.predicted_category}</span></td>
        <td>${Math.round(item.category_confidence * 100)}%</td>
        <td>
          <span style="display: inline-block; padding: 0.2rem 0.65rem; border-radius: 6px; font-weight: 800; font-size: 0.75rem;" class="badge-card ${prioClass}">
            ${item.predicted_priority}
          </span>
        </td>
        <td>${Math.round(item.priority_confidence * 100)}%</td>
        <td style="font-size: 0.8rem; color: var(--sla-text); font-weight: 600;">${item.resolution_sla.split(" (")[0]}</td>
      `;
      tbody.appendChild(tr);
    });

    resultsCard.style.display = "block";
    statusText.textContent = `Successfully processed ${data.total} complaints!`;
  } catch (err) {
    alert("Batch Prediction Error: " + err.message);
    statusText.textContent = "Error processing batch.";
  } finally {
    runBtn.disabled = false;
  }
}

// Load Analytics Data from Backend
async function loadAnalyticsData() {
  if (cachedAnalyticsData) {
    renderAllCharts(cachedAnalyticsData);
    return;
  }

  try {
    const res = await fetch(`${API_BASE}/api/analytics`);
    if (!res.ok) throw new Error("Analytics endpoint error: " + res.status);
    const data = await res.json();
    cachedAnalyticsData = data;
    renderAllCharts(data);
  } catch (err) {
    console.error("Failed to load analytics:", err);
  }
}

function renderAllCharts(data) {
  renderCategoryChart(data.dataset_summary.category_distribution_train);
  renderPriorityChart(data.dataset_summary.priority_distribution_train);
  renderComparisonChart(data.model_comparison);
  renderConfusionMatrix(data.confusion_matrices.Priority);
}

function renderCategoryChart(catDist) {
  const ctx = document.getElementById("categoryChart");
  if (!ctx) return;

  const { textColor, gridColor } = getThemeChartColors();
  const labels = Object.keys(catDist);
  const counts = Object.values(catDist);

  if (categoryChartInstance) categoryChartInstance.destroy();
  categoryChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Complaint Count",
          data: counts,
          backgroundColor: [
            "#3b82f6",
            "#6366f1",
            "#8b5cf6",
            "#ec4899",
            "#f97316",
            "#10b981",
            "#06b6d4"
          ],
          borderRadius: 6
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { color: textColor },
          grid: { color: gridColor }
        },
        x: {
          grid: { display: false },
          ticks: {
            color: textColor,
            font: { size: 10 },
            callback: function(val, index) {
              const label = this.getLabelForValue(val);
              return label.length > 15 ? label.substr(0, 15) + "..." : label;
            }
          }
        }
      }
    }
  });
}

function renderPriorityChart(prioDist) {
  const ctx = document.getElementById("priorityChart");
  if (!ctx) return;

  const { textColor, doughnutBorder } = getThemeChartColors();
  const labels = Object.keys(prioDist);
  const counts = Object.values(prioDist);

  if (priorityChartInstance) priorityChartInstance.destroy();
  priorityChartInstance = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [
        {
          data: counts,
          backgroundColor: ["#dc2626", "#d97706", "#059669"],
          borderWidth: 2,
          borderColor: doughnutBorder
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: textColor }
        }
      }
    }
  });
}

function renderComparisonChart(modelComp) {
  const ctx = document.getElementById("comparisonChart");
  if (!ctx) return;

  const { textColor, gridColor } = getThemeChartColors();
  const candidates = modelComp.Priority.candidates;
  const modelNames = Object.keys(candidates);
  const accuracies = modelNames.map((m) => candidates[m].cv_accuracy * 100);
  const f1Scores = modelNames.map((m) => candidates[m].cv_f1 * 100);
  const precisions = modelNames.map((m) => candidates[m].cv_precision * 100);
  const recalls = modelNames.map((m) => candidates[m].cv_recall * 100);

  if (comparisonChartInstance) comparisonChartInstance.destroy();
  comparisonChartInstance = new Chart(ctx, {
    type: "bar",
    data: {
      labels: modelNames,
      datasets: [
        { label: "Accuracy (%)", data: accuracies, backgroundColor: "#2563eb", borderRadius: 4 },
        { label: "Weighted F1 (%)", data: f1Scores, backgroundColor: "#10b981", borderRadius: 4 },
        { label: "Weighted Precision (%)", data: precisions, backgroundColor: "#f59e0b", borderRadius: 4 },
        { label: "Weighted Recall (%)", data: recalls, backgroundColor: "#8b5cf6", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 95,
          max: 100,
          ticks: { color: textColor },
          grid: { color: gridColor }
        },
        x: {
          grid: { display: false },
          ticks: { color: textColor }
        }
      },
      plugins: {
        legend: {
          position: "top",
          labels: { color: textColor }
        }
      }
    }
  });
}

function renderConfusionMatrix(cmData) {
  const container = document.getElementById("priority-cm-container");
  if (!container || !cmData) return;

  const labels = cmData.labels;
  const matrix = cmData.matrix;

  let html = `
    <table class="matrix-table">
      <thead>
        <tr>
          <th>Actual \\ Predicted</th>
          ${labels.map((l) => `<th>${l}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
  `;

  labels.forEach((actualLabel, i) => {
    html += `<tr><th>${actualLabel}</th>`;
    labels.forEach((predLabel, j) => {
      const val = matrix[i][j];
      const isMatch = i === j;
      const cellClass = isMatch ? "matrix-cell-match" : (val > 0 ? "" : "matrix-cell-zero");
      html += `<td class="${cellClass}">${val}</td>`;
    });
    html += `</tr>`;
  });

  html += `</tbody></table>`;
  container.innerHTML = html;
}
