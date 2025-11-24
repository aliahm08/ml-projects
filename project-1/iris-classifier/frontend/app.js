const DATA_URL = "../backend/output/predictions.json";

document.addEventListener("DOMContentLoaded", initCaseStudy);

async function initCaseStudy() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Failed to load ${DATA_URL} (${response.status})`);
    }
    const payload = await response.json();
    const report = payload.classification_report || {};
    const perClass = report.per_class || [];
    const extremes = computeClassExtremes(perClass);
    renderHeroStats(payload.metadata, extremes);
    renderTrainingInsights(payload.metadata, extremes);
    renderClassificationTable(report);
    renderDigitCards(perClass, payload.samples);
    renderTakeaways(payload.metadata, extremes, report);
  } catch (error) {
    const main = document.querySelector("main");
    main.innerHTML = `
      <section class="material-card">
        <h2>Unable to load case-study data</h2>
        <p>${error.message}</p>
        <p>Run <code>python backend/pipeline.py</code> and serve the repo with <code>python -m http.server</code>.</p>
      </section>
    `;
    console.error(error);
  }
}

function computeClassExtremes(perClass) {
  if (!perClass || perClass.length === 0) {
    return {
      best: null,
      worst: null,
    };
  }
  const best = perClass.reduce((acc, entry) =>
    entry.recall > acc.recall ? entry : acc
  );
  const worst = perClass.reduce((acc, entry) =>
    entry.recall < acc.recall ? entry : acc
  );
  return { best, worst };
}

function renderHeroStats(meta, extremes) {
  const formatter = new Intl.NumberFormat();
  document.getElementById("stat-total").textContent = formatter.format(
    meta.total_samples
  );
  document.getElementById("stat-accuracy").textContent = `${(
    meta.accuracy * 100
  ).toFixed(2)}%`;
  document.getElementById("stat-generated").textContent = meta.generated_at;
  document.getElementById("stat-best-label").textContent = extremes.best
    ? `Digit ${extremes.best.label}`
    : "–";
  document.getElementById("stat-worst-label").textContent = extremes.worst
    ? `Digit ${extremes.worst.label}`
    : "–";
}

function renderTrainingInsights(meta, extremes) {
  const accuracyCopy = `After ${meta.total_samples.toLocaleString()} test images, the classifier settled at ${(meta.accuracy * 100).toFixed(2)}% accuracy using the saved scaler + model artifacts.`;
  document.getElementById("insight-accuracy").textContent = accuracyCopy;

  const bestCopy = extremes.best
    ? `Digits shaped like ${extremes.best.label}s are rarely misread: ${(extremes.best.recall * 100).toFixed(1)}% of those samples land in the correct bucket.`
    : "No per-class metrics available yet.";
  document.getElementById("insight-best").textContent = bestCopy;

  const worstCopy = extremes.worst
    ? `The most troublesome label is ${extremes.worst.label} with ${(extremes.worst.recall * 100).toFixed(
        1
      )}% accuracy, often confused with neighboring shapes.`
    : "No hotspots detected.";
  document.getElementById("insight-worst").textContent = worstCopy;
}

function renderClassificationTable(report) {
  const tbody = document.querySelector("#report-table tbody");
  tbody.innerHTML = "";

  const perClass = report.per_class || [];
  perClass.forEach((entry) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${entry.label}</td>
      <td>${(entry.precision * 100).toFixed(2)}%</td>
      <td>${(entry.recall * 100).toFixed(2)}%</td>
      <td>${(entry.f1 * 100).toFixed(2)}%</td>
      <td>${entry.support}</td>
    `;
    tbody.appendChild(tr);
  });

  if (report.macro_avg) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>Macro avg</strong></td>
      <td>${(report.macro_avg.precision * 100).toFixed(2)}%</td>
      <td>${(report.macro_avg.recall * 100).toFixed(2)}%</td>
      <td>${(report.macro_avg.f1 * 100).toFixed(2)}%</td>
      <td>${report.macro_avg.support}</td>
    `;
    tbody.appendChild(tr);
  }

  if (report.weighted_avg) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>Weighted avg</strong></td>
      <td>${(report.weighted_avg.precision * 100).toFixed(2)}%</td>
      <td>${(report.weighted_avg.recall * 100).toFixed(2)}%</td>
      <td>${(report.weighted_avg.f1 * 100).toFixed(2)}%</td>
      <td>${report.weighted_avg.support}</td>
    `;
    tbody.appendChild(tr);
  }
}

function renderDigitCards(perClass, samples) {
  const container = document.getElementById("digit-cards");
  container.innerHTML = "";
  if (!perClass || perClass.length === 0) {
    container.innerHTML = "<p>No classification metrics available.</p>";
    return;
  }

  perClass.forEach((entry) => {
    const card = document.createElement("article");
    card.className = "digit-card";
    const canvas = document.createElement("canvas");
    canvas.width = 84;
    canvas.height = 84;

    const sample = findSampleForLabel(samples, entry.label);
    const pixels = sample
      ? sample.pixels
      : Array.from({ length: 28 }, () => Array(28).fill(0));
    drawDigit(canvas, pixels);

    const meta = document.createElement("div");
    meta.className = "meta";
    const precision = (entry.precision * 100).toFixed(2);
    const recall = (entry.recall * 100).toFixed(2);
    const f1 = (entry.f1 * 100).toFixed(2);
    const note = sample
      ? sample.predicted_label === sample.true_label
        ? "Representative correct prediction."
        : "Example of a confusing stroke."
      : "No sample cached for this digit.";

    meta.innerHTML = `
      <strong>Digit ${entry.label}</strong><br/>
      Precision: ${precision}%<br/>
      Recall: ${recall}%<br/>
      F1-score: ${f1}%<br/>
      Support: ${entry.support}<br/>
      <em>${note}</em>
    `;

    card.appendChild(canvas);
    card.appendChild(meta);
    container.appendChild(card);
  });
}

function renderTakeaways(meta, extremes, report) {
  const list = document.getElementById("insight-takeaways");
  list.innerHTML = "";

  const items = [
    `The current run was generated on ${meta.generated_at} using the ${
      meta.using_mini_sklearn ? "mini_sklearn fallback" : "scikit-learn"
    } trainer.`,
    `Overall accuracy remains ${(meta.accuracy * 100).toFixed(
      2
    )}% on the held-out 10k examples.`,
  ];

  if (extremes.best) {
    items.push(
      `Best performer: digit ${extremes.best.label} hits ${(extremes.best.recall * 100).toFixed(
        1
      )}% accuracy.`
    );
  }
  if (extremes.worst) {
    items.push(
      `Needs work: digit ${extremes.worst.label} only reaches ${(extremes.worst.recall * 100).toFixed(
        1
      )}% accuracy—worth future data augmentation.`
    );
  }
  if (report && report.weighted_avg) {
    items.push(
      `Weighted-average F1 sits at ${(report.weighted_avg.f1 * 100).toFixed(
        2
      )}%, reflecting class imbalance handling.`
    );
  }

  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    list.appendChild(li);
  });
}

function findSampleForLabel(samples, label) {
  if (!samples || samples.length === 0) {
    return null;
  }
  const exact = samples.find(
    (sample) =>
      sample.true_label === label && sample.predicted_label === label
  );
  if (exact) {
    return exact;
  }
  return samples.find(
    (sample) =>
      sample.true_label === label || sample.predicted_label === label
  );
}

function drawDigit(canvas, pixels) {
  const ctx = canvas.getContext("2d");
  const flat = pixels.flat();
  const imageData = ctx.createImageData(28, 28);
  for (let i = 0; i < flat.length; i += 1) {
    const val = Math.min(255, Math.max(0, Math.round(flat[i] * 255)));
    const offset = i * 4;
    imageData.data[offset] = val;
    imageData.data[offset + 1] = val;
    imageData.data[offset + 2] = val;
    imageData.data[offset + 3] = 255;
  }

  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = 28;
  tempCanvas.height = 28;
  const tempCtx = tempCanvas.getContext("2d");
  tempCtx.putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
}
