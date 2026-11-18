// Mock prediction service — replace with real backend call when ready.
// Backend contract (assumed, update when confirmed):
//   POST /api/predict
//   body: JSON with the fields collected in the form
//   returns: { className, confidence, probabilities: {class: p}, riskLevel, notes }

const CLASSES = [
  {
    key: "N",
    name: "Normal Sinus Rhythm",
    risk: "low",
    description:
      "Regular rhythm originating from the sinus node. Rate typically 60–100 bpm with normal P-QRS-T morphology.",
    recommendation:
      "No immediate concern detected. Maintain routine cardiovascular check-ups.",
  },
  {
    key: "AFIB",
    name: "Atrial Fibrillation",
    risk: "high",
    description:
      "Irregularly irregular rhythm with absent P waves. Raises risk of stroke and heart failure if left untreated.",
    recommendation:
      "Consult a cardiologist promptly. Rate/rhythm control and anticoagulation may be indicated.",
  },
  {
    key: "VT",
    name: "Ventricular Tachycardia",
    risk: "critical",
    description:
      "Rapid rhythm originating from the ventricles. Can compromise cardiac output and progress to VF.",
    recommendation:
      "Treat as a medical emergency. Seek urgent clinical evaluation.",
  },
  {
    key: "PVC",
    name: "Premature Ventricular Contraction",
    risk: "moderate",
    description:
      "Early ventricular beats with wide QRS. Usually benign in healthy hearts but may warrant evaluation when frequent.",
    recommendation:
      "Consider a 24-hour Holter and discuss burden/symptoms with your clinician.",
  },
  {
    key: "BRADY",
    name: "Sinus Bradycardia",
    risk: "moderate",
    description:
      "Sinus rhythm with rate < 60 bpm. May be normal in athletes but pathologic if symptomatic.",
    recommendation:
      "Correlate with symptoms (dizziness, syncope). Clinical review recommended if symptomatic.",
  },
];

function hashSeed(obj) {
  const s = JSON.stringify(obj);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = (h * 16777619) >>> 0;
  }
  return h;
}

// Deterministic pseudo-random in [0,1) seeded by a number
function rand(seed) {
  let x = seed || 1;
  return () => {
    x = (x * 1664525 + 1013904223) >>> 0;
    return x / 2 ** 32;
  };
}

export async function predictArrhythmia(input) {
  // Simulate network latency
  await new Promise((r) => setTimeout(r, 1200));

  const seed = hashSeed(input);
  const r = rand(seed);

  // Bias prediction based on heart rate if provided
  const hr = Number(input.heartRate) || 72;
  let picked;
  if (hr >= 150) picked = CLASSES.find((c) => c.key === "VT");
  else if (hr >= 100 && (input.rhythm === "irregular" || r() > 0.55))
    picked = CLASSES.find((c) => c.key === "AFIB");
  else if (hr < 55) picked = CLASSES.find((c) => c.key === "BRADY");
  else if (r() > 0.75) picked = CLASSES.find((c) => c.key === "PVC");
  else picked = CLASSES.find((c) => c.key === "N");

  // Build probability distribution with picked class dominant
  const base = CLASSES.map(() => 0.05 + r() * 0.1);
  const total0 = base.reduce((a, b) => a + b, 0);
  const normalized = base.map((b) => b / total0);
  const idx = CLASSES.findIndex((c) => c.key === picked.key);
  const boost = 0.55 + r() * 0.3;
  const probs = normalized.map((p, i) =>
    i === idx ? p + boost : p * (1 - boost),
  );
  const sum = probs.reduce((a, b) => a + b, 0);
  const normProbs = probs.map((p) => p / sum);

  const probabilities = CLASSES.reduce((acc, c, i) => {
    acc[c.key] = {
      name: c.name,
      value: Number(normProbs[i].toFixed(4)),
    };
    return acc;
  }, {});

  return {
    classKey: picked.key,
    className: picked.name,
    confidence: Number(normProbs[idx].toFixed(4)),
    riskLevel: picked.risk,
    description: picked.description,
    recommendation: picked.recommendation,
    probabilities,
    input,
    timestamp: new Date().toISOString(),
    _demo: true, // flag so we show a banner on result page
  };
}

export const ARRHYTHMIA_CLASSES = CLASSES;
