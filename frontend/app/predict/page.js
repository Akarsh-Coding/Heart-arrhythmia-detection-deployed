"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import {
  Activity,
  Upload,
  User,
  HeartPulse,
  Gauge,
  Info,
  ArrowRight,
  FileSpreadsheet,
  Pencil,
} from "lucide-react";
import { predictArrhythmia } from "@/lib/mockApi";

const INITIAL = {
  age: "",
  sex: "",
  heartRate: "",
  rhythm: "",
  prInterval: "",
  qrsDuration: "",
  qtInterval: "",
  rrInterval: "",
  systolicBP: "",
  diastolicBP: "",
  chestPain: "no",
  smoker: "no",
  diabetes: "no",
  hypertension: "no",
  notes: "",
};

export default function PredictPage() {
  const router = useRouter();
  const [mode, setMode] = useState("manual"); // "manual" | "upload"
  const [form, setForm] = useState(INITIAL);
  const [file, setFile] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [errors, setErrors] = useState({});

  const update = (name, value) => {
    setForm((f) => ({ ...f, [name]: value }));
    if (errors[name]) setErrors((e) => ({ ...e, [name]: undefined }));
  };

  const validate = () => {
    const e = {};
    if (mode === "manual") {
      if (!form.age || Number(form.age) <= 0) e.age = "Required";
      if (!form.sex) e.sex = "Required";
      if (!form.heartRate || Number(form.heartRate) <= 0)
        e.heartRate = "Required";
    } else {
      if (!file) e.file = "Please choose an ECG file";
    }
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const loadSample = () => {
    setForm({
      age: "58",
      sex: "male",
      heartRate: "118",
      rhythm: "irregular",
      prInterval: "",
      qrsDuration: "96",
      qtInterval: "420",
      rrInterval: "510",
      systolicBP: "138",
      diastolicBP: "88",
      chestPain: "yes",
      smoker: "yes",
      diabetes: "no",
      hypertension: "yes",
      notes: "Patient reports intermittent palpitations over the last 3 days.",
    });
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    if (!validate()) return;

    setSubmitting(true);
    try {
      const payload =
        mode === "manual"
          ? { ...form, source: "manual" }
          : {
              source: "ecg-upload",
              fileName: file?.name,
              fileSize: file?.size,
              // In the real API you'll send the file via FormData instead.
            };

      const result = await predictArrhythmia(payload);

      // Pass result to the result page via sessionStorage
      if (typeof window !== "undefined") {
        sessionStorage.setItem("cs:lastResult", JSON.stringify(result));
      }
      router.push("/result");
    } catch (err) {
      setErrors({ _global: "Something went wrong. Please try again." });
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 sm:py-14">
      <div className="mb-8">
        <span className="inline-flex items-center gap-2 rounded-full bg-primary-50 px-3 py-1 text-xs font-medium text-primary-700">
          <Activity size={14} />
          Analysis
        </span>
        <h1 className="mt-3 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
          Run a rhythm analysis
        </h1>
        <p className="mt-2 max-w-2xl text-slate-600">
          Choose how you'd like to provide the input. Fields marked with{" "}
          <span className="text-primary-600">*</span> are required.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="mb-8 flex flex-wrap items-center gap-3">
        <ModeTab
          active={mode === "manual"}
          onClick={() => setMode("manual")}
          icon={<Pencil size={16} />}
          label="Manual entry"
        />
        <ModeTab
          active={mode === "upload"}
          onClick={() => setMode("upload")}
          icon={<FileSpreadsheet size={16} />}
          label="Upload ECG file"
        />
        <button
          type="button"
          onClick={loadSample}
          className="ml-auto text-sm font-medium text-primary-600 hover:text-primary-700"
        >
          Load sample data
        </button>
      </div>

      <form onSubmit={onSubmit} className="space-y-6">
        {errors._global && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {errors._global}
          </div>
        )}

        {mode === "manual" ? (
          <>
            {/* Demographics */}
            <Section
              icon={<User size={16} />}
              title="Patient demographics"
              description="Basic identifying information helps calibrate predictions."
            >
              <div className="grid gap-5 md:grid-cols-3">
                <Field
                  label="Age"
                  required
                  error={errors.age}
                  htmlFor="age"
                  suffix="years"
                >
                  <input
                    id="age"
                    type="number"
                    min="0"
                    max="120"
                    className="input-base"
                    value={form.age}
                    onChange={(e) => update("age", e.target.value)}
                    placeholder="e.g. 54"
                  />
                </Field>

                <Field label="Sex" required error={errors.sex} htmlFor="sex">
                  <select
                    id="sex"
                    className="input-base"
                    value={form.sex}
                    onChange={(e) => update("sex", e.target.value)}
                  >
                    <option value="">Select…</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="other">Other</option>
                  </select>
                </Field>

                <Field label="Chest pain" htmlFor="chestPain">
                  <select
                    id="chestPain"
                    className="input-base"
                    value={form.chestPain}
                    onChange={(e) => update("chestPain", e.target.value)}
                  >
                    <option value="no">No</option>
                    <option value="atypical">Atypical</option>
                    <option value="typical">Typical angina</option>
                    <option value="yes">Non-specific</option>
                  </select>
                </Field>
              </div>
            </Section>

            {/* Vitals */}
            <Section
              icon={<HeartPulse size={16} />}
              title="Vitals & ECG intervals"
              description="Core measurements used by the model."
            >
              <div className="grid gap-5 md:grid-cols-3">
                <Field
                  label="Heart rate"
                  required
                  error={errors.heartRate}
                  htmlFor="heartRate"
                  suffix="bpm"
                >
                  <input
                    id="heartRate"
                    type="number"
                    min="20"
                    max="250"
                    className="input-base"
                    value={form.heartRate}
                    onChange={(e) => update("heartRate", e.target.value)}
                    placeholder="e.g. 78"
                  />
                </Field>

                <Field label="Rhythm" htmlFor="rhythm">
                  <select
                    id="rhythm"
                    className="input-base"
                    value={form.rhythm}
                    onChange={(e) => update("rhythm", e.target.value)}
                  >
                    <option value="">Unknown</option>
                    <option value="regular">Regular</option>
                    <option value="irregular">Irregular</option>
                  </select>
                </Field>

                <Field
                  label="PR interval"
                  htmlFor="prInterval"
                  suffix="ms"
                  hint="Normal 120–200 ms"
                >
                  <input
                    id="prInterval"
                    type="number"
                    min="0"
                    max="500"
                    className="input-base"
                    value={form.prInterval}
                    onChange={(e) => update("prInterval", e.target.value)}
                    placeholder="e.g. 160"
                  />
                </Field>

                <Field
                  label="QRS duration"
                  htmlFor="qrsDuration"
                  suffix="ms"
                  hint="Normal < 120 ms"
                >
                  <input
                    id="qrsDuration"
                    type="number"
                    min="0"
                    max="500"
                    className="input-base"
                    value={form.qrsDuration}
                    onChange={(e) => update("qrsDuration", e.target.value)}
                    placeholder="e.g. 92"
                  />
                </Field>

                <Field
                  label="QT interval"
                  htmlFor="qtInterval"
                  suffix="ms"
                  hint="Normal 350–450 ms"
                >
                  <input
                    id="qtInterval"
                    type="number"
                    min="0"
                    max="700"
                    className="input-base"
                    value={form.qtInterval}
                    onChange={(e) => update("qtInterval", e.target.value)}
                    placeholder="e.g. 420"
                  />
                </Field>

                <Field label="RR interval" htmlFor="rrInterval" suffix="ms">
                  <input
                    id="rrInterval"
                    type="number"
                    min="0"
                    max="3000"
                    className="input-base"
                    value={form.rrInterval}
                    onChange={(e) => update("rrInterval", e.target.value)}
                    placeholder="e.g. 780"
                  />
                </Field>
              </div>
            </Section>

            {/* BP + risk factors */}
            <Section
              icon={<Gauge size={16} />}
              title="Blood pressure & risk factors"
              description="Context that informs overall risk assessment."
            >
              <div className="grid gap-5 md:grid-cols-3">
                <Field
                  label="Systolic BP"
                  htmlFor="systolicBP"
                  suffix="mmHg"
                >
                  <input
                    id="systolicBP"
                    type="number"
                    min="50"
                    max="260"
                    className="input-base"
                    value={form.systolicBP}
                    onChange={(e) => update("systolicBP", e.target.value)}
                    placeholder="e.g. 120"
                  />
                </Field>

                <Field label="Diastolic BP" htmlFor="diastolicBP" suffix="mmHg">
                  <input
                    id="diastolicBP"
                    type="number"
                    min="30"
                    max="160"
                    className="input-base"
                    value={form.diastolicBP}
                    onChange={(e) => update("diastolicBP", e.target.value)}
                    placeholder="e.g. 80"
                  />
                </Field>

                <Field label="Smoker" htmlFor="smoker">
                  <select
                    id="smoker"
                    className="input-base"
                    value={form.smoker}
                    onChange={(e) => update("smoker", e.target.value)}
                  >
                    <option value="no">No</option>
                    <option value="former">Former</option>
                    <option value="yes">Current</option>
                  </select>
                </Field>

                <Field label="Diabetes" htmlFor="diabetes">
                  <select
                    id="diabetes"
                    className="input-base"
                    value={form.diabetes}
                    onChange={(e) => update("diabetes", e.target.value)}
                  >
                    <option value="no">No</option>
                    <option value="type1">Type 1</option>
                    <option value="type2">Type 2</option>
                  </select>
                </Field>

                <Field label="Hypertension" htmlFor="hypertension">
                  <select
                    id="hypertension"
                    className="input-base"
                    value={form.hypertension}
                    onChange={(e) => update("hypertension", e.target.value)}
                  >
                    <option value="no">No</option>
                    <option value="yes">Yes</option>
                    <option value="treated">Yes, treated</option>
                  </select>
                </Field>
              </div>

              <div className="mt-5">
                <Field label="Clinical notes (optional)" htmlFor="notes">
                  <textarea
                    id="notes"
                    rows={3}
                    className="input-base"
                    value={form.notes}
                    onChange={(e) => update("notes", e.target.value)}
                    placeholder="Any additional symptoms, history, or observations…"
                  />
                </Field>
              </div>
            </Section>
          </>
        ) : (
          <Section
            icon={<Upload size={16} />}
            title="ECG file upload"
            description="Upload an ECG recording in CSV / TXT / WFDB format."
          >
            <label
              htmlFor="ecgFile"
              className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-10 transition ${
                errors.file
                  ? "border-red-300 bg-red-50"
                  : "border-slate-300 bg-slate-50 hover:border-primary-400 hover:bg-primary-50/40"
              }`}
            >
              <div className="grid h-12 w-12 place-items-center rounded-xl bg-primary-100 text-primary-700">
                <Upload size={20} />
              </div>
              <div className="text-center">
                <p className="text-sm font-semibold text-slate-800">
                  {file ? file.name : "Click to choose a file or drag it here"}
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Supported: .csv, .txt, .dat, .hea (max 10 MB)
                </p>
              </div>
              <input
                id="ecgFile"
                type="file"
                accept=".csv,.txt,.dat,.hea"
                className="hidden"
                onChange={(e) => {
                  setFile(e.target.files?.[0] || null);
                  if (errors.file) setErrors({ ...errors, file: undefined });
                }}
              />
            </label>
            {errors.file && (
              <p className="mt-2 text-xs text-red-600">{errors.file}</p>
            )}

            <div className="mt-4 flex items-start gap-2 rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-800">
              <Info size={14} className="mt-0.5 shrink-0" />
              <span>
                In this demo the file is only read to extract its name/size.
                Once the backend is ready it will be forwarded via{" "}
                <code className="rounded bg-white/70 px-1">FormData</code> to
                the real inference endpoint.
              </span>
            </div>
          </Section>
        )}

        {/* Submit */}
        <div className="flex flex-col-reverse items-stretch gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-slate-500">
            By running an analysis you acknowledge this tool is for research
            and triage support, not clinical diagnosis.
          </p>
          <div className="flex gap-3">
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                setForm(INITIAL);
                setFile(null);
                setErrors({});
              }}
            >
              Reset
            </button>
            <button type="submit" className="btn-primary" disabled={submitting}>
              {submitting ? (
                <>
                  <span className="spinner !h-4 !w-4 !border-2" />
                  Analyzing…
                </>
              ) : (
                <>
                  Run analysis
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

function ModeTab({ active, onClick, icon, label }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition ${
        active
          ? "border-primary-600 bg-primary-600 text-white shadow-sm"
          : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function Section({ icon, title, description, children }) {
  return (
    <section className="card">
      <div className="mb-5 flex items-start gap-3">
        <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-primary-50 text-primary-600">
          {icon}
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-900">{title}</h2>
          <p className="text-sm text-slate-500">{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function Field({ label, required, error, htmlFor, suffix, hint, children }) {
  return (
    <div>
      <label htmlFor={htmlFor} className="label-base">
        {label}
        {required && <span className="ml-0.5 text-primary-600">*</span>}
        {suffix && (
          <span className="ml-1 text-xs font-normal text-slate-500">
            ({suffix})
          </span>
        )}
      </label>
      {children}
      {error ? (
        <p className="mt-1 text-xs text-red-600">{error}</p>
      ) : hint ? (
        <p className="mt-1 text-xs text-slate-500">{hint}</p>
      ) : null}
    </div>
  );
}
