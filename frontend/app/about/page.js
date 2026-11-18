import Link from "next/link";
import { HeartPulse, GitBranch, Database, Cpu, ArrowRight } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-12 sm:px-6 sm:py-16">
      <span className="inline-flex items-center gap-2 rounded-full bg-primary-50 px-3 py-1 text-xs font-medium text-primary-700">
        <HeartPulse size={14} />
        About the project
      </span>
      <h1 className="mt-3 text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
        How CardioSense works
      </h1>
      <p className="mt-3 max-w-2xl text-slate-600">
        A short overview of the model, data, and current build status.
      </p>

      <div className="mt-10 space-y-6">
        <Card
          icon={<Cpu size={18} />}
          title="The model"
          text="A supervised classifier trained to distinguish five common rhythm classes from ECG-derived features and clinical parameters. The deployed artifact ships as a pickled model behind a Flask/FastAPI endpoint."
        />
        <Card
          icon={<Database size={18} />}
          title="Training data"
          text="Primarily derived from the MIT-BIH Arrhythmia Database, supplemented by feature engineering on RR / PR / QRS / QT intervals and selected clinical covariates."
        />
        <Card
          icon={<GitBranch size={18} />}
          title="Current status"
          text="This UI ships with a mock prediction service so the full flow can be reviewed while the backend is being finalized. Swap lib/mockApi.js for a fetch call once the inference endpoint is live."
        />
      </div>

      <div className="mt-10 rounded-2xl border border-slate-200 bg-slate-50 p-6">
        <h2 className="text-base font-semibold text-slate-900">
          Wiring up the real backend
        </h2>
        <p className="mt-2 text-sm text-slate-600">
          Replace the mock call inside{" "}
          <code className="rounded bg-white px-1.5 py-0.5 text-xs">
            lib/mockApi.js
          </code>{" "}
          with something like:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-900 p-4 text-xs leading-relaxed text-slate-100">
{`export async function predictArrhythmia(input) {
  const res = await fetch(process.env.NEXT_PUBLIC_API_URL + "/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error("Prediction failed");
  return res.json();
}`}
        </pre>
      </div>

      <div className="mt-10 flex flex-wrap items-center gap-3">
        <Link href="/predict" className="btn-primary">
          Try the analyzer
          <ArrowRight size={16} />
        </Link>
        <a
          href="https://github.com/SahilInstinct/Heart-arrhythmia-detection-deployed"
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary"
        >
          View backend repo
        </a>
      </div>
    </div>
  );
}

function Card({ icon, title, text }) {
  return (
    <div className="card">
      <div className="flex items-start gap-3">
        <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-primary-50 text-primary-600">
          {icon}
        </div>
        <div>
          <h3 className="text-base font-semibold text-slate-900">{title}</h3>
          <p className="mt-1.5 text-sm text-slate-600">{text}</p>
        </div>
      </div>
    </div>
  );
}
