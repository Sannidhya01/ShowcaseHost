import Link from "next/link";
import { notFound } from "next/navigation";
import { AppShell } from "@/components/app-shell";
import { EvaluationDashboard } from "@/features/evaluations/evaluation-dashboard";

export default function EvaluationsPage() {
  if (process.env.NODE_ENV === "production") notFound();

  return (
    <AppShell>
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-2 text-sm font-medium uppercase tracking-wide text-cyan-700">
            Development tools
          </p>
          <h1 className="text-4xl font-semibold tracking-tight text-slate-950">
            Model evaluations
          </h1>
          <p className="mt-3 max-w-2xl leading-7 text-slate-600">
            Run deterministic benchmark scoring and inspect aggregate and per-example results.
          </p>
        </div>
        <Link className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm" href="/">
          Back to app
        </Link>
      </header>
      <EvaluationDashboard />
    </AppShell>
  );
}
