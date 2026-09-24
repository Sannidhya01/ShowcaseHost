import { AppShell } from "@/components/app-shell";
import { ApiHealthStatus } from "@/features/health/api-health-status";
import { RepositoryIngestion } from "@/features/repositories/repository-ingestion";
import Link from "next/link";

export default function Home() {
  return (
    <AppShell>
      <section className="flex flex-1 flex-col justify-center gap-6">
        <div>
          <p className="mb-3 text-sm font-medium uppercase tracking-wide text-cyan-700">
            Codebase intelligence foundation
          </p>
          <h1 className="text-5xl font-semibold tracking-normal text-slate-950">ShowcaseHost</h1>
        </div>
        <div className="max-w-2xl border-l-4 border-cyan-600 pl-5">
          <p className="text-lg leading-8 text-slate-700">
            A modular monolith foundation for repository ingestion, structural parsing, hybrid
            search, and cited codebase answers.
          </p>
        </div>
        <ApiHealthStatus />
        <RepositoryIngestion />
        {process.env.NODE_ENV !== "production" ? (
          <Link
            className="w-fit text-sm font-medium text-cyan-700 underline-offset-4 hover:underline"
            href="/evaluations"
          >
            Open evaluation workspace →
          </Link>
        ) : null}
      </section>
    </AppShell>
  );
}
