"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiClientError, apiClient, apiUrl } from "@/lib/api-client";
import type { CurrentUser, EmbeddingModel, IngestionJob, Repository } from "@/types/api";

type RepositoryDisplayStatus = {
  label: "Ready" | "Not Ready" | "Processing" | "Failed";
  className: string;
  error: string | null;
};

function displayStatus(
  repository: Repository,
  job: IngestionJob | undefined,
): RepositoryDisplayStatus {
  if (job?.status === "failed") {
    return {
      label: "Failed",
      className: "text-red-700",
      error: job.error_message || job.error_code || "Repository ingestion failed",
    };
  }
  if (job?.chunking_status === "failed") {
    return {
      label: "Failed",
      className: "text-red-700",
      error: job.chunking_error_message || job.chunking_error_code || "Code chunking failed",
    };
  }
  if (job?.embedding_status === "failed") {
    return {
      label: "Failed",
      className: "text-red-700",
      error:
        job.embedding_error_message || job.embedding_error_code || "Embedding generation failed",
    };
  }
  const processing =
    job?.status === "queued" ||
    job?.status === "running" ||
    job?.chunking_status === "queued" ||
    job?.chunking_status === "running" ||
    job?.embedding_status === "queued" ||
    job?.embedding_status === "running" ||
    (job?.status === "succeeded" && !job.embedding_status);
  if (processing) {
    return { label: "Processing", className: "text-cyan-700", error: null };
  }
  if (repository.chat_ready || job?.embedding_status === "succeeded") {
    return { label: "Ready", className: "text-emerald-700", error: null };
  }
  return { label: "Not Ready", className: "text-slate-500", error: null };
}

export function RepositoryIngestion() {
  const router = useRouter();
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [repositories, setRepositories] = useState<Repository[]>([]);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModel[]>([]);
  const [selectedModels, setSelectedModels] = useState<Record<string, string>>({});
  const [jobs, setJobs] = useState<Record<string, IngestionJob>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingChat, setPendingChat] = useState<string | null>(null);

  useEffect(() => {
    apiClient
      .getCurrentUser()
      .then(async (currentUser) => {
        const [connectedRepositories, models] = await Promise.all([
          apiClient.getRepositories(),
          apiClient.getEmbeddingModels(),
        ]);
        setUser(currentUser);
        setRepositories(connectedRepositories);
        setEmbeddingModels(models);
        setJobs(
          Object.fromEntries(
            connectedRepositories
              .filter((repository) => repository.latest_ingestion)
              .map((repository) => [repository.id, repository.latest_ingestion!]),
          ),
        );
      })
      .catch((caught: unknown) => {
        if (!(caught instanceof ApiClientError && caught.status === 401)) {
          setError(caught instanceof Error ? caught.message : "Unable to load repositories");
        }
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const active = Object.values(jobs).filter(
      (job) =>
        job.status === "queued" ||
        job.status === "running" ||
        job.chunking_status === "queued" ||
        job.chunking_status === "running" ||
        job.embedding_status === "queued" ||
        job.embedding_status === "running" ||
        (job.status === "succeeded" && !job.embedding_status && job.chunking_status !== "failed"),
    );
    if (active.length === 0) return;
    const timer = window.setInterval(() => {
      void Promise.all(active.map((job) => apiClient.getIngestion(job.id))).then((updates) => {
        setJobs((current) => {
          const next = { ...current };
          for (const update of updates) next[update.repository_id] = update;
          return next;
        });
      });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [jobs]);

  useEffect(() => {
    if (pendingChat && jobs[pendingChat]?.embedding_status === "succeeded") {
      router.push(`/repositories/${pendingChat}/chat`);
    }
  }, [jobs, pendingChat, router]);

  async function ingest(repository: Repository) {
    setError(null);
    setPendingChat(repository.id);
    try {
      const job = await apiClient.startIngestion(
        repository.id,
        selectedModels[repository.id] || null,
      );
      setJobs((current) => ({ ...current, [repository.id]: job }));
    } catch (caught) {
      setPendingChat(null);
      setError(caught instanceof Error ? caught.message : "Unable to start ingestion");
    }
  }

  function openChat(repository: Repository) {
    if (repository.chat_ready || jobs[repository.id]?.embedding_status === "succeeded") {
      router.push(`/repositories/${repository.id}/chat`);
      return;
    }
    void ingest(repository);
  }

  async function logout() {
    await apiClient.logout();
    setUser(null);
    setRepositories([]);
    setJobs({});
  }

  if (loading) return <p className="text-sm text-slate-600">Loading GitHub connection…</p>;

  if (!user) {
    return (
      <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <h2 className="text-xl font-semibold">Connect a repository</h2>
        <p className="mt-2 text-sm leading-6 text-slate-600">
          Sign in with GitHub, then install the read-only ShowcaseHost GitHub App on repositories
          you choose.
        </p>
        <a
          className="mt-5 inline-flex rounded-lg bg-slate-950 px-4 py-2 text-sm font-medium text-white"
          href={apiUrl("/auth/github/start")}
        >
          Sign in with GitHub
        </a>
        {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Repositories</h2>
          <p className="mt-1 text-sm text-slate-600">Signed in as @{user.github_login}</p>
        </div>
        <div className="flex gap-2">
          <a
            className="rounded-lg bg-cyan-700 px-3 py-2 text-sm font-medium text-white"
            href={apiUrl("/github/install/start")}
          >
            Add repositories
          </a>
          <button
            className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
            onClick={() => void logout()}
            type="button"
          >
            Sign out
          </button>
        </div>
      </div>
      {repositories.length === 0 ? (
        <p className="mt-6 text-sm text-slate-600">No GitHub App repositories are connected yet.</p>
      ) : (
        <ul className="mt-6 divide-y divide-slate-200">
          {repositories.map((repository) => {
            const job = jobs[repository.id];
            const busy =
              job?.status === "queued" ||
              job?.status === "running" ||
              job?.chunking_status === "queued" ||
              job?.chunking_status === "running" ||
              job?.embedding_status === "queued" ||
              job?.embedding_status === "running";
            const status = displayStatus(repository, job);
            return (
              <li
                className="relative flex flex-wrap items-center justify-between gap-4 py-4 pr-12"
                key={repository.id}
              >
                <div>
                  <p className="font-medium text-slate-900">{repository.full_name}</p>
                  <p className="mt-1 text-xs text-slate-500">
                    {repository.private ? "Private" : "Public"} · {repository.default_branch} ·{" "}
                    <span className={status.className}>{status.label}</span>
                  </p>
                  {status.error ? (
                    <p className="mt-1 text-xs text-red-700">{status.error}</p>
                  ) : null}
                </div>
                <div className="flex flex-col items-end gap-2">
                  <button
                    className="rounded-lg bg-cyan-700 px-3 py-2 text-sm font-medium text-white disabled:cursor-wait disabled:opacity-60"
                    disabled={busy}
                    onClick={() => openChat(repository)}
                    type="button"
                  >
                    {busy
                      ? job?.embedding_status === "queued" || job?.embedding_status === "running"
                        ? "Preparing embeddings…"
                        : job?.chunking_status === "queued" || job?.chunking_status === "running"
                          ? "Preparing code…"
                          : "Connecting repository…"
                      : "Chat with repository"}
                  </button>
                </div>
                <details className="absolute right-0 top-4">
                  <summary
                    aria-label={`Repository options for ${repository.full_name}`}
                    className="flex h-8 w-8 cursor-pointer list-none items-center justify-center rounded-full text-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900"
                  >
                    ⋯
                  </summary>
                  <div className="absolute right-0 z-10 mt-1 w-72 rounded-lg border border-slate-200 bg-white p-3 shadow-lg">
                    <dl className="mb-3 space-y-2 border-b border-slate-200 pb-3 text-xs">
                      <div>
                        <dt className="font-medium text-slate-500">Snapshot</dt>
                        <dd className="mt-0.5 font-mono text-slate-800">
                          {job?.commit_sha ? job.commit_sha.slice(0, 12) : "Not created"}
                        </dd>
                      </div>
                      <div>
                        <dt className="font-medium text-slate-500">Code chunks</dt>
                        <dd className="mt-0.5 text-slate-800">
                          {job?.chunking_status
                            ? job.chunking_status === "succeeded"
                              ? "Ready"
                              : job.chunking_status
                            : "Not started"}
                        </dd>
                      </div>
                      <div>
                        <dt className="font-medium text-slate-500">Embeddings</dt>
                        <dd className="mt-0.5 text-slate-800">
                          {job?.embedding_status === "succeeded"
                            ? `Ready${job.embedding_model ? ` · ${job.embedding_model}` : ""}`
                            : job?.embedding_status === "queued" ||
                                job?.embedding_status === "running"
                              ? `Processing · ${Math.round((job.embedding_progress ?? 0) * 100)}%`
                              : (job?.embedding_status ?? "Not started")}
                        </dd>
                        {job?.embedding_status === "queued" ||
                        job?.embedding_status === "running" ? (
                          <progress
                            aria-label="Embedding progress"
                            className="mt-1 h-1.5 w-full"
                            max={1}
                            value={job.embedding_progress ?? 0}
                          />
                        ) : null}
                      </div>
                    </dl>
                    <label className="block text-xs font-medium text-slate-700">
                      Embedding model
                      <select
                        className="mt-2 w-full rounded border border-slate-300 bg-white px-2 py-1.5"
                        disabled={busy}
                        onChange={(event) =>
                          setSelectedModels((current) => ({
                            ...current,
                            [repository.id]: event.target.value,
                          }))
                        }
                        value={selectedModels[repository.id] ?? ""}
                      >
                        <option value="">Auto (recommended)</option>
                        {embeddingModels.map((model) => (
                          <option key={model.id} value={model.id}>
                            {model.id} · {model.dimension}d
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </details>
              </li>
            );
          })}
        </ul>
      )}
      {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}
    </section>
  );
}
