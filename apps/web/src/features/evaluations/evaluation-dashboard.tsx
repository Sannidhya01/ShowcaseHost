"use client";

import { useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { GenerationModel } from "@/types/api";
import type {
  EvaluationBenchmark,
  EvaluationBenchmarkId,
  EvaluationRun,
  EvaluationRunRequest,
  EvaluationRetrieval,
} from "./types";

const FALLBACK_MODEL = "qwen/qwen3.8-27b";

function validSeed(value: string) {
  return /^\d+$/.test(value) && BigInt(value) <= BigInt("9223372036854775807");
}

function savedSeed(run: EvaluationRun) {
  const value = run.request.seed;
  if (typeof value === "number" && !Number.isSafeInteger(value)) return null;
  return value !== undefined && validSeed(String(value)) ? String(value) : null;
}

function label(value: string) {
  if (value === "official_pass_at_1") return "Official pass@1 (0.8)";
  if (value === "mean_reciprocal_rank") return "Retrieval MRR";
  return value
    .replace(/^average_/, "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function score(value: number) {
  return value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
}

const METRICS_BY_BENCHMARK: Record<EvaluationBenchmarkId, Set<string>> = {
  codequeries: new Set([
    "examples",
    "relevance_accuracy",
    "relevance_precision",
    "relevance_recall",
    "relevance_f1",
  ]),
  repoqa: new Set(["examples", "official_pass_at_1", "mean_reciprocal_rank"]),
};

function date(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function statusStyle(status: EvaluationRun["status"]) {
  if (status === "succeeded") return "bg-emerald-50 text-emerald-700";
  if (status === "failed") return "bg-red-50 text-red-700";
  if (status === "completed_with_errors") return "bg-amber-50 text-amber-700";
  if (status === "cancelled") return "bg-slate-100 text-slate-700";
  return "bg-cyan-50 text-cyan-700";
}

function activeBenchmarkRuns(runs: EvaluationRun[]) {
  return runs.filter((run) => String(run.request.benchmark) !== "infibench");
}

export function EvaluationDashboard() {
  const [benchmarks, setBenchmarks] = useState<EvaluationBenchmark[]>([]);
  const [models, setModels] = useState<GenerationModel[]>([]);
  const [runs, setRuns] = useState<EvaluationRun[]>([]);
  const [benchmarkId, setBenchmarkId] = useState<EvaluationBenchmarkId>("codequeries");
  const [model, setModel] = useState(FALLBACK_MODEL);
  const [split, setSplit] = useState("");
  const [limit, setLimit] = useState(50);
  const [seed, setSeed] = useState("");
  const [retrieval, setRetrieval] = useState<EvaluationRetrieval | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [cancellingRunId, setCancellingRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selected = useMemo(
    () => benchmarks.find((benchmark) => benchmark.id === benchmarkId),
    [benchmarkId, benchmarks],
  );
  const availableSplits = useMemo(() => {
    if (!selected) return [];
    if (selected.splits?.length) {
      return selected.splits.map((item) => ({
        ...item,
        sampling_groups: item.sampling_groups?.length ? item.sampling_groups : [item.id],
      }));
    }
    return [
      {
        id: selected.default_split,
        label: selected.default_split.replace(/\b\w/g, (letter) => letter.toUpperCase()),
        config: selected.default_config,
        examples: selected.max_run_examples ?? 100,
        sampling_groups: [selected.default_split],
      },
    ];
  }, [selected]);
  const selectedSplit = useMemo(
    () => availableSplits.find((item) => item.id === split),
    [availableSplits, split],
  );
  const maximumExamples = Math.min(
    selected?.max_run_examples ?? 100,
    selectedSplit?.examples ?? 100,
  );
  const minimumExamples = selectedSplit?.sampling_groups.length ?? 1;

  useEffect(() => {
    Promise.all([
      apiClient.getEvaluationBenchmarks(),
      apiClient.getEvaluationRuns(),
      apiClient.getEvaluationModels(),
      apiClient.getEvaluationRetrieval(),
    ])
      .then(([catalog, history, availableModels, tuning]) => {
        setRetrieval(tuning);
        setBenchmarks(catalog);
        setRuns(activeBenchmarkRuns(history));
        setModels(availableModels);
        if (catalog[0]) {
          setBenchmarkId(catalog[0].id);
          setSplit(catalog[0].default_split);
        }
        const defaultModel = availableModels.find((item) => item.is_default) ?? availableModels[0];
        if (defaultModel) setModel(defaultModel.id);
      })
      .catch((caught: unknown) => {
        setError(caught instanceof Error ? caught.message : "Unable to load evaluations");
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const active = runs.filter((run) => run.status === "queued" || run.status === "running");
    if (active.length === 0) return;
    const timer = window.setInterval(() => {
      void Promise.all(active.map((run) => apiClient.getEvaluationRun(run.id))).then((updates) => {
        setRuns((current) => {
          const byId = new Map(updates.map((run) => [run.id, run]));
          return current.map((run) => byId.get(run.id) ?? run);
        });
      });
    }, 1500);
    return () => window.clearInterval(timer);
  }, [runs]);

  async function submitRun(request: EvaluationRunRequest) {
    setSubmitting(true);
    setError(null);
    try {
      const tuning = await apiClient.getEvaluationRetrieval();
      const run = await apiClient.startEvaluationRun(request);
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      setRetrieval(tuning);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to start evaluation");
    } finally {
      setSubmitting(false);
    }
  }

  async function startRun() {
    if (!selected || (seed.trim() && !validSeed(seed.trim()))) return;
    await submitRun({
      benchmark: selected.id,
      model,
      config: selectedSplit?.config ?? selected.default_config,
      split: split || selected.default_split,
      offset: 0,
      limit,
      ...(seed.trim() ? { seed: seed.trim() } : {}),
    });
  }

  async function rerun(run: EvaluationRun) {
    const previousSeed = savedSeed(run);
    if (previousSeed === null) return;
    // Preserve sampling and model settings; the server uses its current retrieval tuning.
    await submitRun({ ...run.request, seed: previousSeed });
  }

  async function cancelRun(runId: string) {
    setCancellingRunId(runId);
    setError(null);
    try {
      const update = await apiClient.cancelEvaluationRun(runId);
      setRuns((current) => current.map((run) => (run.id === update.id ? update : run)));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to cancel evaluation");
    } finally {
      setCancellingRunId(null);
    }
  }

  async function retryFailed(runId: string) {
    setSubmitting(true);
    setError(null);
    try {
      const run = await apiClient.retryFailedEvaluationExamples(runId);
      setRuns((current) => [run, ...current.filter((item) => item.id !== run.id)]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to retry failed examples");
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <p className="text-sm text-slate-600">Loading evaluation workspace…</p>;

  return (
    <div className="space-y-8">
      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="grid gap-5 md:grid-cols-2">
          <label className="text-sm font-medium text-slate-700">
            Benchmark
            <select
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              onChange={(event) => {
                const nextId = event.target.value as EvaluationBenchmarkId;
                setBenchmarkId(nextId);
                const next = benchmarks.find((benchmark) => benchmark.id === nextId);
                if (next) setSplit(next.default_split);
                const nextSplit = next?.splits?.find((item) => item.id === next.default_split);
                if (nextSplit) {
                  setLimit((current) => Math.max(current, nextSplit.sampling_groups.length));
                }
              }}
              value={benchmarkId}
            >
              {benchmarks.map((benchmark) => (
                <option key={benchmark.id} value={benchmark.id}>
                  {benchmark.name}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Model {benchmarkId === "codequeries" ? "(RepoQA generation only)" : ""}
            {models.length ? (
              <select
                className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
                disabled={benchmarkId === "codequeries"}
                onChange={(event) => setModel(event.target.value)}
                value={model}
              >
                {models.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label} · {item.provider === "openrouter" ? "OpenRouter" : "Groq"}
                  </option>
                ))}
              </select>
            ) : (
              <input
                className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5"
                onChange={(event) => setModel(event.target.value)}
                value={model}
              />
            )}
          </label>
          <label className="text-sm font-medium text-slate-700">
            Split
            <select
              className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5"
              onChange={(event) => {
                const nextSplit = event.target.value;
                setSplit(nextSplit);
                const splitDefinition = availableSplits.find((item) => item.id === nextSplit);
                if (splitDefinition)
                  setLimit((current) =>
                    Math.max(
                      splitDefinition.sampling_groups.length,
                      Math.min(
                        current,
                        splitDefinition.examples,
                        selected?.max_run_examples ?? 100,
                      ),
                    ),
                  );
              }}
              value={split}
            >
              {availableSplits.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label} ({item.examples.toLocaleString()})
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm font-medium text-slate-700">
            Examples
            <input
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5"
              max={maximumExamples}
              min={minimumExamples}
              onChange={(event) => setLimit(Number(event.target.value))}
              type="number"
              value={limit}
            />
          </label>
          <label className="text-sm font-medium text-slate-700">
            Sampling seed (optional)
            <input
              className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2.5"
              type="text"
              inputMode="numeric"
              placeholder="Leave blank for a new random sample"
              value={seed}
              onChange={(event) => setSeed(event.target.value)}
              aria-describedby="sampling-seed-help"
              aria-invalid={Boolean(seed.trim() && !validSeed(seed.trim()))}
            />
          </label>
        </div>
        <p className="mt-3 text-xs text-slate-600" id="sampling-seed-help">
          Reuse a seed with the same benchmark, split, and example count to repeat the sample. Seeds
          must be whole numbers from 0 to 9223372036854775807.
        </p>
        {retrieval ? (
          <p className="mt-3 text-sm text-slate-700">
            Current hybrid retrieval: vector {retrieval.dense_weight} · keyword{" "}
            {retrieval.keyword_weight}
            {" · "}
            {retrieval.candidate_limit_per_channel} candidates per channel →{" "}
            {retrieval.rerank_candidate_limit} reranked by {retrieval.reranker_model} →{" "}
            {retrieval.result_limit} chunks at score ≥ {retrieval.relevance_threshold} and ≥{" "}
            {retrieval.relative_relevance_threshold} of the query&apos;s best score.
          </p>
        ) : null}

        {selected ? (
          <div className="mt-5 rounded-xl bg-slate-50 p-4 text-sm text-slate-600">
            <p className="font-medium text-slate-900">{selected.metric}</p>
            <p className="mt-1">Dataset: {selected.dataset_id}</p>
            <p className="mt-1">
              Dataset total: {selected.total_examples?.toLocaleString() ?? "not reported"} examples
              · Per-run limit: {(selected.max_run_examples ?? 100).toLocaleString()}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {availableSplits.map((item) => (
                <span className="rounded-full bg-white px-2.5 py-1 text-xs" key={item.id}>
                  {item.label}: {item.examples.toLocaleString()}
                </span>
              ))}
            </div>
            <p className="mt-3 text-xs">
              Random sampling is balanced across:{" "}
              {selectedSplit?.sampling_groups.join(", ") ?? "the selected split"}.
            </p>
            <div className="mt-3 flex flex-wrap gap-4 text-cyan-700">
              <a href={selected.github_url} rel="noreferrer" target="_blank">
                GitHub ↗
              </a>
              <a href={selected.dataset_url} rel="noreferrer" target="_blank">
                Dataset ↗
              </a>
              <a href={selected.paper_url} rel="noreferrer" target="_blank">
                Paper ↗
              </a>
            </div>
          </div>
        ) : null}

        <div className="mt-5 rounded-xl border border-cyan-100 bg-cyan-50 p-4 text-sm leading-6 text-cyan-950">
          <p className="font-semibold">Retrieval-first evaluation</p>
          <p className="mt-1">
            CodeQueries applies its official relevance-classification metrics to retrieved blocks
            without calling a generation model. RepoQA chunks and embeds the full repository,
            retrieves ranked chunks, generates from that retrieved context, and scores the final
            answer with the official 0.8 similarity threshold. Every completed example is saved.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            {[20, 50, 100]
              .filter((value) => value <= maximumExamples)
              .map((value) => (
                <button
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    limit === value ? "bg-cyan-700 text-white" : "bg-white text-cyan-800"
                  }`}
                  key={value}
                  onClick={() => setLimit(value)}
                  type="button"
                >
                  {value} examples
                </button>
              ))}
          </div>
        </div>

        <button
          className="mt-5 rounded-lg bg-cyan-700 px-5 py-2.5 text-sm font-semibold text-white disabled:cursor-wait disabled:opacity-60"
          disabled={
            submitting ||
            !selected ||
            !model ||
            limit < minimumExamples ||
            limit > maximumExamples ||
            Boolean(seed.trim() && !validSeed(seed.trim()))
          }
          onClick={() => void startRun()}
          type="button"
        >
          {submitting ? "Starting…" : "Run evaluation"}
        </button>
        {error ? <p className="mt-4 text-sm text-red-700">{error}</p> : null}
      </section>

      <section>
        <div className="flex items-end justify-between gap-3">
          <div>
            <h2 className="text-2xl font-semibold">Runs</h2>
            <p className="mt-1 text-sm text-slate-600">Newest runs appear first.</p>
          </div>
          <button
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
            onClick={() =>
              void apiClient.getEvaluationRuns().then(activeBenchmarkRuns).then(setRuns)
            }
            type="button"
          >
            Refresh
          </button>
        </div>

        {runs.length === 0 ? (
          <div className="mt-4 rounded-xl border border-dashed border-slate-300 p-8 text-center text-sm text-slate-500">
            No evaluation runs yet.
          </div>
        ) : (
          <div className="mt-4 space-y-4">
            {runs.map((run) => (
              <article
                className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"
                key={run.id}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="font-semibold text-slate-950">
                      {benchmarks.find((item) => item.id === run.request.benchmark)?.name ??
                        run.request.benchmark}
                    </h3>
                    <p className="mt-1 text-xs text-slate-500">
                      {run.request.model} · {date(run.created_at)}
                    </p>
                  </div>
                  <span
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyle(run.status)}`}
                  >
                    {run.status.replaceAll("_", " ")}
                  </span>
                </div>

                <p className="mt-3 text-xs text-slate-600">
                  Sampling seed: {savedSeed(run) ?? "Unavailable for this legacy run"}
                </p>
                {run.result?.retrieval ? (
                  <p className="mt-2 text-xs text-slate-600">
                    Run weights: vector {run.result.retrieval.dense_weight} · keyword{" "}
                    {run.result.retrieval.keyword_weight}
                  </p>
                ) : null}
                {run.result?.dataset?.sample_fingerprint ? (
                  <p className="mt-2 break-all text-xs text-slate-500">
                    Sample fingerprint: {run.result.dataset.sample_fingerprint}
                  </p>
                ) : null}
                {run.status !== "queued" && run.status !== "running" ? (
                  <div className="mt-3">
                    {run.status !== "succeeded" &&
                    savedSeed(run) !== null &&
                    run.result?.dataset?.sample_fingerprint ? (
                      <button
                        className="mr-3 rounded-lg border border-amber-200 px-3 py-2 text-sm font-semibold text-amber-700 disabled:opacity-60"
                        disabled={submitting}
                        onClick={() => void retryFailed(run.id)}
                        type="button"
                      >
                        Retry failed examples
                      </button>
                    ) : null}
                    <button
                      className="rounded-lg border border-cyan-200 px-3 py-2 text-sm font-semibold text-cyan-700 disabled:opacity-60"
                      disabled={submitting || savedSeed(run) === null}
                      onClick={() => void rerun(run)}
                      type="button"
                    >
                      Rerun same examples
                    </button>
                    <p className="mt-1 text-xs text-slate-500">
                      Reuses this run&apos;s seed, sample settings, and model with current retrieval
                      weights. Restart the API after changing tuning settings.
                    </p>
                    {run.status !== "succeeded" ? (
                      <p className="mt-1 text-xs text-slate-500">
                        Retry failed examples preserves successful scores and requires the original
                        retrieval settings.
                      </p>
                    ) : null}
                  </div>
                ) : null}

                {run.status === "queued" || run.status === "running" ? (
                  <div className="mt-4">
                    <div className="mb-1 flex justify-between text-xs text-slate-500">
                      <span>Progress</span>
                      <span>
                        {run.completed_examples}/{run.total_examples ?? run.request.limit}
                      </span>
                    </div>
                    <progress
                      className="h-2 w-full"
                      max={run.total_examples ?? run.request.limit}
                      value={run.completed_examples}
                    />
                    <button
                      className="mt-3 rounded-lg border border-red-200 bg-white px-3 py-1.5 text-xs font-semibold text-red-700 disabled:opacity-60"
                      disabled={cancellingRunId === run.id}
                      onClick={() => void cancelRun(run.id)}
                      type="button"
                    >
                      {cancellingRunId === run.id ? "Cancelling…" : "Cancel evaluation"}
                    </button>
                  </div>
                ) : null}

                {run.error ? <p className="mt-4 text-sm text-red-700">{run.error}</p> : null}

                {run.result ? (
                  <>
                    {run.result.summary ? (
                      <p className="mt-4 text-sm text-slate-700">
                        Scored {run.result.summary.scored_examples}/
                        {run.total_examples ?? run.request.limit}
                        {" · "}
                        {run.result.summary.retrying_examples} retrying
                        {" · "}
                        {run.result.summary.failed_examples} failed
                        {run.result.summary.failed_examples > 0
                          ? ". Scores cover successful examples only; this is an incomplete evaluation."
                          : ""}
                      </p>
                    ) : null}
                    {run.retry_of ? (
                      <p className="mt-2 text-xs text-slate-500">Recovery of run {run.retry_of}</p>
                    ) : null}
                    <dl className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                      {Object.entries(run.result.metrics)
                        .filter(([key]) => METRICS_BY_BENCHMARK[run.request.benchmark].has(key))
                        .map(([key, value]) => (
                          <div className="rounded-lg bg-slate-50 p-3" key={key}>
                            <dt className="text-xs font-medium text-slate-500">{label(key)}</dt>
                            <dd className="mt-1 text-xl font-semibold text-slate-900">
                              {key === "examples" ? value : score(value)}
                            </dd>
                          </div>
                        ))}
                    </dl>
                    <details className="mt-4 border-t border-slate-100 pt-4">
                      <summary className="cursor-pointer text-sm font-medium text-cyan-700">
                        View {run.result.examples.length} example results
                      </summary>
                      <pre className="mt-3 max-h-96 overflow-auto rounded-lg bg-slate-950 p-4 text-xs leading-5 text-slate-100">
                        {JSON.stringify(run.result.examples, null, 2)}
                      </pre>
                    </details>
                  </>
                ) : null}
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
