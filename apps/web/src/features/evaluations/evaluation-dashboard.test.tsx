import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EvaluationDashboard } from "./evaluation-dashboard";

const benchmark = {
  id: "codequeries",
  name: "CodeQueries",
  dataset_id: "thepurpleowl/codequeries",
  default_config: "ideal",
  default_split: "test",
  metric: "Official retrieval relevance accuracy, precision, recall, and F1",
  github_url: "https://github.com/example/codequeries",
  dataset_url: "https://example.com/dataset",
  paper_url: "https://example.com/paper",
  citation: "Example citation",
  total_examples: 171346,
  max_run_examples: 100,
  splits: [
    {
      id: "train",
      label: "Train",
      config: "ideal",
      examples: 102962,
      sampling_groups: ["negative", "positive"],
    },
    {
      id: "validation",
      label: "Validation",
      config: "ideal",
      examples: 11183,
      sampling_groups: ["negative", "positive"],
    },
    {
      id: "test",
      label: "Test",
      config: "ideal",
      examples: 57201,
      sampling_groups: ["negative", "positive"],
    },
  ],
};

const retrieval = {
  strategy: "hybrid_rrf",
  dense_weight: 1.25,
  keyword_weight: 1,
  candidate_limit_per_channel: 32,
  rerank_candidate_limit: 20,
  reranker_model: "BAAI/bge-reranker-v2-m3",
  reranker_fallback_model: "BAAI/bge-reranker-large",
  relevance_threshold: 0.00017189,
  relative_relevance_threshold: 0.855,
  result_limit: 5,
  rrf_k: 60,
  dense_backend: "in_memory_cosine",
  keyword_backend: "in_memory_bm25",
};

describe("EvaluationDashboard", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("reports partial scores and retries only failed examples through the recovery endpoint", async () => {
    const partial = {
      id: "partial",
      status: "completed_with_errors",
      created_at: "2026-09-08T10:00:00Z",
      request: { benchmark: "codequeries", model: "openai/gpt-oss-20b", seed: "42", limit: 100 },
      result: {
        metrics: { examples: 99 },
        examples: [],
        dataset: { sample_fingerprint: "sample" },
        summary: {
          scored_examples: 99,
          failed_examples: 1,
          retrying_examples: 0,
          retried_examples: 1,
        },
      },
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/benchmarks")) return new Response(JSON.stringify([benchmark]));
      if (url.endsWith("/models")) return new Response(JSON.stringify([]));
      if (url.endsWith("/retrieval")) return new Response(JSON.stringify(retrieval));
      if (url.endsWith("/runs")) return new Response(JSON.stringify([partial]));
      if (url.endsWith("/runs/partial/retry-failed") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            ...partial,
            id: "recovered",
            retry_of: "partial",
            status: "succeeded",
            result: {
              ...partial.result,
              metrics: { examples: 100 },
              summary: { ...partial.result.summary, scored_examples: 100, failed_examples: 0 },
            },
          }),
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<EvaluationDashboard />);
    expect(await screen.findByText(/Scores cover successful examples only/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry failed examples" }));
    expect(await screen.findByText(/Scored 100\/100/)).toBeInTheDocument();
    const posts = fetchMock.mock.calls.filter(([, init]) => init?.method === "POST");
    expect(posts).toHaveLength(1);
    expect(String(posts[0][0])).toContain("/runs/partial/retry-failed");
  });

  it("loads the catalog and starts an evaluation run", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/internal/evaluations/retrieval")) {
        return new Response(JSON.stringify(retrieval), { status: 200 });
      }
      if (url.endsWith("/internal/evaluations/benchmarks")) {
        return new Response(JSON.stringify([benchmark]), { status: 200 });
      }
      if (url.endsWith("/internal/evaluations/models")) {
        return new Response(
          JSON.stringify([
            {
              id: "qwen/qwen3.8-27b",
              label: "Qwen",
              tier: "primary",
              is_default: false,
              provider: "groq",
            },
            {
              id: "openai/gpt-oss-20b",
              label: "GPT-OSS 20B",
              tier: "primary",
              is_default: true,
              provider: "openrouter",
            },
          ]),
          { status: 200 },
        );
      }
      if (url.endsWith("/internal/evaluations/runs") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            id: "run-1",
            status: "queued",
            created_at: "2026-08-27T10:00:00Z",
            started_at: null,
            finished_at: null,
            completed_examples: 0,
            total_examples: null,
            request: JSON.parse(String(init.body)),
            result: null,
            error: null,
          }),
          { status: 202 },
        );
      }
      if (url.endsWith("/internal/evaluations/runs/run-1/cancel") && init?.method === "POST") {
        return new Response(
          JSON.stringify({
            id: "run-1",
            status: "cancelled",
            created_at: "2026-08-27T10:00:00Z",
            started_at: null,
            finished_at: "2026-08-27T10:00:01Z",
            completed_examples: 0,
            total_examples: null,
            request: {
              benchmark: "codequeries",
              model: "qwen/qwen3.8-27b",
              config: "ideal",
              split: "test",
              offset: 0,
              limit: 50,
            },
            result: null,
            error: null,
          }),
          { status: 200 },
        );
      }
      if (url.endsWith("/internal/evaluations/runs")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<EvaluationDashboard />);

    expect(
      await screen.findByText("Official retrieval relevance accuracy, precision, recall, and F1"),
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "GPT-OSS 20B · OpenRouter" })).toBeInTheDocument();
    expect(screen.getByLabelText(/Model \(RepoQA generation only\)/)).toBeDisabled();
    expect(screen.getByText(/balanced across: negative, positive/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Run evaluation" }));

    await waitFor(() => expect(screen.getByText("queued")).toBeInTheDocument());
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
    expect(post).toBeDefined();
    expect(JSON.parse(String(post?.[1]?.body))).toMatchObject({
      benchmark: "codequeries",
      split: "test",
      limit: 50,
      config: "ideal",
    });
    expect(screen.getByRole("option", { name: "Validation (11,183)" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel evaluation" }));
    await waitFor(() => expect(screen.getByText("cancelled")).toBeInTheDocument());
  });

  it("tolerates a benchmark response from an API that has not restarted yet", async () => {
    const {
      splits: _splits,
      total_examples: _total,
      max_run_examples: _maximum,
      ...legacy
    } = benchmark;
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/internal/evaluations/retrieval")) {
        return new Response(JSON.stringify(retrieval), { status: 200 });
      }
      if (url.endsWith("/internal/evaluations/benchmarks")) {
        return new Response(JSON.stringify([legacy]), { status: 200 });
      }
      if (url.endsWith("/internal/evaluations/runs")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      if (url.endsWith("/internal/evaluations/models")) {
        return new Response(JSON.stringify([]), { status: 200 });
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<EvaluationDashboard />);

    expect(
      await screen.findByText("Official retrieval relevance accuracy, precision, recall, and F1"),
    ).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Test (100)" })).toBeInTheDocument();
    expect(screen.getByText(/Dataset total: not reported/)).toBeInTheDocument();
  });

  it("reruns a saved sample with its exact seed and current server weights", async () => {
    const request = {
      benchmark: "codequeries",
      model: "openai/gpt-oss-20b",
      config: "ideal",
      split: "validation",
      offset: 9,
      limit: 20,
      seed: "9223372036854775807",
    };
    const historical = {
      id: "old-run",
      status: "succeeded",
      created_at: "2026-09-01T10:00:00Z",
      completed_examples: 20,
      total_examples: 20,
      request,
      result: {
        metrics: { examples: 20 },
        examples: [],
        retrieval: { ...retrieval, dense_weight: 1, keyword_weight: 1.5 },
        dataset: { sample_fingerprint: "same-sample" },
      },
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/benchmarks")) return new Response(JSON.stringify([benchmark]));
      if (url.endsWith("/models")) return new Response(JSON.stringify([]));
      if (url.endsWith("/retrieval")) return new Response(JSON.stringify(retrieval));
      if (url.endsWith("/runs") && init?.method === "POST") {
        return new Response(
          JSON.stringify({ ...historical, id: "new-run", request: JSON.parse(String(init.body)) }),
          { status: 202 },
        );
      }
      if (url.endsWith("/runs")) return new Response(JSON.stringify([historical]));
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<EvaluationDashboard />);
    expect(await screen.findByText(/Sampling seed: 9223372036854775807/)).toBeInTheDocument();
    expect(
      screen.getByText(/Current hybrid retrieval: vector 1.25 · keyword 1/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Run weights: vector 1 · keyword 1.5/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Rerun same examples" }));
    await waitFor(() => {
      const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
      expect(JSON.parse(String(post?.[1]?.body))).toEqual(request);
    });
  });

  it("validates and submits an explicit seed without converting it to a JS number", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/benchmarks")) return new Response(JSON.stringify([benchmark]));
      if (url.endsWith("/models") || (url.endsWith("/runs") && !init?.method)) {
        return new Response(JSON.stringify([]));
      }
      if (url.endsWith("/retrieval")) return new Response(JSON.stringify(retrieval));
      if (init?.method === "POST") {
        return new Response(
          JSON.stringify({
            id: "seed-run",
            status: "succeeded",
            created_at: "2026-09-01T10:00:00Z",
            request: JSON.parse(String(init.body)),
            result: null,
          }),
        );
      }
      throw new Error(`Unexpected request: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<EvaluationDashboard />);
    const input = await screen.findByLabelText("Sampling seed (optional)");
    const button = screen.getByRole("button", { name: "Run evaluation" });
    fireEvent.change(input, { target: { value: "9223372036854775808" } });
    expect(button).toBeDisabled();
    fireEvent.change(input, { target: { value: "1.5" } });
    expect(button).toBeDisabled();
    fireEvent.change(input, { target: { value: "9223372036854775807" } });
    fireEvent.click(button);
    await waitFor(() => {
      const post = fetchMock.mock.calls.find(([, init]) => init?.method === "POST");
      expect(JSON.parse(String(post?.[1]?.body)).seed).toBe("9223372036854775807");
    });
  });
});
