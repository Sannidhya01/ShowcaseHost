import { clientEnv } from "@/config/env";
import type {
  ChatHistoryMessage,
  ChatSource,
  ChunkingJob,
  CurrentUser,
  EmbeddingModel,
  GenerationModel,
  HealthStatus,
  IngestionJob,
  Repository,
} from "@/types/api";
import type {
  EvaluationBenchmark,
  EvaluationRun,
  EvaluationRunRequest,
  EvaluationRetrieval,
} from "@/features/evaluations/types";

type ChatStreamHandlers = {
  onSources: (sources: ChatSource[]) => void;
  onDelta: (content: string) => void;
};

export class ApiClientError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${clientEnv.NEXT_PUBLIC_API_BASE_URL}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `API request failed: ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      message = body.detail ?? message;
    } catch {
      // Preserve the status-based fallback for non-JSON errors.
    }
    throw new ApiClientError(message, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const apiClient = {
  getHealth: () => requestJson<HealthStatus>("/health"),
  getCurrentUser: () => requestJson<CurrentUser>("/auth/me"),
  getRepositories: () => requestJson<Repository[]>("/repositories"),
  getEmbeddingModels: () => requestJson<EmbeddingModel[]>("/embedding-models"),
  getGenerationModels: () => requestJson<GenerationModel[]>("/generation-models"),
  getEvaluationBenchmarks: () =>
    requestJson<EvaluationBenchmark[]>("/internal/evaluations/benchmarks"),
  getEvaluationModels: () => requestJson<GenerationModel[]>("/internal/evaluations/models"),
  getEvaluationRetrieval: () => requestJson<EvaluationRetrieval>("/internal/evaluations/retrieval"),
  getEvaluationRuns: () => requestJson<EvaluationRun[]>("/internal/evaluations/runs"),
  getEvaluationRun: (runId: string) =>
    requestJson<EvaluationRun>(`/internal/evaluations/runs/${runId}`),
  startEvaluationRun: (request: EvaluationRunRequest) =>
    requestJson<EvaluationRun>("/internal/evaluations/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    }),
  cancelEvaluationRun: (runId: string) =>
    requestJson<EvaluationRun>(`/internal/evaluations/runs/${runId}/cancel`, {
      method: "POST",
    }),
  retryFailedEvaluationExamples: (runId: string) =>
    requestJson<EvaluationRun>(`/internal/evaluations/runs/${runId}/retry-failed`, {
      method: "POST",
    }),
  startIngestion: (repositoryId: string, embeddingModel: string | null) =>
    requestJson<IngestionJob>(`/repositories/${repositoryId}/ingestions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ embedding_model: embeddingModel }),
    }),
  getIngestion: (jobId: string) => requestJson<IngestionJob>(`/ingestions/${jobId}`),
  getChunkingJob: (jobId: string) => requestJson<ChunkingJob>(`/chunking-jobs/${jobId}`),
  logout: () => requestJson<void>("/auth/logout", { method: "POST" }),
  streamChat: async (
    repositoryId: string,
    model: string,
    message: string,
    history: ChatHistoryMessage[],
    handlers: ChatStreamHandlers,
    signal?: AbortSignal,
  ) => {
    const response = await fetch(
      `${clientEnv.NEXT_PUBLIC_API_BASE_URL}/repositories/${repositoryId}/chat`,
      {
        method: "POST",
        credentials: "include",
        headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
        body: JSON.stringify({ model, message, history }),
        signal,
      },
    );
    if (!response.ok) {
      let message = `Chat request failed: ${response.statusText}`;
      try {
        const body = (await response.json()) as { detail?: string };
        message = body.detail ?? message;
      } catch {
        // Preserve the status fallback for non-JSON errors.
      }
      throw new ApiClientError(message, response.status);
    }
    if (!response.body) throw new ApiClientError("Chat stream was unavailable");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done }).replaceAll("\r\n", "\n");
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        let event = "message";
        const data: string[] = [];
        for (const line of frame.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).trim());
        }
        if (data.length === 0) continue;
        const payload = JSON.parse(data.join("\n")) as unknown;
        if (event === "sources") handlers.onSources(payload as ChatSource[]);
        if (event === "delta") handlers.onDelta((payload as { content: string }).content);
        if (event === "error") {
          throw new ApiClientError((payload as { message: string }).message);
        }
      }
      if (done) break;
    }
  },
};

export const apiUrl = (path: string) => `${clientEnv.NEXT_PUBLIC_API_BASE_URL}${path}`;
