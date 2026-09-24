export type HealthResponse = {
  status: string;
  service: string;
  version: string;
};

export type ReadinessResponse = {
  status: "ready" | "degraded";
  checks: Record<string, "ok" | "skipped" | "unavailable">;
};

export type CurrentUser = {
  id: string;
  github_login: string;
  avatar_url: string | null;
};

export type Repository = {
  id: string;
  github_repository_id: number;
  full_name: string;
  private: boolean;
  default_branch: string;
  last_observed_sha: string | null;
  chat_ready: boolean;
  latest_ingestion: IngestionJob | null;
};

export type GenerationModel = {
  id: string;
  label: string;
  tier: "primary" | "faster";
  provider: "groq" | "openrouter";
  is_default: boolean;
};

export type ChatHistoryMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatSource = {
  number: number;
  chunk_id: string;
  path: string;
  language: string;
  start_line: number;
  end_line: number;
  snapshot_sha: string;
};

export type IngestionStatus = "queued" | "running" | "succeeded" | "failed";
export type EmbeddingStatus = "queued" | "running" | "succeeded" | "failed";

export type EmbeddingModel = {
  id: string;
  dimension: number;
  context_tokens: number;
};

export type IngestionJob = {
  id: string;
  repository_id: string;
  status: IngestionStatus;
  attempt_count: number;
  commit_sha: string | null;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  chunking_job_id: string | null;
  chunking_status: ChunkingStatus | null;
  chunking_error_code: string | null;
  chunking_error_message: string | null;
  embedding_job_id: string | null;
  embedding_status: EmbeddingStatus | null;
  embedding_error_code: string | null;
  embedding_error_message: string | null;
  embedding_progress: number | null;
  embedding_model: string | null;
};

export type ChunkingStatus = "queued" | "running" | "succeeded" | "failed";

export type ChunkingJob = {
  id: string;
  snapshot_id: string;
  profile_key: string;
  status: ChunkingStatus;
  attempt_count: number;
  discovered_files: number;
  supported_files: number;
  reused_files: number;
  skipped_files: number;
  failed_files: number;
  chunks_created: number;
  processed_bytes: number;
  adaptive_retries: number;
  has_warnings: boolean;
  language_timings_ms: Record<string, unknown>;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};
