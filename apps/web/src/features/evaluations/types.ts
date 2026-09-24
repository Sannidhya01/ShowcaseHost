export type EvaluationBenchmarkId = "codequeries" | "repoqa";

export type EvaluationBenchmarkSplit = {
  id: string;
  label: string;
  config: string;
  examples: number;
  sampling_groups: string[];
};

export type EvaluationBenchmark = {
  id: EvaluationBenchmarkId;
  name: string;
  dataset_id: string;
  default_config: string;
  default_split: string;
  metric: string;
  github_url: string;
  dataset_url: string;
  paper_url: string;
  citation: string;
  total_examples: number;
  max_run_examples: number;
  splits: EvaluationBenchmarkSplit[];
};

export type EvaluationRunRequest = {
  benchmark: EvaluationBenchmarkId;
  model: string;
  config?: string;
  split?: string;
  offset: number;
  limit: number;
  seed?: string | number;
};

export type EvaluationRetrieval = {
  strategy: string;
  dense_weight: number;
  keyword_weight: number;
  candidate_limit_per_channel: number;
  rerank_candidate_limit: number;
  reranker_model: string;
  reranker_fallback_model: string;
  relevance_threshold: number;
  relative_relevance_threshold: number;
  result_limit: number;
  rrf_k: number;
  dense_backend: string;
  keyword_backend: string;
};

export type EvaluationResult = {
  summary?: {
    scored_examples: number;
    failed_examples: number;
    retrying_examples: number;
    retried_examples: number;
    score_scope: string;
  };
  metrics: Record<string, number>;
  dataset?: {
    sampling_seed?: string;
    sample_fingerprint?: string;
    sampled_example_ids?: string[];
  };
  retrieval?: EvaluationRetrieval;
  examples: Array<Record<string, unknown>>;
};

export type EvaluationRun = {
  id: string;
  status: "queued" | "running" | "succeeded" | "completed_with_errors" | "failed" | "cancelled";
  retry_of?: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  completed_examples: number;
  total_examples: number | null;
  request: EvaluationRunRequest;
  result: EvaluationResult | null;
  error: string | null;
};
