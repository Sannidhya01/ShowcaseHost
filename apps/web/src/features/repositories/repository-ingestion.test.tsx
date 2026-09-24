import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { RepositoryIngestion } from "./repository-ingestion";

const push = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ useRouter: () => ({ push }) }));

function response(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 401 ? "Unauthorized" : "OK",
    json: async () => body,
  };
}

describe("RepositoryIngestion", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    push.mockReset();
  });

  it("offers GitHub sign-in when unauthenticated", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ detail: "Authentication required" }, 401)),
    );

    render(<RepositoryIngestion />);

    expect(await screen.findByRole("link", { name: "Sign in with GitHub" })).toHaveAttribute(
      "href",
      "http://localhost:8000/auth/github/start",
    );
  });

  it("lists repositories and starts ingestion", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(response({ id: "user", github_login: "octocat", avatar_url: null }))
      .mockResolvedValueOnce(
        response([
          {
            id: "repo-id",
            github_repository_id: 1,
            full_name: "octocat/hello",
            private: false,
            default_branch: "main",
            last_observed_sha: null,
            chat_ready: false,
            latest_ingestion: null,
          },
        ]),
      )
      .mockResolvedValueOnce(
        response([{ id: "BAAI/bge-m3", dimension: 1024, context_tokens: 8192 }]),
      )
      .mockResolvedValueOnce(
        response({
          id: "job-id",
          repository_id: "repo-id",
          status: "succeeded",
          attempt_count: 1,
          commit_sha: "a".repeat(40),
          error_code: null,
          error_message: null,
          created_at: new Date().toISOString(),
          started_at: null,
          finished_at: new Date().toISOString(),
          chunking_job_id: "chunking-job-id",
          chunking_status: "succeeded",
          embedding_job_id: "embedding-job-id",
          embedding_status: "succeeded",
          embedding_progress: 1,
          embedding_model: "BAAI/bge-m3",
        }),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<RepositoryIngestion />);
    expect(await screen.findByText("Not Ready")).toBeInTheDocument();
    fireEvent.click(await screen.findByRole("button", { name: "Chat with repository" }));

    await waitFor(() =>
      expect(screen.getByText("Ready", { selector: "span" })).toBeInTheDocument(),
    );
    const snapshot = screen.getByText("aaaaaaaaaaaa");
    expect(snapshot).not.toBeVisible();
    fireEvent.click(screen.getByLabelText("Repository options for octocat/hello"));
    expect(snapshot).toBeVisible();
    expect(screen.getByText("Ready · BAAI/bge-m3")).toBeVisible();
    await waitFor(() => expect(push).toHaveBeenCalledWith("/repositories/repo-id/chat"));
    expect(fetchMock).toHaveBeenLastCalledWith(
      "http://localhost:8000/repositories/repo-id/ingestions",
      expect.objectContaining({ method: "POST", credentials: "include" }),
    );
  });

  it("shows a single processing status while pipeline details remain hidden", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(response({ id: "user", github_login: "octocat", avatar_url: null }))
        .mockResolvedValueOnce(
          response([
            {
              id: "repo-id",
              github_repository_id: 1,
              full_name: "octocat/processing",
              private: true,
              default_branch: "develop",
              last_observed_sha: null,
              chat_ready: false,
              latest_ingestion: {
                id: "job-id",
                repository_id: "repo-id",
                status: "succeeded",
                attempt_count: 1,
                commit_sha: "c".repeat(40),
                error_code: null,
                error_message: null,
                created_at: new Date().toISOString(),
                started_at: null,
                finished_at: new Date().toISOString(),
                chunking_job_id: "chunk-id",
                chunking_status: "running",
                chunking_error_code: null,
                chunking_error_message: null,
                embedding_job_id: null,
                embedding_status: null,
                embedding_error_code: null,
                embedding_error_message: null,
                embedding_progress: null,
                embedding_model: null,
              },
            },
          ]),
        )
        .mockResolvedValueOnce(response([])),
    );

    render(<RepositoryIngestion />);

    expect(await screen.findByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("cccccccccccc")).not.toBeVisible();
    expect(screen.queryByText(/Code chunking running/)).not.toBeInTheDocument();
  });

  it("restores and displays a durable chunking failure reason", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(response({ id: "user", github_login: "octocat", avatar_url: null }))
        .mockResolvedValueOnce(
          response([
            {
              id: "repo-id",
              github_repository_id: 1,
              full_name: "octocat/empty",
              private: false,
              default_branch: "main",
              last_observed_sha: null,
              chat_ready: false,
              latest_ingestion: {
                id: "job-id",
                repository_id: "repo-id",
                status: "succeeded",
                attempt_count: 1,
                commit_sha: "b".repeat(40),
                error_code: null,
                error_message: null,
                created_at: new Date().toISOString(),
                started_at: null,
                finished_at: new Date().toISOString(),
                chunking_job_id: "chunk-id",
                chunking_status: "failed",
                chunking_error_code: "no_chunkable_files",
                chunking_error_message: "Snapshot contains no successfully chunked code files",
                embedding_job_id: null,
                embedding_status: null,
                embedding_progress: null,
                embedding_model: null,
              },
            },
          ]),
        )
        .mockResolvedValueOnce(response([])),
    );

    render(<RepositoryIngestion />);

    expect(await screen.findByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Snapshot contains no successfully chunked code files")).toHaveClass(
      "text-red-700",
    );
    expect(screen.getByText("bbbbbbbbbbbb")).not.toBeVisible();
    expect(screen.queryByText("Embedding model (advanced)")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Repository options for octocat/empty")).toBeInTheDocument();
  });
});
