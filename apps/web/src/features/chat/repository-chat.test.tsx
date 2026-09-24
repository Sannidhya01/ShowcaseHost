import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { clearChatSession } from "./chat-session-store";
import { RepositoryChat } from "./repository-chat";

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "OK",
    json: async () => body,
  };
}

function streamResponse(frames: string[]) {
  const encoder = new TextEncoder();
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    body: new ReadableStream({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode(frame));
        controller.close();
      },
    }),
  };
}

describe("RepositoryChat", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    clearChatSession("repo-id");
    Element.prototype.scrollIntoView = vi.fn();
  });

  it("streams an answer, renders citations, switches models, and cleans chat", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "repo-id",
            github_repository_id: 1,
            full_name: "octocat/hello",
            private: false,
            default_branch: "main",
            last_observed_sha: "a".repeat(40),
            chat_ready: true,
          },
        ]),
      )
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "qwen/qwen3.8-27b",
            label: "Qwen 3.8 27B",
            tier: "primary",
            is_default: true,
          },
          {
            id: "openai/gpt-oss-20b",
            label: "GPT-OSS 20B",
            tier: "faster",
            is_default: false,
          },
        ]),
      )
      .mockResolvedValueOnce(
        streamResponse([
          'event: sources\ndata: [{"number":1,"chunk_id":"chunk","path":"src/main.py","language":"python","start_line":5,"end_line":9,"snapshot_sha":"aaaaaaaa"}]\n\n',
          'event: delta\ndata: {"content":"The **entrypoint** "}\n\n',
          'event: delta\ndata: {"content":"is main [1]."}\n\n',
          "event: done\ndata: {}\n\n",
        ]),
      );
    vi.stubGlobal("fetch", fetchMock);

    render(<RepositoryChat repositoryId="repo-id" />);

    expect(await screen.findByRole("heading", { name: "octocat/hello" })).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Generation model"), {
      target: { value: "openai/gpt-oss-20b" },
    });
    fireEvent.change(screen.getByLabelText("Ask about this repository"), {
      target: { value: "Where is the entrypoint?" },
    });
    fireEvent.keyDown(screen.getByLabelText("Ask about this repository"), {
      key: "Enter",
      shiftKey: false,
    });

    expect(await screen.findByText("entrypoint")).toHaveProperty("tagName", "STRONG");
    expect(screen.getByText(/is main \[1\]\./)).toBeInTheDocument();
    expect(screen.getByText("src/main.py:5-9")).toBeInTheDocument();
    const request = fetchMock.mock.calls[2];
    expect(request[0]).toContain("/repositories/repo-id/chat");
    expect(JSON.parse(request[1].body)).toMatchObject({
      model: "openai/gpt-oss-20b",
      message: "Where is the entrypoint?",
      history: [],
    });

    fireEvent.click(screen.getByRole("button", { name: "Clean Chat" }));
    await waitFor(() => expect(screen.queryByText("entrypoint")).not.toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "octocat/hello" })).toBeInTheDocument();
  });

  it("shows a recoverable API error", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "repo-id",
            github_repository_id: 1,
            full_name: "octocat/hello",
            private: false,
            default_branch: "main",
            last_observed_sha: null,
            chat_ready: true,
          },
        ]),
      )
      .mockResolvedValueOnce(
        jsonResponse([
          {
            id: "qwen/qwen3.8-27b",
            label: "Qwen 3.8 27B",
            tier: "primary",
            is_default: true,
          },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse({ detail: "Repository retrieval is unavailable" }, 502));
    vi.stubGlobal("fetch", fetchMock);
    render(<RepositoryChat repositoryId="repo-id" />);

    const composer = await screen.findByLabelText("Ask about this repository");
    fireEvent.change(composer, { target: { value: "Explain it" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findAllByText("Repository retrieval is unavailable")).toHaveLength(2);
    expect(screen.getByLabelText("Ask about this repository")).not.toBeDisabled();
    fireEvent.change(screen.getByLabelText("Ask about this repository"), {
      target: { value: "Try again" },
    });
    expect(screen.getByRole("button", { name: "Send" })).not.toBeDisabled();
  });
});
