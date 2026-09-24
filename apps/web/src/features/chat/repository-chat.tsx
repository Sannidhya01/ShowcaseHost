"use client";

import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { FormEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { ChatHistoryMessage, ChatSource, GenerationModel, Repository } from "@/types/api";
import {
  clearChatSession,
  type DisplayMessage,
  getChatSession,
  setChatSession,
} from "./chat-session-store";

export function RepositoryChat({ repositoryId }: { repositoryId: string }) {
  const [repository, setRepository] = useState<Repository | null>(null);
  const [models, setModels] = useState<GenerationModel[]>([]);
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<DisplayMessage[]>(() => getChatSession(repositoryId));
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(true);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    Promise.all([apiClient.getRepositories(), apiClient.getGenerationModels()])
      .then(([repositories, availableModels]) => {
        const current = repositories.find((item) => item.id === repositoryId) ?? null;
        setRepository(current);
        setModels(availableModels);
        setModel(
          availableModels.find((item) => item.is_default)?.id ?? availableModels[0]?.id ?? "",
        );
        if (!current) setError("Repository not found or no longer accessible");
      })
      .catch((caught: unknown) =>
        setError(caught instanceof Error ? caught.message : "Unable to load repository chat"),
      )
      .finally(() => setLoading(false));
  }, [repositoryId]);

  useEffect(() => {
    setChatSession(
      repositoryId,
      messages.filter((message) => !message.error),
    );
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, repositoryId]);

  const history = useMemo<ChatHistoryMessage[]>(
    () =>
      messages
        .filter((message) => !message.error && message.content)
        .slice(-12)
        .map(({ role, content }) => ({ role, content })),
    [messages],
  );

  async function submit(event?: FormEvent) {
    event?.preventDefault();
    const prompt = question.trim();
    if (!prompt || streaming || !model) return;
    setQuestion("");
    setError(null);
    setStreaming(true);
    const priorHistory = history;
    setMessages((current) => [
      ...current,
      { role: "user", content: prompt },
      { role: "assistant", content: "", sources: [] },
    ]);
    try {
      await apiClient.streamChat(repositoryId, model, prompt, priorHistory, {
        onSources: (sources: ChatSource[]) =>
          setMessages((current) => {
            const next = [...current];
            next[next.length - 1] = { ...next[next.length - 1], sources };
            return next;
          }),
        onDelta: (content: string) =>
          setMessages((current) => {
            const next = [...current];
            const last = next[next.length - 1];
            next[next.length - 1] = { ...last, content: last.content + content };
            return next;
          }),
      });
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to generate an answer";
      setError(message);
      setMessages((current) => {
        const next = [...current];
        const last = next[next.length - 1];
        next[next.length - 1] = {
          ...last,
          content: last.content || message,
          error: true,
        };
        return next;
      });
    } finally {
      setStreaming(false);
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  function cleanChat() {
    clearChatSession(repositoryId);
    setMessages([]);
    setError(null);
  }

  const repositoryName = repository?.full_name ?? "Repository";
  const hasConversation = messages.length > 0;

  return (
    <main className="min-h-screen bg-[#090b10] text-slate-100">
      <header className="fixed inset-x-0 top-0 z-20 border-b border-white/10 bg-[#090b10]/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5">
          <div className="flex min-w-0 items-center gap-4">
            <Link
              aria-label="Return to repositories"
              className="rounded-lg border border-white/10 px-3 py-1.5 text-sm text-slate-300 hover:bg-white/5"
              href="/"
            >
              ← Repositories
            </Link>
            <span
              className={`truncate font-mono text-sm transition-all duration-500 ${
                hasConversation ? "translate-y-0 opacity-100" : "-translate-y-3 opacity-0"
              }`}
            >
              {repositoryName}
            </span>
          </div>
          <button
            className="rounded-lg px-3 py-1.5 text-sm text-slate-400 hover:bg-white/5 hover:text-white disabled:opacity-40"
            disabled={!hasConversation || streaming}
            onClick={cleanChat}
            type="button"
          >
            Clean Chat
          </button>
        </div>
      </header>

      <section className="mx-auto flex min-h-screen max-w-3xl flex-col px-5 pb-48 pt-24">
        {loading ? <p className="text-sm text-slate-500">Loading repository…</p> : null}
        {!loading && !hasConversation ? (
          <div className="flex flex-1 items-center justify-center pb-28">
            <div className="text-center transition-all duration-500">
              <p className="mb-3 text-xs uppercase tracking-[0.3em] text-cyan-400">
                Ask the codebase
              </p>
              <h1 className="font-mono text-2xl text-slate-200">{repositoryName}</h1>
              <p className="mt-3 text-sm text-slate-500">
                Grounded answers with file and line citations
              </p>
            </div>
          </div>
        ) : null}

        <div className="space-y-8" aria-live="polite">
          {messages.map((message, index) => (
            <article
              className={message.role === "user" ? "ml-auto max-w-2xl" : "max-w-3xl"}
              key={`${message.role}-${index}`}
            >
              <p className="mb-2 text-xs uppercase tracking-wider text-slate-500">
                {message.role === "user" ? "You" : "ShowcaseHost"}
              </p>
              <div
                className={`text-[15px] leading-7 ${
                  message.role === "user"
                    ? "rounded-2xl bg-white/8 px-4 py-3 text-slate-200"
                    : message.error
                      ? "text-rose-300"
                      : "text-slate-300"
                }`}
              >
                {message.role === "assistant" && message.content ? (
                  <ReactMarkdown
                    components={{
                      a: ({ children, ...props }) => (
                        <a
                          className="text-cyan-400 underline"
                          rel="noreferrer"
                          target="_blank"
                          {...props}
                        >
                          {children}
                        </a>
                      ),
                      blockquote: ({ children }) => (
                        <blockquote className="my-3 border-l-2 border-cyan-500/50 pl-4 text-slate-400">
                          {children}
                        </blockquote>
                      ),
                      code: ({ children, className, ...props }) => (
                        <code
                          className={`${className ?? ""} rounded bg-black/40 px-1.5 py-0.5 font-mono text-[0.9em] text-cyan-100`}
                          {...props}
                        >
                          {children}
                        </code>
                      ),
                      h1: ({ children }) => (
                        <h1 className="mb-3 mt-5 text-xl font-semibold text-white">{children}</h1>
                      ),
                      h2: ({ children }) => (
                        <h2 className="mb-2 mt-5 text-lg font-semibold text-white">{children}</h2>
                      ),
                      h3: ({ children }) => (
                        <h3 className="mb-2 mt-4 font-semibold text-white">{children}</h3>
                      ),
                      li: ({ children }) => <li className="ml-5 list-disc pl-1">{children}</li>,
                      ol: ({ children }) => <ol className="my-3 space-y-1">{children}</ol>,
                      p: ({ children }) => <p className="my-2 first:mt-0 last:mb-0">{children}</p>,
                      pre: ({ children }) => (
                        <pre className="my-3 overflow-x-auto rounded-xl border border-white/10 bg-black/40 p-4 text-sm">
                          {children}
                        </pre>
                      ),
                      ul: ({ children }) => <ul className="my-3 space-y-1">{children}</ul>,
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  message.content || (streaming && index === messages.length - 1 ? "Thinking…" : "")
                )}
              </div>
              {message.sources?.length ? (
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {message.sources.map((source) => (
                    <div
                      className="rounded-lg border border-white/10 bg-white/[0.025] px-3 py-2 font-mono text-xs text-slate-400"
                      key={source.chunk_id}
                    >
                      <span className="mr-2 text-cyan-400">[{source.number}]</span>
                      {source.path}:{source.start_line}-{source.end_line}
                    </div>
                  ))}
                </div>
              ) : null}
            </article>
          ))}
          <div ref={endRef} />
        </div>
      </section>

      <div className="fixed inset-x-0 bottom-0 z-20 bg-gradient-to-t from-[#090b10] via-[#090b10] to-transparent px-5 pb-6 pt-12">
        <form
          className="mx-auto max-w-3xl rounded-2xl border border-white/10 bg-[#11141b] p-3 shadow-2xl shadow-black/40"
          onSubmit={(event) => void submit(event)}
        >
          <textarea
            aria-label="Ask about this repository"
            className="min-h-16 w-full resize-none bg-transparent px-2 py-1 text-sm leading-6 text-white outline-none placeholder:text-slate-600"
            disabled={streaming || !repository}
            maxLength={4000}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={onComposerKeyDown}
            placeholder="Ask about architecture, code paths, dependencies, or defects…"
            value={question}
          />
          <div className="flex items-center justify-between gap-3 border-t border-white/8 pt-3">
            <select
              aria-label="Generation model"
              className="max-w-52 rounded-lg bg-white/5 px-2 py-1.5 text-xs text-slate-300 outline-none"
              disabled={streaming}
              onChange={(event) => setModel(event.target.value)}
              value={model}
            >
              {models.map((item) => (
                <option className="bg-slate-900" key={item.id} value={item.id}>
                  {item.label} · {item.provider === "openrouter" ? "OpenRouter" : "Groq"}
                </option>
              ))}
            </select>
            <button
              className="rounded-lg bg-cyan-400 px-4 py-2 text-sm font-semibold text-slate-950 disabled:opacity-40"
              disabled={!question.trim() || streaming || !model}
              type="submit"
            >
              {streaming ? "Answering…" : "Send"}
            </button>
          </div>
          {error ? <p className="px-2 pt-2 text-xs text-rose-300">{error}</p> : null}
        </form>
      </div>
    </main>
  );
}
