import type { ChatSource } from "@/types/api";

export type DisplayMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: ChatSource[];
  error?: boolean;
};

const sessions = new Map<string, DisplayMessage[]>();

export function getChatSession(repositoryId: string): DisplayMessage[] {
  return sessions.get(repositoryId) ?? [];
}

export function setChatSession(repositoryId: string, messages: DisplayMessage[]): void {
  sessions.set(repositoryId, messages);
}

export function clearChatSession(repositoryId: string): void {
  sessions.delete(repositoryId);
}
