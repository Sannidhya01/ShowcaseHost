import { RepositoryChat } from "@/features/chat/repository-chat";

export default async function RepositoryChatPage({
  params,
}: {
  params: Promise<{ repositoryId: string }>;
}) {
  const { repositoryId } = await params;
  return <RepositoryChat repositoryId={repositoryId} />;
}
