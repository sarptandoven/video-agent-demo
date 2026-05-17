"use client";

import { Video } from "lucide-react";
import { toast } from "sonner";
import { ChatWorkspace, type ChatSubmitValues } from "@/components/chat-workspace";
import { useCreateProject, useProject, useSendProjectMessage } from "@/hooks/use-project";

export default function HomePage() {
  const createProject = useCreateProject();
  const projectId = createProject.data?.project_id ?? null;
  const { data: job, isError: isJobError } = useProject(projectId);
  const sendMessage = useSendProjectMessage(projectId);

  const currentJob = job ?? createProject.data ?? null;
  const isBusy =
    createProject.isPending ||
    sendMessage.isPending ||
    currentJob?.status === "queued" ||
    currentJob?.status === "running";

  if (isJobError) {
    toast.error("Could not read project status.", { id: "polling-error" });
  }

  const handleCreate = (values: ChatSubmitValues) => {
    createProject.mutate(values, {
      onSuccess: () => toast.success("Generation started."),
      onError: (error) => toast.error(error instanceof Error ? error.message : "Could not start generation."),
    });
  };

  const handleMessage = (message: string) => {
    sendMessage.mutate(
      { message },
      {
        onSuccess: () => toast.success("Message sent to agent."),
        onError: (error) => toast.error(error instanceof Error ? error.message : "Could not send message."),
      },
    );
  };

  const handleNewCreate = () => {
    createProject.reset();
    sendMessage.reset();
  };

  return (
    <main className="flex min-h-screen flex-col bg-cinematic text-zinc-50 font-sans selection:bg-emerald-500/30 selection:text-emerald-100">
      <header className="border-b border-white/10 bg-zinc-950/80">
        <div className="flex h-14 items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-white text-zinc-950">
              <Video className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">Local Video Composer</h1>
              <p className="text-xs text-zinc-500">Agent chat plus render outputs</p>
            </div>
          </div>
          <div className="hidden rounded-full border border-white/10 px-3 py-1 text-xs text-zinc-400 sm:block">
            {currentJob?.project_id ? currentJob.project_id.slice(0, 8) : "No project"}
          </div>
        </div>
      </header>

      <ChatWorkspace
        job={currentJob}
        isBusy={Boolean(isBusy)}
        onCreate={handleCreate}
        onMessage={handleMessage}
        onNewCreate={handleNewCreate}
      />
    </main>
  );
}
