import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getProject, createProject, sendProjectMessage } from "@/lib/api";
import type { ProjectStatusResponse, CreateProjectPayload, ProjectMessagePayload } from "@/lib/types";

export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: ["project", projectId],
    queryFn: async () => {
      if (!projectId) throw new Error("No project ID");
      return getProject(projectId);
    },
    enabled: !!projectId,
    refetchInterval: (query) => {
      const data = query.state.data as ProjectStatusResponse | undefined;
      const isRunning = data?.status === "queued" || data?.status === "running";
      return isRunning ? 2000 : false;
    },
  });
}

export function useCreateProject() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: CreateProjectPayload) => createProject(data),
    onSuccess: (data) => {
      queryClient.setQueryData(["project", data.project_id], data);
    },
  });
}

export function useSendProjectMessage(projectId: string | null) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: ProjectMessagePayload) => {
      if (!projectId) throw new Error("Start a project before sending follow-up messages.");
      return sendProjectMessage(projectId, data);
    },
    onSuccess: (data) => {
      queryClient.setQueryData(["project", data.project_id], data);
      queryClient.invalidateQueries({ queryKey: ["project", data.project_id] });
    },
  });
}
