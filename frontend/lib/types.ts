export interface CreateProjectPayload {
  prompt: string;
  duration_seconds?: number | null;
  scene_count?: number | null;
  aspect_ratio: string;
  resolution: string;
  image_model?: MagicImageModel | null;
  image_resolution?: MagicImageResolution | null;
  video_model?: MagicVideoModel | null;
  video_resolution?: string | null;
}

export type MagicImageModel =
  | "default"
  | "flux-schnell"
  | "z-image-turbo"
  | "seedream-v4"
  | "nano-banana"
  | "nano-banana-2"
  | "nano-banana-pro";

export type MagicImageResolution = "640px" | "1k" | "2k" | "4k";

export type MagicVideoModel =
  | "default"
  | "ltx-2"
  | "ltx-2.3"
  | "wan-2.2"
  | "seedance"
  | "seedance-2.0"
  | "kling-2.5"
  | "kling-3.0"
  | "sora-2"
  | "veo3.1"
  | "veo3.1-lite"
  | "kling-1.6";

export interface ProjectMessagePayload {
  message: string;
}

export interface SceneSpec {
  id: string;
  narration: string;
  image_prompt: string;
  video_prompt: string;
  duration_seconds: number;
}

export interface VideoPlan {
  title: string;
  narration: string;
  aspect_ratio: string;
  resolution: string;
  scenes: SceneSpec[];
}

export interface GeneratedImage {
  scene_id: string;
  path: string;
  url?: string;
  prompt: string;
  model: string;
  provider_job_id?: string | null;
  provider_url?: string | null;
}

export interface GeneratedSegment {
  scene_id: string;
  path: string;
  url?: string;
  prompt: string;
  model: string;
  duration_seconds: number;
  provider_job_id?: string | null;
  provider_url?: string | null;
}

export interface TokenOutput {
  token_output_path: string;
  provider: string;
  model: string;
  usage: {
    requests: number;
    input_tokens: number;
    cached_input_tokens: number;
    uncached_input_tokens: number;
    output_tokens: number;
    reasoning_tokens: number;
    tool_search_tokens: number;
    total_tokens: number;
  };
  cost: {
    input_usd: number;
    cached_input_usd: number;
    output_usd: number;
    total_usd: number;
  };
}

export interface WorkflowManifest {
  project_id: string;
  title: string;
  created_at: string;
  aspect_ratio: string;
  resolution: string;
  image_model: string;
  image_resolution?: string;
  image_style_tool?: string;
  video_model: string;
  video_resolution?: string;
  video_audio?: boolean;
  audio_model: string;
  render_status: "complete" | "partial";
  completed_scene_count: number;
  failed_scene_count: number;
  failed_scenes: Array<{
    scene_id: string;
    stage: string;
    error: string;
  }>;
  plan: VideoPlan;
  images: GeneratedImage[];
  videos: GeneratedSegment[];
  voiceover: { path: string; url?: string; model: string; duration_seconds: number };
  token_output: TokenOutput;
  token_output_path: string;
  gpt_cost_usd: number;
  final_video_path: string;
  final_video_url?: string;
  manifest_path: string;
}

export interface ProjectState {
  version: number;
  project_id: string;
  created_at: string;
  updated_at: string;
  status: Record<string, unknown>;
  user_preferences: Record<string, unknown>;
  provider_settings: Record<string, unknown>;
  current_plan?: Record<string, unknown> | null;
  scene_assets: {
    voiceover?: WorkflowManifest["voiceover"] | null;
    images: GeneratedImage[];
    videos: GeneratedSegment[];
    final_video_path?: string | null;
    manifest_path?: string | null;
  };
  failures: WorkflowManifest["failed_scenes"];
  decisions: Array<{
    created_at: string;
    decision: string;
    rationale?: string;
    scene_id?: string;
    tool?: string;
    metadata?: Record<string, unknown>;
  }>;
  messages: Array<{
    created_at: string;
    role: "user" | "assistant" | string;
    content: string;
    metadata?: Record<string, unknown>;
  }>;
}

export interface CreateProjectResponse {
  project_id: string;
  status: ProjectStatus;
  stage: ProjectStage;
  progress: number;
  message: string;
  updated_at: string;
  status_url: string;
  manifest?: WorkflowManifest;
  project_state?: ProjectState;
  error?: string;
}

export type ProjectStatus = "queued" | "running" | "succeeded" | "failed";

export type ProjectStage =
  | "queued"
  | "planning"
  | "regenerate_scene"
  | "replace_voiceover"
  | "restitching"
  | "revise_narration"
  | "voiceover_images"
  | "video_generation"
  | "stitching"
  | "complete"
  | "failed"
  | "image_generation"
  | "message_complete"
  | "message_failed"
  | "message_queued"
  | "message_running"
  | "retry_scene"
  | "voiceover"
  | string;

export type ProjectStatusResponse = CreateProjectResponse;
