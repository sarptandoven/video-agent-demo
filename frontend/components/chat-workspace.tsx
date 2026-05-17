"use client";

import { useMemo, useState } from "react";
import { Bot, ChevronDown, ImageIcon, Loader2, PanelRightOpen, Plus, Send, Settings2, Sparkles, Video } from "lucide-react";
import { Button } from "@/components/ui/button";
import { mediaUrl } from "@/lib/api";
import type {
  CreateProjectPayload,
  GeneratedImage,
  MagicImageModel,
  MagicImageResolution,
  MagicVideoModel,
  ProjectStatusResponse,
} from "@/lib/types";

export type ChatSubmitValues = CreateProjectPayload;

type OutputSelection =
  | { type: "story" }
  | { type: "image"; image: GeneratedImage; index: number }
  | { type: "final" };

type AutoNumber = number | "auto";
type ImageModelSelection = MagicImageModel | "auto";
type VideoModelSelection = MagicVideoModel | "auto";

const IMAGE_MODELS: Array<{ value: ImageModelSelection; label: string }> = [
  { value: "auto", label: "Agent chooses" },
  { value: "default", label: "Magic default" },
  { value: "seedream-v4", label: "Seedream v4" },
  { value: "z-image-turbo", label: "Z Image Turbo" },
  { value: "flux-schnell", label: "Flux Schnell" },
  { value: "nano-banana", label: "Nano Banana" },
  { value: "nano-banana-2", label: "Nano Banana 2" },
  { value: "nano-banana-pro", label: "Nano Banana Pro" },
];

const IMAGE_RESOLUTIONS: MagicImageResolution[] = ["640px", "1k", "2k", "4k"];

const VIDEO_MODELS: Array<{ value: VideoModelSelection; label: string }> = [
  { value: "auto", label: "Agent chooses" },
  { value: "default", label: "Magic default" },
  { value: "ltx-2.3", label: "LTX 2.3" },
  { value: "ltx-2", label: "LTX 2" },
  { value: "seedance", label: "Seedance" },
  { value: "seedance-2.0", label: "Seedance 2.0" },
  { value: "kling-2.5", label: "Kling 2.5" },
  { value: "kling-3.0", label: "Kling 3.0" },
  { value: "veo3.1", label: "Veo 3.1" },
  { value: "veo3.1-lite", label: "Veo 3.1 Lite" },
  { value: "sora-2", label: "Sora 2" },
  { value: "wan-2.2", label: "Wan 2.2" },
];

const VIDEO_RESOLUTIONS = ["480p", "720p", "1080p"] as const;
const ASPECT_RATIOS = ["9:16", "16:9", "1:1"] as const;

function statusText(job: ProjectStatusResponse | null) {
  if (!job) return "Ready";
  if (job.error) return job.error;
  return job.message;
}

function projectMessages(job: ProjectStatusResponse | null, localPrompt: string | null) {
  const messages = [...(job?.project_state?.messages ?? [])];
  if (messages.length === 0 && localPrompt) {
    messages.push({
      role: "user",
      content: localPrompt,
      created_at: job?.updated_at ?? new Date().toISOString(),
    });
  }
  if (job && messages.length === 1 && job.status !== "queued") {
    messages.push({
      role: "assistant",
      content: job.manifest?.title ? `I drafted and rendered "${job.manifest.title}".` : job.message,
      created_at: job.updated_at,
    });
  }
  return messages;
}

function storyPreview(job: ProjectStatusResponse | null) {
  const plan = job?.manifest?.plan ?? job?.project_state?.current_plan;
  if (!plan || typeof plan !== "object") return null;
  const title = "title" in plan && typeof plan.title === "string" ? plan.title : "Story";
  const narration = "narration" in plan && typeof plan.narration === "string" ? plan.narration : "";
  return { title, narration };
}

export function ChatWorkspace({
  job,
  isBusy,
  onCreate,
  onMessage,
  onNewCreate,
}: {
  job: ProjectStatusResponse | null;
  isBusy: boolean;
  onCreate: (values: ChatSubmitValues) => void;
  onMessage: (message: string) => void;
  onNewCreate: () => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [followUp, setFollowUp] = useState("");
  const [localPrompt, setLocalPrompt] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [selection, setSelection] = useState<OutputSelection>({ type: "story" });
  const [duration, setDuration] = useState<AutoNumber>("auto");
  const [sceneCount, setSceneCount] = useState<AutoNumber>("auto");
  const [aspectRatio, setAspectRatio] = useState<(typeof ASPECT_RATIOS)[number]>("9:16");
  const [resolution, setResolution] = useState<(typeof VIDEO_RESOLUTIONS)[number]>("720p");
  const [imageModel, setImageModel] = useState<ImageModelSelection>("seedream-v4");
  const [imageResolution, setImageResolution] = useState<MagicImageResolution>("1k");
  const [videoModel, setVideoModel] = useState<VideoModelSelection>("ltx-2.3");
  const [videoResolution, setVideoResolution] = useState<(typeof VIDEO_RESOLUTIONS)[number]>("720p");

  const messages = useMemo(() => projectMessages(job, localPrompt), [job, localPrompt]);
  const story = storyPreview(job);
  const images = job?.manifest?.images ?? job?.project_state?.scene_assets.images ?? [];
  const finalVideo = mediaUrl(job?.manifest?.final_video_url ?? job?.manifest?.final_video_path);
  const poster = mediaUrl(images[0]?.url ?? images[0]?.path);
  const canSendFollowUp = Boolean(job?.project_id) && !isBusy;

  const startNewCreate = () => {
    setPrompt("");
    setFollowUp("");
    setLocalPrompt(null);
    setSelection({ type: "story" });
    onNewCreate();
  };

  const submitInitial = () => {
    const cleaned = prompt.trim();
    if (!cleaned) return;
    setLocalPrompt(cleaned);
    setSelection({ type: "story" });
    onCreate({
      prompt: cleaned,
      duration_seconds: duration === "auto" ? null : duration,
      scene_count: sceneCount === "auto" ? null : sceneCount,
      aspect_ratio: aspectRatio,
      resolution,
      image_model: imageModel === "auto" ? null : imageModel,
      image_resolution: imageResolution,
      video_model: videoModel === "auto" ? null : videoModel,
      video_resolution: videoResolution,
    });
    setPrompt("");
  };

  const submitFollowUp = () => {
    const cleaned = followUp.trim();
    if (!cleaned || !canSendFollowUp) return;
    onMessage(cleaned);
    setFollowUp("");
  };

  return (
    <div className="grid min-h-0 flex-1 gap-4 px-4 py-4 lg:grid-cols-[minmax(360px,440px)_1fr]">
      <section className="flex min-h-0 flex-col rounded-lg border border-white/10 bg-zinc-950/80">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-md bg-white text-zinc-950">
              <Bot className="h-4 w-4" />
            </div>
            <div>
              <h1 className="text-sm font-semibold text-white">Video Agent</h1>
              <p className="text-xs text-zinc-400">{statusText(job)}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {job?.project_id && (
              <Button type="button" variant="outline" size="sm" onClick={startNewCreate} className="h-8 gap-1.5 px-2.5">
                <Plus className="h-3.5 w-3.5" />
                New
              </Button>
            )}
            {isBusy && <Loader2 className="h-4 w-4 animate-spin text-emerald-300" />}
          </div>
        </div>

        <div className="custom-scrollbar flex-1 space-y-3 overflow-y-auto px-4 py-4">
          {messages.map((message, index) => (
            <div key={`${message.created_at}-${index}`} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[86%] rounded-lg px-3 py-2 text-sm leading-relaxed ${message.role === "user" ? "bg-white text-zinc-950" : "bg-zinc-900 text-zinc-200 border border-white/10"}`}>
                {message.content}
              </div>
            </div>
          ))}
          {messages.length === 0 && (
            <div className="rounded-lg border border-dashed border-white/10 p-4 text-sm text-zinc-400">
              Describe the story or result you want. The agent will choose prompts, models, and render steps.
            </div>
          )}
        </div>

        <div className="border-t border-white/10 p-3">
          {!job?.project_id ? (
            <div className="space-y-3">
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submitInitial();
                  }
                }}
                className="h-28 w-full resize-none rounded-lg border border-white/10 bg-black px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-400"
                placeholder="Make a tense cinematic story about a courier crossing a neon city during a blackout..."
                disabled={isBusy}
              />
              <button type="button" onClick={() => setShowSettings(!showSettings)} className="flex w-full items-center justify-between rounded-md border border-white/10 px-3 py-2 text-xs text-zinc-300">
                <span className="flex items-center gap-2"><Settings2 className="h-3.5 w-3.5" /> Advanced controls</span>
                <ChevronDown className={`h-3.5 w-3.5 transition ${showSettings ? "rotate-180" : ""}`} />
              </button>
              {showSettings && (
                <div className="grid gap-2 rounded-lg border border-white/10 bg-black/50 p-3 text-xs sm:grid-cols-2">
                  <Select label="Image model" value={imageModel} onChange={(value) => setImageModel(value as ImageModelSelection)} options={IMAGE_MODELS} />
                  <Select label="Image res" value={imageResolution} onChange={(value) => setImageResolution(value as MagicImageResolution)} options={IMAGE_RESOLUTIONS.map((value) => ({ value, label: value }))} />
                  <Select label="Video model" value={videoModel} onChange={(value) => setVideoModel(value as VideoModelSelection)} options={VIDEO_MODELS} />
                  <Select label="Video res" value={videoResolution} onChange={(value) => setVideoResolution(value as typeof videoResolution)} options={VIDEO_RESOLUTIONS.map((value) => ({ value, label: value }))} />
                  <Select label="Aspect" value={aspectRatio} onChange={(value) => setAspectRatio(value as typeof aspectRatio)} options={ASPECT_RATIOS.map((value) => ({ value, label: value }))} />
                  <Select label="Output res" value={resolution} onChange={(value) => setResolution(value as typeof resolution)} options={VIDEO_RESOLUTIONS.map((value) => ({ value, label: value }))} />
                  <AutoNumberField label="Seconds" value={duration} min={5} max={60} onChange={setDuration} />
                  <AutoNumberField label="Scenes" value={sceneCount} min={1} max={10} onChange={setSceneCount} />
                </div>
              )}
              <Button type="button" onClick={submitInitial} disabled={isBusy || !prompt.trim()} className="w-full gap-2">
                {isBusy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                Generate
              </Button>
            </div>
          ) : (
            <div className="flex gap-2">
              <textarea
                value={followUp}
                onChange={(event) => setFollowUp(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    submitFollowUp();
                  }
                }}
                className="min-h-12 flex-1 resize-none rounded-lg border border-white/10 bg-black px-3 py-2 text-sm text-zinc-100 outline-none focus:border-emerald-400"
                placeholder="Ask for a revision..."
                disabled={!canSendFollowUp}
              />
              <Button type="button" onClick={submitFollowUp} disabled={!followUp.trim() || !canSendFollowUp} size="icon" aria-label="Send message">
                <Send className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      </section>

      <section className="grid min-h-0 gap-4 lg:grid-cols-[minmax(280px,360px)_1fr]">
        <OutputRail story={story} images={images} finalVideo={finalVideo} selection={selection} onSelect={setSelection} />
        <OutputDetail selection={selection} story={story} finalVideo={finalVideo} poster={poster} />
      </section>
    </div>
  );
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: Array<{ value: string; label: string }> }) {
  return (
    <label className="grid gap-1">
      <span className="text-zinc-500">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="h-9 rounded-md border border-white/10 bg-zinc-950 px-2 text-zinc-100 outline-none">
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function AutoNumberField({ label, value, min, max, onChange }: { label: string; value: AutoNumber; min: number; max: number; onChange: (value: AutoNumber) => void }) {
  const options = Array.from({ length: max - min + 1 }, (_, index) => min + index);
  return (
    <label className="grid gap-1">
      <span className="text-zinc-500">{label}</span>
      <select
        value={value === "auto" ? "auto" : String(value)}
        onChange={(event) => onChange(event.target.value === "auto" ? "auto" : Number(event.target.value))}
        className="h-9 rounded-md border border-white/10 bg-zinc-950 px-2 text-zinc-100 outline-none"
      >
        <option value="auto">Auto</option>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function OutputRail({ story, images, finalVideo, selection, onSelect }: { story: ReturnType<typeof storyPreview>; images: GeneratedImage[]; finalVideo: string; selection: OutputSelection; onSelect: (selection: OutputSelection) => void }) {
  return (
    <div className="custom-scrollbar min-h-0 overflow-y-auto rounded-lg border border-white/10 bg-zinc-950/70 p-3">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-white">Outputs</h2>
        <PanelRightOpen className="h-4 w-4 text-zinc-500" />
      </div>
      <div className="grid gap-2">
        {story && (
          <button type="button" onClick={() => onSelect({ type: "story" })} className={`rounded-lg border p-3 text-left ${selection.type === "story" ? "border-emerald-400 bg-emerald-500/10" : "border-white/10 bg-black/40"}`}>
            <div className="text-xs font-semibold text-emerald-300">Story</div>
            <div className="mt-1 line-clamp-2 text-sm text-zinc-200">{story.title}</div>
          </button>
        )}
        {images.map((image, index) => (
          <button key={`${image.scene_id}-${image.path}`} type="button" onClick={() => onSelect({ type: "image", image, index })} className={`overflow-hidden rounded-lg border text-left ${selection.type === "image" && selection.image.path === image.path ? "border-emerald-400" : "border-white/10"}`}>
            <img src={mediaUrl(image.url ?? image.path)} alt={`Scene ${index + 1}`} className="aspect-video w-full object-cover" />
            <div className="p-2 text-xs text-zinc-300">Image {index + 1}</div>
          </button>
        ))}
        {finalVideo && (
          <button type="button" onClick={() => onSelect({ type: "final" })} className={`rounded-lg border p-3 text-left ${selection.type === "final" ? "border-emerald-400 bg-emerald-500/10" : "border-white/10 bg-black/40"}`}>
            <div className="flex items-center gap-2 text-sm font-semibold text-white"><Video className="h-4 w-4" /> Final video</div>
          </button>
        )}
      </div>
    </div>
  );
}

function OutputDetail({ selection, story, finalVideo, poster }: { selection: OutputSelection; story: ReturnType<typeof storyPreview>; finalVideo: string; poster: string }) {
  return (
    <div className="min-h-0 rounded-lg border border-white/10 bg-zinc-950/70 p-4">
      {selection.type === "image" && (
        <div className="grid h-full content-start gap-4">
          <img src={mediaUrl(selection.image.url ?? selection.image.path)} alt={`Scene ${selection.index + 1}`} className="max-h-[62vh] w-full rounded-lg object-contain bg-black" />
          <div className="rounded-lg border border-white/10 bg-black/40 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold text-emerald-300"><ImageIcon className="h-4 w-4" /> Scene image {selection.index + 1}</div>
            <p className="text-sm leading-relaxed text-zinc-300">{selection.image.prompt}</p>
          </div>
        </div>
      )}
      {selection.type === "final" && finalVideo && (
        <video className="max-h-[75vh] w-full rounded-lg bg-black object-contain" controls src={finalVideo} poster={poster || undefined} />
      )}
      {selection.type === "story" && (
        <div className="grid content-start gap-4">
          <div className="rounded-lg border border-white/10 bg-black/40 p-4">
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-300">Story</div>
            <h2 className="text-xl font-semibold text-white">{story?.title ?? "Story will appear here"}</h2>
            <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-zinc-300">{story?.narration ?? "The agent will show the drafted story as soon as the plan exists."}</p>
          </div>
        </div>
      )}
    </div>
  );
}
