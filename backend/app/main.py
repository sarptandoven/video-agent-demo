from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

from agents import Agent, ModelSettings, RunContextWrapper, Runner, ToolSearchTool, function_tool, tool_namespace
from agents.usage import Usage, serialize_usage
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .render_state import (
    append_project_message,
    append_project_decision,
    artifact_path,
    clear_scene_failures,
    initialize_project_state as write_initial_project_state,
    ordered_scene_assets,
    read_json_artifact,
    read_project_state,
    record_scene_failures,
    remove_json_artifact,
    update_project_state,
    upsert_scene_assets,
    write_json_artifact,
)
from .tools import (
    ProjectContext,
    generate_image_asset,
    generate_video_asset,
    generate_video_assets_batch,
    generate_voiceover_asset,
    pick_download,
    probe_media_duration,
    stitch_assets,
)


ROOT = Path(__file__).resolve().parents[2]
SHARED_ENV = Path("/Users/tanmay/Magic Hour ML role/.env")


def env() -> dict[str, str]:
    values = {key: value for key, value in dotenv_values(SHARED_ENV).items() if value is not None}
    values.update({key: value for key, value in dotenv_values(ROOT / ".env").items() if value is not None})
    values.update(os.environ)
    return values


ENV = env()
OUTPUT_DIR = Path(ENV.get("OUTPUT_DIR", ROOT / "outputs")).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OPENAI_API_KEY", ENV.get("OPENAI_API_KEY", ""))
logging.basicConfig(level=ENV.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("video-agent")

OPENAI_TEXT_PRICING_USD_PER_1M = {
    "gpt-5.5": {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 30.00,
        "long_context_threshold_input_tokens": 272_000,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.5/",
    },
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.00,
        "long_context_threshold_input_tokens": 272_000,
        "long_context_input_multiplier": 2.0,
        "long_context_output_multiplier": 1.5,
        "source": "https://developers.openai.com/api/docs/models/gpt-5.4/",
    }
}


AspectRatio = Literal["9:16", "16:9", "1:1"]
Resolution = Literal["480p", "720p", "1080p"]
MagicImageModel = Literal[
    "default",
    "flux-schnell",
    "z-image-turbo",
    "seedream-v4",
    "nano-banana",
    "nano-banana-2",
    "nano-banana-pro",
]
MagicImageResolution = Literal["640px", "1k", "2k", "4k"]
MagicImageStyleTool = Literal[
    "general",
    "ai-photo-generator",
    "ai-character-generator",
    "ai-landscape-generator",
    "ai-illustration-generator",
    "ai-art-generator",
    "movie-poster-generator",
    "architecture-generator",
    "ai-background-generator",
]
MagicVideoModel = Literal[
    "default",
    "ltx-2",
    "ltx-2.3",
    "wan-2.2",
    "seedance",
    "seedance-2.0",
    "kling-2.5",
    "kling-3.0",
    "sora-2",
    "veo3.1",
    "veo3.1-lite",
    "kling-1.6",
]
ProjectStatus = Literal["queued", "running", "succeeded", "failed"]
ProgressCallback = Callable[[str, int, str], Awaitable[None]]
PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
FISH_AUDIO_BRACKET_CUE_PATTERN = re.compile(r"\[([^\[\]\n]{1,80})\]")
FISH_AUDIO_LEGACY_PAREN_CUE_PATTERN = re.compile(r"\(([a-z][a-z -]{0,32})\)")
PROMPT_DURATION_UNDER_PATTERN = re.compile(r"\b(?:keep\s+it\s+)?under\s+(\d{1,2})\s+seconds?\b", re.IGNORECASE)
PROMPT_DURATION_PATTERN = re.compile(r"\b(\d{1,2})(?:\s*-\s*|\s+)seconds?\b", re.IGNORECASE)
PROMPT_NUMBER_WORDS: dict[str, int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
PROMPT_COUNT_TOKEN = r"\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten"
PROMPT_EXACT_SCENE_PATTERNS = (
    re.compile(rf"\bexactly\s+(?P<count>{PROMPT_COUNT_TOKEN})\s+(?:distinct\s+|unique\s+)?(?:scenes?|stages?)\b", re.IGNORECASE),
    re.compile(rf"\buse\s+(?P<count>{PROMPT_COUNT_TOKEN})\s+(?:distinct\s+|unique\s+)?(?:scenes?|stages?)\b", re.IGNORECASE),
    re.compile(rf"\binclude\s+(?P<count>{PROMPT_COUNT_TOKEN})\s+(?:distinct\s+|unique\s+)?(?:scenes?|stages?)\b", re.IGNORECASE),
)
PROMPT_MIN_SCENE_PATTERN = re.compile(
    rf"\bat\s+least\s+(?P<count>{PROMPT_COUNT_TOKEN})\s+(?:distinct\s+|unique\s+)?(?:scenes?|stages?)\b",
    re.IGNORECASE,
)
PROMPT_AGENT_DECIDES_SCENE_PATTERNS = (
    re.compile(r"\b(?:you\s+(?:must\s+)?)?(?:decide|determine|choose)\b.{0,100}\b(?:number\s+of\s+scenes|how\s+many\s+scenes|scene\s+count)\b", re.IGNORECASE),
    re.compile(r"\b(?:number\s+of\s+scenes|how\s+many\s+scenes|scene\s+count)\b.{0,100}\b(?:you\s+)?(?:decide|determine|choose)\b", re.IGNORECASE),
)
DEFAULT_AUTO_DURATION_SECONDS = 15
DEFAULT_AUTO_SCENE_BUDGET_COUNT = 4
DEFAULT_MAGIC_HOUR_IMAGE_MODEL: MagicImageModel = "seedream-v4"
DEFAULT_MAGIC_HOUR_VIDEO_MODEL: MagicVideoModel = "ltx-2.3"
DEFAULT_AGENT_MAX_TURNS = 30
DEFAULT_TTS_WORDS_PER_SECOND = 2.8
MIN_TTS_WORDS_PER_SECOND = 1.6
MAX_TTS_WORDS_PER_SECOND = 3.6
NARRATION_MIN_FACTOR = 0.9
NARRATION_MAX_FACTOR = 1.0
SCENE_CROSSFADE_SECONDS = 0.5
REQUIRED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "MAGIC_HOUR_API_KEY",
    "FISH_AUDIO_API_KEY",
    "FISH_AUDIO_REFERENCE_ID",
)
REQUIRED_SYSTEM_COMMANDS = ("ffmpeg", "ffprobe")
PROJECTS: dict[str, dict[str, Any]] = {}

MAGIC_IMAGE_MODELS: tuple[MagicImageModel, ...] = (
    "default",
    "flux-schnell",
    "z-image-turbo",
    "seedream-v4",
    "nano-banana",
    "nano-banana-2",
    "nano-banana-pro",
)
MAGIC_IMAGE_RESOLUTIONS: tuple[MagicImageResolution, ...] = ("640px", "1k", "2k", "4k")
MAGIC_IMAGE_STYLE_TOOLS: tuple[MagicImageStyleTool, ...] = (
    "general",
    "ai-photo-generator",
    "ai-character-generator",
    "ai-landscape-generator",
    "ai-illustration-generator",
    "ai-art-generator",
    "movie-poster-generator",
    "architecture-generator",
    "ai-background-generator",
)
MAGIC_VIDEO_MODELS: tuple[MagicVideoModel, ...] = (
    "default",
    "ltx-2",
    "ltx-2.3",
    "wan-2.2",
    "seedance",
    "seedance-2.0",
    "kling-2.5",
    "kling-3.0",
    "sora-2",
    "veo3.1",
    "veo3.1-lite",
    "kling-1.6",
)
MAGIC_IMAGE_MODEL_RESOLUTIONS: dict[str, set[str]] = {
    "flux-schnell": {"640px", "1k", "2k"},
    "z-image-turbo": {"640px", "1k", "2k"},
    "seedream-v4": {"640px", "1k", "2k", "4k"},
    "nano-banana": {"640px", "1k"},
    "nano-banana-2": {"640px", "1k", "2k", "4k"},
    "nano-banana-pro": {"1k", "2k", "4k"},
}
MAGIC_VIDEO_MODEL_RESOLUTIONS: dict[str, set[str]] = {
    "ltx-2": {"480p", "720p", "1080p"},
    "ltx-2.3": {"480p", "720p", "1080p"},
    "wan-2.2": {"480p", "720p", "1080p"},
    "seedance": {"480p", "720p", "1080p"},
    "seedance-2.0": {"480p", "720p"},
    "kling-2.5": {"720p", "1080p"},
    "kling-3.0": {"720p", "1080p"},
    "sora-2": {"720p"},
    "veo3.1": {"720p", "1080p"},
    "veo3.1-lite": {"720p", "1080p"},
    "kling-1.6": {"720p", "1080p"},
}
MAGIC_VIDEO_MODEL_DURATIONS: dict[str, set[int]] = {
    "ltx-2": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30},
    "ltx-2.3": {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30},
    "wan-2.2": {3, 4, 5, 6, 7, 8, 9, 10, 15},
    "seedance": {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
    "seedance-2.0": {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    "kling-2.5": {5, 10},
    "kling-3.0": {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    "sora-2": {4, 8, 12, 24, 36, 48, 60},
    "veo3.1": {4, 6, 8, 16, 24, 32, 40, 48, 56},
    "veo3.1-lite": {8, 16, 24, 32, 40, 48, 56},
    "kling-1.6": {5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60},
}
IMAGE_PROMPT_DESCRIPTION = (
    "Provider prompt for the still image. Write a stable cinematic keyframe later to be used for image-to-video generations.: "
    "describe only what is visible in one frame, including subject identity, action pose, foreground/background, "
    "lighting, lens/framing, texture, palette, and continuity details. Avoid text, logos, UI, captions, "
    "multi-panel layouts, and anything the later video prompt must invent."
)
VIDEO_PROMPT_DESCRIPTION = (
    "Provider prompt for animating that exact still image. Use one camera move and at most one subject motion; "
    "only animate what already exists in the image. Use no cuts. Do not add new objects, locations, cuts, scene changes, "
    "transformations, text, or events that are not grounded in the keyframe."
)


class Scene(BaseModel):
    id: str
    narration: str
    image_prompt: str = Field(description=IMAGE_PROMPT_DESCRIPTION)
    video_prompt: str = Field(description=VIDEO_PROMPT_DESCRIPTION)
    duration_seconds: int = Field(ge=1, le=30)


class VideoPlan(BaseModel):
    title: str
    narration: str
    visual_bible: str = Field(default="", max_length=900)
    scenes: list[Scene] = Field(min_length=1, max_length=10)


class SceneNarrationRevision(BaseModel):
    scene_id: str
    narration: str


class CreateProjectRequest(BaseModel):
    prompt: str = Field(min_length=3, max_length=2_000)
    duration_seconds: int | None = Field(default=None, ge=1, le=60)
    scene_count: int | None = Field(default=None, ge=1, le=10)
    aspect_ratio: AspectRatio = "9:16"
    resolution: Resolution = "720p"
    image_model: MagicImageModel | None = None
    video_model: MagicVideoModel | None = None
    image_resolution: MagicImageResolution | None = None
    video_resolution: Resolution | None = None


class ProjectMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)


@dataclass(frozen=True)
class SpeechBudget:
    words_per_second: float
    min_words: int
    max_words: int
    scene_duration_total_seconds: float
    final_duration_seconds: int


SceneConstraintMode = Literal["exact", "minimum", "agent_decides"]


@dataclass(frozen=True)
class GenerationConstraints:
    duration_seconds: int
    duration_source: Literal["prompt", "request", "auto"]
    duration_is_upper_bound: bool
    scene_mode: SceneConstraintMode
    scene_count: int | None
    scene_source: Literal["prompt", "request", "auto"]
    scene_budget_count: int


INSTRUCTIONS = """
You are a cinematic art director and autonomous video production agent.
Do not ask clarification questions. Infer missing details, make taste decisions,
and own the creative loop and render decisions.

Use the video_studio tools as a bounded production loop:
1. Call draft_video_plan with the complete title, narration, visual bible, and
   scene list.
2. Call generate_voiceover and generate_scene_images after the plan is saved.
3. Call animate_scene_videos after images exist.
4. Call inspect_render_status whenever an asset step is incomplete or unclear.
5. Call retry_scene for failed or missing scene assets before final stitching.
6. Call stitch_final_video once there is at least one completed scene video and
   a voiceover.
7. Call record_project_decision for important creative choices, retry choices,
   or user-preference interpretations that should persist in project_state.json.
8. After inspecting a result, use revision tools for narrow patches:
   regenerate_scene for one bad scene, revise_narration for script edits,
   replace_voiceover for stale audio, and restitch_video to verify the patched
   edit.

Quality rules:
- Narration is spoken voiceover copy for Fish Audio: it should sound like a
  tiny story being told aloud, not camera direction, not image description, and
  not a production note.
- Write a specific, natural narration with tension, intention, and payoff. No
  hype filler, no "this video".
- Use 3-5 scenes unless the user explicitly asks for more.
- Image prompts should be concrete: subject, setting, light, composition,
  style, mood, and continuity details.
- Treat each image prompt as the stable cinematic keyframe that the video model
  will animate. It must describe what is visible in one frame, not a sequence.
- Video prompts should describe camera motion and subject motion only.
- Video prompts should animate only what already exists in that keyframe: one
  camera move, at most one subject motion, no cuts, no scene changes, and no
  new objects.
- Choose Magic Hour image and image-to-video models yourself when the user does
  not specify them. Use model-specific strengths and constraints from the user
  brief and tool parameter descriptions.
- Keep motion realistic and easy for image-to-video to follow.
- Avoid text, logos, captions, distorted hands, and impossible camera moves.
- For speed, prefer 4-6 second scenes. Use longer durations only if needed.
- Make the total close to the requested duration.
- Keep visual continuity across scenes: subject identity, palette, lens language,
  camera energy, and environmental details should feel intentionally directed.
- If the user's requested scene count conflicts with quality, choose the scene
  count that makes the best final video and explain that choice in the title or
  narration only if needed.
""".strip()


PLANNING_INSTRUCTIONS = """
You are a senior cinematic art director. Return one complete VideoPlan that
matches the supplied timing brief. Do not ask clarification questions.

Planning rules:
- The full narration must fit the spoken-word budget in the user brief.
- Narration is spoken voiceover copy for Fish Audio. It should tell a compact
  story with character intention, obstacle, change, and payoff.
- Do not write narration as image prompt prose, not camera direction, and not a production note.
  Avoid lens, wardrobe, lighting, blocking, and model-facing
  visual inventory in narration unless it matters to the spoken story.
- Scene narrations should be one short sentence each and should combine cleanly
  into the full narration.
- Add Fish Audio S2 expression cues in brackets at the start of narration
  sentences where useful, for example [whispers softly], [speaks calmly],
  [curious], [tense], [laughs quietly], or [emphasis]. Keep cues sparse and
  natural.
- Image prompts should be concrete: subject, setting, light, composition,
  style, mood, and important visual details.
- Video prompts must describe only camera or subject motion that can happen in
  the current still image.
- Treat image_prompt as a stable cinematic keyframe and video_prompt as a small,
  grounded motion instruction for that exact keyframe.
- Keep all prompts compact. Do not add captions, logos, text, impossible camera
  moves, or new continuity details not present in the plan.
""".strip()


app = FastAPI(title="Fast OpenAI Video Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media", StaticFiles(directory=str(OUTPUT_DIR)), name="media")


def missing_configuration() -> list[str]:
    return [key for key in REQUIRED_ENV_KEYS if not ENV.get(key)]


def missing_system_dependencies() -> list[str]:
    return [command for command in REQUIRED_SYSTEM_COMMANDS if shutil.which(command) is None]


def assert_runtime_ready() -> None:
    missing_config = missing_configuration()
    missing_dependencies = missing_system_dependencies()
    if missing_config or missing_dependencies:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Project is not ready to render locally.",
                "missing_config": missing_config,
                "missing_dependencies": missing_dependencies,
            },
        )


def project_dir_for(project_id: str) -> Path:
    if not PROJECT_ID_PATTERN.fullmatch(project_id):
        raise ValueError(f"Invalid project id: {project_id}")
    return (OUTPUT_DIR / project_id).resolve()


def public_media_path(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        relative = resolved.relative_to(OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"Media path is outside output directory: {resolved}") from exc
    return f"/media/{relative.as_posix()}"


def with_media_url(asset: dict[str, Any]) -> dict[str, Any]:
    payload = dict(asset)
    if payload.get("path"):
        payload["url"] = public_media_path(payload["path"])
    return payload


def status_file_for(project_id: str) -> Path:
    return project_dir_for(project_id) / "status.json"


async def update_project_status(
    project_id: str,
    *,
    status: ProjectStatus,
    stage: str,
    progress: int,
    message: str,
    manifest: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project_id": project_id,
        "status": status,
        "stage": stage,
        "progress": max(0, min(progress, 100)),
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status_url": f"/api/projects/{project_id}",
    }
    if manifest is not None:
        payload["manifest"] = manifest
    if error is not None:
        payload["error"] = error

    PROJECTS[project_id] = payload
    status_path = status_file_for(project_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    state_path = status_path.parent / "project_state.json"
    if state_path.exists():
        state_ctx = ProjectContext(project_id=project_id, project_dir=status_path.parent, aspect_ratio="", resolution="")
        update_project_state(
            state_ctx,
            status={
                "status": status,
                "stage": stage,
                "progress": payload["progress"],
                "message": message,
                **({"error": error} if error is not None else {}),
            },
        )
    return payload


def read_project_status(project_id: str) -> dict[str, Any] | None:
    project_dir = project_dir_for(project_id)
    state_ctx = ProjectContext(project_id=project_id, project_dir=project_dir, aspect_ratio="", resolution="")
    state = read_project_state(state_ctx) if (project_dir / "project_state.json").exists() else None
    if project_id in PROJECTS:
        payload = dict(PROJECTS[project_id])
        if state is not None:
            payload["project_state"] = state
        return payload

    status_path = project_dir / "status.json"
    if status_path.exists():
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if state is not None:
            payload["project_state"] = state
        return payload

    manifest_path = project_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = {
            "project_id": project_id,
            "status": "succeeded",
            "stage": "complete",
            "progress": 100,
            "message": "Video is ready.",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status_url": f"/api/projects/{project_id}",
            "manifest": manifest,
        }
        if state is not None:
            payload["project_state"] = state
        return payload

    return None


def context(project_id: str, request: CreateProjectRequest) -> ProjectContext:
    return ProjectContext(
        project_id=project_id,
        project_dir=project_dir_for(project_id),
        aspect_ratio=request.aspect_ratio,
        resolution=request.video_resolution or request.resolution,
        magic_hour_api_key=ENV.get("MAGIC_HOUR_API_KEY", ""),
        fish_audio_api_key=ENV.get("FISH_AUDIO_API_KEY", ""),
        fish_audio_reference_id=ENV.get("FISH_AUDIO_REFERENCE_ID", ""),
        image_model=request.image_model or default_magic_hour_image_model(),
        image_resolution=request.image_resolution or ENV.get("MAGIC_HOUR_IMAGE_RESOLUTION", default_image_resolution(request.resolution)),
        image_style_tool=ENV.get("MAGIC_HOUR_IMAGE_STYLE_TOOL", "general"),
        video_model=request.video_model or default_magic_hour_video_model(),
        video_audio=ENV.get("MAGIC_HOUR_VIDEO_AUDIO", "false").lower() in {"1", "true", "yes"},
        audio_model=ENV.get("FISH_AUDIO_MODEL", "s2-pro"),
        audio_format=ENV.get("FISH_AUDIO_FORMAT", "mp3"),
    )


def context_for_existing_project(project_id: str) -> ProjectContext:
    project_dir = project_dir_for(project_id)
    state_ctx = ProjectContext(project_id=project_id, project_dir=project_dir, aspect_ratio="", resolution="")
    state = read_project_state(state_ctx)
    preferences = state.get("user_preferences") or {}
    providers = state.get("provider_settings") or {}
    return ProjectContext(
        project_id=project_id,
        project_dir=project_dir,
        aspect_ratio=str(providers.get("aspect_ratio") or preferences.get("aspect_ratio") or "9:16"),
        resolution=str(providers.get("resolution") or preferences.get("resolution") or "720p"),
        magic_hour_api_key=ENV.get("MAGIC_HOUR_API_KEY", ""),
        fish_audio_api_key=ENV.get("FISH_AUDIO_API_KEY", ""),
        fish_audio_reference_id=ENV.get("FISH_AUDIO_REFERENCE_ID", ""),
        image_model=str(providers.get("image_model") or default_magic_hour_image_model()),
        image_resolution=str(
            providers.get("image_resolution")
            or ENV.get("MAGIC_HOUR_IMAGE_RESOLUTION", default_image_resolution(str(providers.get("resolution") or preferences.get("resolution") or "720p")))
        ),
        image_style_tool=str(providers.get("image_style_tool") or ENV.get("MAGIC_HOUR_IMAGE_STYLE_TOOL", "general")),
        video_model=str(providers.get("video_model") or default_magic_hour_video_model()),
        video_audio=bool(providers.get("video_audio") or ENV.get("MAGIC_HOUR_VIDEO_AUDIO", "false").lower() in {"1", "true", "yes"}),
        audio_model=str(providers.get("audio_model") or ENV.get("FISH_AUDIO_MODEL", "s2-pro")),
        audio_format=str(providers.get("audio_format") or ENV.get("FISH_AUDIO_FORMAT", "mp3")),
    )


def default_image_resolution(video_resolution: str) -> MagicImageResolution:
    return {"480p": "640px", "720p": "1k", "1080p": "2k"}.get(video_resolution, "1k")  # type: ignore[return-value]


def explicit_magic_hour_default(value: str | None, fallback: str) -> str:
    configured = (value or "").strip()
    if not configured or configured == "default":
        return fallback
    return configured


def default_magic_hour_image_model() -> str:
    return explicit_magic_hour_default(ENV.get("MAGIC_HOUR_IMAGE_MODEL"), DEFAULT_MAGIC_HOUR_IMAGE_MODEL)


def default_magic_hour_video_model() -> str:
    return explicit_magic_hour_default(ENV.get("MAGIC_HOUR_VIDEO_MODEL"), DEFAULT_MAGIC_HOUR_VIDEO_MODEL)


def configured_agent_max_turns() -> int:
    raw = ENV.get("OPENAI_AGENT_MAX_TURNS")
    if not raw:
        return DEFAULT_AGENT_MAX_TURNS
    try:
        return max(11, min(int(raw), 80))
    except ValueError:
        logger.warning("Ignoring invalid OPENAI_AGENT_MAX_TURNS override: %s", raw)
        return DEFAULT_AGENT_MAX_TURNS


def user_preferences_for_request(request: CreateProjectRequest) -> dict[str, Any]:
    return request.model_dump(mode="json")


def provider_settings_for_context(ctx: ProjectContext) -> dict[str, Any]:
    return {
        "image_model": ctx.image_model,
        "image_resolution": ctx.image_resolution,
        "image_style_tool": ctx.image_style_tool,
        "video_model": ctx.video_model,
        "video_resolution": ctx.resolution,
        "video_audio": ctx.video_audio,
        "audio_model": ctx.audio_model,
        "audio_format": ctx.audio_format,
        "aspect_ratio": ctx.aspect_ratio,
        "resolution": ctx.resolution,
    }


def initialize_project_state(ctx: ProjectContext, request: CreateProjectRequest) -> dict[str, Any]:
    return write_initial_project_state(
        ctx,
        user_preferences=user_preferences_for_request(request),
        provider_settings=provider_settings_for_context(ctx),
    )


def ensure_project_state(ctx: ProjectContext, request: CreateProjectRequest) -> dict[str, Any]:
    if artifact_path(ctx, "project_state").exists():
        return read_project_state(ctx)
    return initialize_project_state(ctx, request)


def count_spoken_words(text: str) -> int:
    return len(WORD_PATTERN.findall(strip_fish_audio_expression_cues(text)))


def fish_audio_expression_cues(text: str) -> list[str]:
    return FISH_AUDIO_BRACKET_CUE_PATTERN.findall(text)


def strip_fish_audio_expression_cues(text: str) -> str:
    without_bracket_cues = FISH_AUDIO_BRACKET_CUE_PATTERN.sub(" ", text)
    return FISH_AUDIO_LEGACY_PAREN_CUE_PATTERN.sub(" ", without_bracket_cues)


def compact_words(text: str, max_words: int) -> str:
    cleaned = " ".join(text.split())
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words]).rstrip(" ,;:") + "."


def clamped_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def configured_tts_words_per_second() -> float | None:
    raw = ENV.get("FISH_AUDIO_WORDS_PER_SECOND") or ENV.get("TTS_WORDS_PER_SECOND")
    if not raw:
        return None
    try:
        return clamped_float(float(raw), MIN_TTS_WORDS_PER_SECOND, MAX_TTS_WORDS_PER_SECOND)
    except ValueError:
        logger.warning("Ignoring invalid TTS words-per-second override: %s", raw)
        return None


def estimate_tts_words_per_second(audio_model: str, output_dir: Path | None = None) -> float:
    configured = configured_tts_words_per_second()
    if configured is not None:
        return round(configured, 2)

    samples: list[float] = []
    root = output_dir or OUTPUT_DIR
    if root.exists():
        for manifest_path in root.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("audio_model") != audio_model:
                    continue
                narration = str((manifest.get("plan") or {}).get("narration") or manifest.get("narration") or "")
                word_count = count_spoken_words(narration)
                if word_count < 10:
                    continue
                voiceover_path = (manifest.get("voiceover") or {}).get("path")
                if not voiceover_path:
                    continue
                voiceover_file = Path(voiceover_path)
                if not voiceover_file.is_absolute():
                    voiceover_file = manifest_path.parent / voiceover_file
                if not voiceover_file.exists():
                    continue
                duration = probe_media_duration(voiceover_file)
                if duration < 3:
                    continue
                samples.append(word_count / duration)
            except Exception as exc:
                logger.debug("Skipping TTS calibration sample %s: %s", manifest_path, exc)

    if not samples:
        fallback = float(ENV.get("FISH_AUDIO_DEFAULT_WORDS_PER_SECOND", DEFAULT_TTS_WORDS_PER_SECOND))
        return round(clamped_float(fallback, MIN_TTS_WORDS_PER_SECOND, MAX_TTS_WORDS_PER_SECOND), 2)

    samples.sort()
    midpoint = len(samples) // 2
    if len(samples) % 2:
        median = samples[midpoint]
    else:
        median = (samples[midpoint - 1] + samples[midpoint]) / 2
    return round(clamped_float(median, MIN_TTS_WORDS_PER_SECOND, MAX_TTS_WORDS_PER_SECOND), 2)


def prompt_number_value(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    return PROMPT_NUMBER_WORDS.get(value.lower())


def bounded_prompt_count(value: int | None) -> int | None:
    if value is None or value < 1 or value > 10:
        return None
    return value


def extract_prompt_duration(prompt: str) -> tuple[int, bool] | None:
    under_match = PROMPT_DURATION_UNDER_PATTERN.search(prompt)
    if under_match:
        duration = int(under_match.group(1))
        if 1 <= duration <= 60:
            return duration, True

    duration_match = PROMPT_DURATION_PATTERN.search(prompt)
    if not duration_match:
        return None
    duration = int(duration_match.group(1))
    if 1 <= duration <= 60:
        return duration, False
    return None


def extract_prompt_scene_constraint(prompt: str) -> tuple[SceneConstraintMode, int | None] | None:
    for pattern in PROMPT_EXACT_SCENE_PATTERNS:
        exact_match = pattern.search(prompt)
        if exact_match:
            count = bounded_prompt_count(prompt_number_value(exact_match.group("count")))
            if count is not None:
                return "exact", count

    minimum_match = PROMPT_MIN_SCENE_PATTERN.search(prompt)
    if minimum_match:
        count = bounded_prompt_count(prompt_number_value(minimum_match.group("count")))
        if count is not None:
            return "minimum", count

    if any(pattern.search(prompt) for pattern in PROMPT_AGENT_DECIDES_SCENE_PATTERNS):
        return "agent_decides", None

    return None


def resolve_generation_constraints(request: CreateProjectRequest) -> GenerationConstraints:
    prompt_duration = extract_prompt_duration(request.prompt)
    if prompt_duration is not None:
        duration_seconds, duration_is_upper_bound = prompt_duration
        duration_source: Literal["prompt", "request", "auto"] = "prompt"
    elif request.duration_seconds is not None:
        duration_seconds = request.duration_seconds
        duration_is_upper_bound = False
        duration_source = "request"
    else:
        duration_seconds = DEFAULT_AUTO_DURATION_SECONDS
        duration_is_upper_bound = False
        duration_source = "auto"

    prompt_scene_constraint = extract_prompt_scene_constraint(request.prompt)
    if prompt_scene_constraint is not None:
        scene_mode, scene_count = prompt_scene_constraint
        scene_source: Literal["prompt", "request", "auto"] = "prompt"
    elif request.scene_count is not None:
        scene_mode = "exact"
        scene_count = request.scene_count
        scene_source = "request"
    else:
        scene_mode = "agent_decides"
        scene_count = None
        scene_source = "auto"

    scene_budget_count = scene_count if scene_count is not None else DEFAULT_AUTO_SCENE_BUDGET_COUNT
    return GenerationConstraints(
        duration_seconds=duration_seconds,
        duration_source=duration_source,
        duration_is_upper_bound=duration_is_upper_bound,
        scene_mode=scene_mode,
        scene_count=scene_count,
        scene_source=scene_source,
        scene_budget_count=scene_budget_count,
    )


def speech_budget_for_request(request: CreateProjectRequest, ctx: ProjectContext) -> SpeechBudget:
    constraints = resolve_generation_constraints(request)
    words_per_second = estimate_tts_words_per_second(ctx.audio_model)
    raw_scene_duration = constraints.duration_seconds + max(constraints.scene_budget_count - 1, 0) * SCENE_CROSSFADE_SECONDS
    min_words = max(4, math.floor(constraints.duration_seconds * words_per_second * NARRATION_MIN_FACTOR))
    max_words = max(min_words + 1, math.ceil(constraints.duration_seconds * words_per_second * NARRATION_MAX_FACTOR))
    return SpeechBudget(
        words_per_second=words_per_second,
        min_words=min_words,
        max_words=max_words,
        scene_duration_total_seconds=raw_scene_duration,
        final_duration_seconds=constraints.duration_seconds,
    )


def magic_hour_model_catalog_for_agent() -> str:
    return "\n".join(
        [
            "Magic Hour image models:",
            "- seedream-v4: detailed cinematic keyframes with strong descriptive prompt adherence at 640px/1k/2k/4k; default for this app.",
            "- default: Magic Hour recommended image model; do not use unless the user explicitly asks for Magic Hour's default.",
            "- flux-schnell: low-cost fast drafts at 640px/1k/2k.",
            "- z-image-turbo: low-cost fast drafts at 640px/1k/2k.",
            "- nano-banana: higher-cost image model for polished creative output at 640px/1k.",
            "- nano-banana-2: higher-cost model with broader image counts and up to 4k.",
            "- nano-banana-pro: highest-cost professional image model at 1k/2k/4k.",
            "Magic Hour image-to-video models:",
            "- ltx-2.3: default for this app; fast iteration with audio, lip-sync, and end frame support; supports 1-10/15/20/25/30 second clips and 480p/720p/1080p.",
            "- ltx-2: older fast-iteration LTX option with the same I2V duration and resolution set.",
            "- default: Magic Hour recommended video model; do not use unless the user explicitly asks for Magic Hour's default.",
            "- wan-2.2: fast strong visuals/effects, supports 3-10/15 second clips and 480p/720p/1080p.",
            "- seedance: fast iteration, supports 2-12 second clips and 480p/720p/1080p.",
            "- seedance-2.0: quality and consistency, supports 4-15 second clips and 480p/720p.",
            "- kling-2.5: motion/action/camera control, supports 5 or 10 second clips and 720p/1080p.",
            "- kling-3.0: cinematic multi-shot storytelling, supports 3-15 second clips and 720p/1080p.",
            "- sora-2: story-first creativity, supports 4/8/12/24/36/48/60 second clips and 720p.",
            "- veo3.1: realistic visuals and prompt adherence, supports 4/6/8/16/24/32/40/48/56 second clips and 720p/1080p.",
            "- veo3.1-lite: faster affordable high-quality video, supports 8/16/24/32/40/48/56 second clips and 720p/1080p.",
            "Supported I2V durations and resolutions are model-specific; choose scene durations and tool parameters that match the selected video model.",
        ]
    )


def duration_constraint_line(constraints: GenerationConstraints) -> str:
    if constraints.duration_source == "prompt":
        if constraints.duration_is_upper_bound:
            return f"Prompt duration constraint: under {constraints.duration_seconds} seconds. Treat this as a hard upper bound."
        return f"Prompt duration constraint: {constraints.duration_seconds} seconds. This overrides UI/default duration controls."
    if constraints.duration_source == "request":
        return f"User-selected duration constraint: {constraints.duration_seconds} seconds."
    return f"Duration constraint: agent decides. Use a compact {constraints.duration_seconds}-second budget unless the prompt demands different pacing."


def image_model_policy_line(request: CreateProjectRequest, ctx: ProjectContext) -> str:
    if request.image_model:
        return f"User-selected image model: {request.image_model}."
    return f"Default image model: {ctx.image_model}. Use another image model only if the user explicitly asks or the prompt clearly needs that model-specific capability."


def video_model_policy_line(request: CreateProjectRequest, ctx: ProjectContext) -> str:
    if request.video_model:
        return f"User-selected image-to-video model: {request.video_model}."
    return f"Default image-to-video model: {ctx.video_model}. Use another model only if the user explicitly asks or the prompt clearly needs that model-specific capability."


def scene_constraint_line(constraints: GenerationConstraints) -> str:
    if constraints.scene_mode == "exact" and constraints.scene_count is not None:
        return f"Scene count constraint: exactly {constraints.scene_count} scenes."
    if constraints.scene_mode == "minimum" and constraints.scene_count is not None:
        return f"Scene count constraint: at least {constraints.scene_count} scenes or stages."
    return "Scene count constraint: agent decides. Choose the count needed for clarity, usually 3-5 scenes."


def build_generation_brief(request: CreateProjectRequest, ctx: ProjectContext) -> str:
    constraints = resolve_generation_constraints(request)
    budget = speech_budget_for_request(request, ctx)
    return "\n".join(
        [
            f"Prompt: {request.prompt}",
            f"Aspect ratio: {request.aspect_ratio}",
            f"Resolution: {request.resolution}",
            image_model_policy_line(request, ctx),
            f"User-selected image resolution: {request.image_resolution or 'agent chooses'}.",
            video_model_policy_line(request, ctx),
            f"User-selected video resolution: {request.video_resolution or request.resolution}.",
            duration_constraint_line(constraints),
            scene_constraint_line(constraints),
            "Prompt constraints are authoritative: preserve exact counts, minimum stages, prohibitions, metaphors, start/end anchors, and lighting requirements from the prompt.",
            f"Target final runtime: {budget.final_duration_seconds} seconds after crossfades.",
            f"Scene duration total: {budget.scene_duration_total_seconds:.1f} seconds before crossfades.",
            f"Estimated Fish Audio pace: {budget.words_per_second:.2f} words/second.",
            f"Narration budget: {budget.min_words}-{budget.max_words} spoken words.",
            (
                "Narration is spoken voiceover copy for Fish Audio. Tell a compact story with character intention, "
                "obstacle, change, and payoff. Do not write narration as image prompt prose, not camera direction, "
                "and not a production note; keep visual inventory in image_prompt instead."
            ),
            (
                "Fish Audio S2 expression cues: include bracketed natural-language cues at sentence starts, "
                "such as [whispers softly], [speaks calmly], [curious], [tense], [laughs quietly], or [emphasis]. "
                "These cue tokens are not spoken words; keep the spoken words within budget."
            ),
            f"Scene duration formula: total scene seconds should equal final runtime + {SCENE_CROSSFADE_SECONDS:.1f} seconds for each transition between scenes.",
            "Set scene durations as integers that total as close as possible to the scene duration total.",
            "Each image_prompt should be concrete: subject, setting, light, composition, style, mood, and important visual details.",
            "For recurring characters, every image_prompt must repeat the full identity and outfit details; do not rely on relative wording like 'same woman' because each still image is generated independently.",
            "Put the complete recurring character identity in visual_bible so provider prompts can carry it into every independent image generation.",
            "Each video_prompt must describe only camera or subject motion that can happen in the current still image.",
            "Keep image_prompt under 75 words and video_prompt under 35 words.",
            "",
            magic_hour_model_catalog_for_agent(),
        ]
    )


def ensure_supported_image_options(model: str, resolution: str) -> None:
    if model not in MAGIC_IMAGE_MODELS:
        raise ValueError(f"Unsupported Magic Hour image model: {model}")
    if resolution not in MAGIC_IMAGE_RESOLUTIONS:
        raise ValueError(f"Unsupported Magic Hour image resolution: {resolution}")
    supported = MAGIC_IMAGE_MODEL_RESOLUTIONS.get(model)
    if supported is not None and resolution not in supported:
        raise ValueError(f"{model} supports image resolutions {sorted(supported)}, not {resolution}.")


def ensure_supported_video_options(model: str, resolution: str, scenes: list[Scene]) -> None:
    if model not in MAGIC_VIDEO_MODELS:
        raise ValueError(f"Unsupported Magic Hour image-to-video model: {model}")
    supported_resolutions = MAGIC_VIDEO_MODEL_RESOLUTIONS.get(model)
    if supported_resolutions is not None and resolution not in supported_resolutions:
        raise ValueError(f"{model} supports video resolutions {sorted(supported_resolutions)}, not {resolution}.")
    supported_durations = MAGIC_VIDEO_MODEL_DURATIONS.get(model)
    if supported_durations is None:
        return
    unsupported = [scene.duration_seconds for scene in scenes if scene.duration_seconds not in supported_durations]
    if unsupported:
        raise ValueError(
            f"{model} does not support scene duration(s) {sorted(set(unsupported))}. "
            f"Supported I2V durations: {sorted(supported_durations)}."
        )


def context_with_magic_image_settings(
    ctx: ProjectContext,
    *,
    model: str,
    image_resolution: str,
    image_style_tool: str,
) -> ProjectContext:
    ensure_supported_image_options(model, image_resolution)
    if image_style_tool not in MAGIC_IMAGE_STYLE_TOOLS:
        raise ValueError(f"Unsupported Magic Hour image style tool: {image_style_tool}")
    return replace(ctx, image_model=model, image_resolution=image_resolution, image_style_tool=image_style_tool)


def context_with_magic_video_settings(
    ctx: ProjectContext,
    *,
    model: str,
    resolution: str,
    audio: bool,
    scenes: list[Scene],
) -> ProjectContext:
    ensure_supported_video_options(model, resolution, scenes)
    return replace(ctx, video_model=model, resolution=resolution, video_audio=audio)


def compact_project_status_for_agent(status: dict[str, Any] | None) -> dict[str, Any]:
    if not status:
        return {}
    return {
        key: value
        for key, value in status.items()
        if key not in {"project_state"}
    }


def build_project_message_brief(project_id: str, message: str, ctx: ProjectContext) -> str:
    snapshot = {
        "project_state": read_project_state(ctx),
        "status": compact_project_status_for_agent(read_project_status(project_id)),
    }
    return "\n".join(
        [
            "A user sent a follow-up message for an existing video project.",
            "Keep the frontend separate; this is a backend agent turn over persisted project state.",
            "Do not start from scratch unless the user explicitly requests a new render.",
            "Inspect the saved project state and artifacts before deciding what to patch.",
            "Use revision tools for narrow changes, then restitch_video when the final edit changes.",
            f"Default image model: {ctx.image_model}. Use another image model only if the user explicitly asks or the prompt clearly needs that model-specific capability.",
            f"Default image-to-video model: {ctx.video_model}. Use another model only if the user explicitly asks or the prompt clearly needs that model-specific capability.",
            f"Project directory: {ctx.project_dir}",
            f"Project state file: {artifact_path(ctx, 'project_state')}",
            "",
            magic_hour_model_catalog_for_agent(),
            "",
            "User message:",
            message,
            "",
            "Current project snapshot JSON:",
            json.dumps(snapshot, indent=2, default=str),
        ]
    )


def agent_response_content(value: Any) -> str:
    if value is None:
        return "Agent turn completed."
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)


def normalize_plan(plan: VideoPlan) -> VideoPlan:
    visual_bible = compact_words(plan.visual_bible, 60)
    scenes = []
    for index, scene in enumerate(plan.scenes, start=1):
        scenes.append(
            scene.model_copy(
                update={
                    "id": f"scene_{index}",
                    "image_prompt": " ".join(scene.image_prompt.split()),
                    "video_prompt": " ".join(scene.video_prompt.split()),
                }
            )
        )
    return plan.model_copy(update={"visual_bible": visual_bible, "scenes": scenes})


def provider_image_prompt(plan: VideoPlan, scene: Scene) -> str:
    prompt = " ".join(scene.image_prompt.split())
    visual_bible = " ".join(plan.visual_bible.split())
    if not visual_bible:
        return prompt
    if visual_bible.lower() in prompt.lower():
        return prompt
    return f"Continuity bible for every scene: {visual_bible}. Scene keyframe: {prompt}"


def narration_stats(plan: VideoPlan, voiceover: dict[str, Any]) -> dict[str, Any]:
    words = count_spoken_words(plan.narration)
    cues = fish_audio_expression_cues(plan.narration)
    duration = float(voiceover.get("duration_seconds") or 0)
    return {
        "word_count": words,
        "expression_cue_count": len(cues),
        "expression_cues": cues,
        "voiceover_duration_seconds": duration,
        "words_per_second": round(words / duration, 3) if duration > 0 else None,
    }


def pricing_key_for_model(model: str) -> str:
    if model in OPENAI_TEXT_PRICING_USD_PER_1M:
        return model
    for known_model in OPENAI_TEXT_PRICING_USD_PER_1M:
        if model.startswith(f"{known_model}-"):
            return known_model
    raise KeyError(f"No OpenAI pricing configured for model: {model}")


def token_detail_value(details: Any, key: str) -> int:
    if hasattr(details, key):
        return int(getattr(details, key) or 0)
    if isinstance(details, dict):
        return int(details.get(key) or 0)
    return 0


def token_output_payload(project_id: str, model: str, usage: Usage) -> dict[str, Any]:
    pricing = OPENAI_TEXT_PRICING_USD_PER_1M[pricing_key_for_model(model)]
    cached_input_tokens = token_detail_value(usage.input_tokens_details, "cached_tokens")
    reasoning_tokens = token_detail_value(usage.output_tokens_details, "reasoning_tokens")
    tool_search_tokens = token_detail_value(usage.input_tokens_details, "tool_search_tokens")
    uncached_input_tokens = max(usage.input_tokens - cached_input_tokens, 0)
    max_request_input_tokens = max(
        [usage.input_tokens, *(entry.input_tokens for entry in usage.request_usage_entries)],
        default=0,
    )
    long_context_applies = max_request_input_tokens > pricing["long_context_threshold_input_tokens"]
    input_multiplier = pricing["long_context_input_multiplier"] if long_context_applies else 1.0
    output_multiplier = pricing["long_context_output_multiplier"] if long_context_applies else 1.0

    input_cost = uncached_input_tokens * pricing["input"] * input_multiplier / 1_000_000
    cached_input_cost = cached_input_tokens * pricing["cached_input"] * input_multiplier / 1_000_000
    output_cost = usage.output_tokens * pricing["output"] * output_multiplier / 1_000_000
    total_cost = input_cost + cached_input_cost + output_cost

    return {
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "pricing": {
            "currency": "USD",
            "unit": "per_1m_tokens",
            "source": pricing["source"],
            "input": pricing["input"],
            "cached_input": pricing["cached_input"],
            "output": pricing["output"],
            "long_context_threshold_input_tokens": pricing["long_context_threshold_input_tokens"],
            "long_context_applies": long_context_applies,
            "input_multiplier": input_multiplier,
            "output_multiplier": output_multiplier,
        },
        "usage": {
            **serialize_usage(usage),
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "reasoning_tokens": reasoning_tokens,
            "tool_search_tokens": tool_search_tokens,
        },
        "cost": {
            "input_usd": round(input_cost, 8),
            "cached_input_usd": round(cached_input_cost, 8),
            "output_usd": round(output_cost, 8),
            "total_usd": round(total_cost, 8),
        },
        "scope": "OpenAI GPT/agent planning run only. Magic Hour, Fish Audio, and ffmpeg costs are not included.",
    }


def pending_token_output(ctx: ProjectContext, model: str) -> dict[str, Any]:
    path = ctx.project_dir / "token_output.json"
    return {
        "project_id": ctx.project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "openai",
        "model": model,
        "usage": {
            "requests": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "uncached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "tool_search_tokens": 0,
            "total_tokens": 0,
        },
        "cost": {
            "input_usd": 0,
            "cached_input_usd": 0,
            "output_usd": 0,
            "total_usd": 0,
        },
        "token_output_path": str(path),
        "scope": "Pending until the OpenAI agent run completes. Magic Hour, Fish Audio, and ffmpeg costs are not included.",
    }


def write_token_output(ctx: ProjectContext, usage: Usage, model: str | None = None) -> dict[str, Any]:
    path = ctx.project_dir / "token_output.json"
    payload = token_output_payload(ctx.project_id, model or video_agent.model, usage)
    payload["token_output_path"] = str(path)
    ctx.project_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def merge_token_output_into_manifest(ctx: ProjectContext, token_output: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ctx.project_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Agent finished without producing a video manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["token_output"] = token_output
    manifest["token_output_path"] = token_output["token_output_path"]
    manifest["gpt_cost_usd"] = token_output["cost"]["total_usd"]
    if manifest.get("final_video_path") and not manifest.get("final_video_url"):
        manifest["final_video_url"] = public_media_path(manifest["final_video_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    update_project_state(ctx, manifest=manifest)
    return manifest


async def plan_video(request: CreateProjectRequest, ctx: ProjectContext) -> tuple[VideoPlan, dict[str, Any]]:
    result = await Runner.run(
        planning_agent,
        input=build_generation_brief(request, ctx),
        context=ctx,
        max_turns=configured_agent_max_turns(),
    )
    token_output = write_token_output(ctx, result.context_wrapper.usage, model=planning_agent.model)
    plan = result.final_output if isinstance(result.final_output, VideoPlan) else VideoPlan.model_validate(result.final_output)
    return normalize_plan(plan), token_output


def load_video_plan(ctx: ProjectContext) -> VideoPlan:
    payload = read_json_artifact(ctx, "plan")
    if not payload:
        raise RuntimeError("No video plan found. Call draft_video_plan before rendering assets.")
    return VideoPlan.model_validate(payload)


def plan_duration_seconds(plan: VideoPlan) -> int:
    return sum(scene.duration_seconds for scene in plan.scenes)


def scene_ids_for(plan: VideoPlan, scene_ids: list[str] | None = None) -> set[str]:
    if not scene_ids:
        return {scene.id for scene in plan.scenes}
    known = {scene.id for scene in plan.scenes}
    requested = set(scene_ids)
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown scene id(s): {', '.join(unknown)}")
    return requested


def save_video_plan(ctx: ProjectContext, plan: VideoPlan) -> VideoPlan:
    write_json_artifact(ctx, "plan", plan.model_dump(mode="json"))
    update_project_state(ctx, current_plan=plan.model_dump(mode="json"))
    return plan


def patch_scene_in_plan(
    plan: VideoPlan,
    scene_id: str,
    *,
    narration: str | None = None,
    image_prompt: str | None = None,
    video_prompt: str | None = None,
    duration_seconds: int | None = None,
) -> tuple[VideoPlan, Scene]:
    patched_scenes: list[Scene] = []
    patched_scene: Scene | None = None
    for scene in plan.scenes:
        if scene.id != scene_id:
            patched_scenes.append(scene)
            continue
        updates: dict[str, Any] = {}
        if narration is not None:
            updates["narration"] = narration
        if image_prompt is not None:
            updates["image_prompt"] = " ".join(image_prompt.split())
        if video_prompt is not None:
            updates["video_prompt"] = " ".join(video_prompt.split())
        if duration_seconds is not None:
            updates["duration_seconds"] = duration_seconds
        patched_scene = scene.model_copy(update=updates)
        patched_scenes.append(patched_scene)
    if patched_scene is None:
        raise ValueError(f"Unknown scene id: {scene_id}")
    return plan.model_copy(update={"scenes": patched_scenes}), patched_scene


def revise_scene_narrations(plan: VideoPlan, revisions: list[SceneNarrationRevision]) -> VideoPlan:
    if not revisions:
        return plan
    revision_by_scene = {revision.scene_id: revision.narration for revision in revisions}
    unknown = sorted(set(revision_by_scene) - {scene.id for scene in plan.scenes})
    if unknown:
        raise ValueError(f"Unknown scene id(s): {', '.join(unknown)}")
    return plan.model_copy(
        update={
            "scenes": [
                scene.model_copy(update={"narration": revision_by_scene[scene.id]})
                if scene.id in revision_by_scene
                else scene
                for scene in plan.scenes
            ]
        }
    )


def invalidate_final_artifacts(ctx: ProjectContext, *, voiceover: bool = False) -> None:
    remove_json_artifact(ctx, "manifest")
    if voiceover:
        remove_json_artifact(ctx, "voiceover")
    update_project_state(ctx, final_video_path=None, manifest_path=None, **({"voiceover": None} if voiceover else {}))


def clear_render_outputs(ctx: ProjectContext) -> None:
    for artifact in ("voiceover", "images", "videos", "failed_scenes", "manifest"):
        remove_json_artifact(ctx, artifact)
    for directory in ("voiceover", "images", "videos"):
        shutil.rmtree(ctx.project_dir / directory, ignore_errors=True)
    for filename in ("final.mp4", "merged.mp4", "merged_timed.mp4"):
        with suppress(FileNotFoundError):
            (ctx.project_dir / filename).unlink()


def build_video_manifest(
    plan: VideoPlan,
    ctx: ProjectContext,
    *,
    images: list[dict[str, Any]],
    videos: list[dict[str, Any]],
    voiceover: dict[str, Any],
    failed_scenes: list[dict[str, str]],
    token_output: dict[str, Any],
    final_video: str,
) -> dict[str, Any]:
    plan_payload = plan.model_dump(mode="json")
    plan_payload["aspect_ratio"] = ctx.aspect_ratio
    plan_payload["resolution"] = ctx.resolution
    provider_settings = read_project_state(ctx).get("provider_settings") or {}

    manifest = {
        "project_id": ctx.project_id,
        "title": plan.title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aspect_ratio": ctx.aspect_ratio,
        "resolution": ctx.resolution,
        "image_model": provider_settings.get("image_model", ctx.image_model),
        "image_resolution": provider_settings.get("image_resolution", ctx.image_resolution),
        "image_style_tool": provider_settings.get("image_style_tool", ctx.image_style_tool),
        "video_model": provider_settings.get("video_model", ctx.video_model),
        "video_resolution": provider_settings.get("video_resolution", ctx.resolution),
        "video_audio": provider_settings.get("video_audio", ctx.video_audio),
        "audio_model": ctx.audio_model,
        "render_status": "partial" if failed_scenes else "complete",
        "completed_scene_count": len(videos),
        "failed_scene_count": len(failed_scenes),
        "failed_scenes": failed_scenes,
        "plan": plan_payload,
        "images": [with_media_url(image) for image in images],
        "videos": [with_media_url(video) for video in videos],
        "voiceover": with_media_url(voiceover),
        "narration_stats": narration_stats(plan, voiceover),
        "token_output": token_output,
        "token_output_path": token_output["token_output_path"],
        "gpt_cost_usd": token_output["cost"]["total_usd"],
        "final_video_path": final_video,
        "final_video_url": public_media_path(final_video),
        "manifest_path": str(ctx.project_dir / "manifest.json"),
    }
    write_json_artifact(ctx, "manifest", manifest)
    update_project_state(
        ctx,
        manifest=manifest,
        failures=failed_scenes,
        final_video_path=final_video,
        manifest_path=manifest["manifest_path"],
    )
    return manifest


async def draft_video_plan_impl(
    ctx: ProjectContext,
    title: str,
    narration: str,
    scenes: list[Scene],
    visual_bible: str = "",
    normalize_scene_ids: bool = True,
) -> dict[str, Any]:
    plan = VideoPlan(title=title, narration=narration, visual_bible=visual_bible, scenes=scenes)
    if normalize_scene_ids:
        plan = normalize_plan(plan)
    ctx.project_dir.mkdir(parents=True, exist_ok=True)
    clear_render_outputs(ctx)
    write_json_artifact(ctx, "plan", plan.model_dump(mode="json"))
    write_json_artifact(ctx, "failed_scenes", [])
    update_project_state(
        ctx,
        current_plan=plan.model_dump(mode="json"),
        voiceover=None,
        images=[],
        videos=[],
        failures=[],
        final_video_path=None,
        manifest_path=None,
        status={"stage": "plan_drafted", "progress": 15, "message": "Creative plan drafted."},
        decision={
            "tool": "draft_video_plan",
            "decision": f"Drafted plan '{plan.title}' with {len(plan.scenes)} scene(s).",
            "metadata": {"scene_ids": [scene.id for scene in plan.scenes]},
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "plan_drafted",
        "plan": plan.model_dump(mode="json"),
        "next_tools": ["generate_voiceover", "generate_scene_images"],
    }


async def generate_voiceover_impl(ctx: ProjectContext) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    voiceover = await generate_voiceover_asset(ctx, plan.narration, plan_duration_seconds(plan))
    write_json_artifact(ctx, "voiceover", voiceover)
    update_project_state(
        ctx,
        voiceover=voiceover,
        status={"stage": "voiceover_generated", "progress": 30, "message": "Voiceover generated."},
        decision={
            "tool": "generate_voiceover",
            "decision": "Generated voiceover for the saved narration.",
            "metadata": {"target_duration_seconds": voiceover.get("target_duration_seconds")},
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "voiceover_generated",
        "voiceover": with_media_url(voiceover),
        "next_tools": ["generate_scene_images", "animate_scene_videos"],
    }


async def generate_scene_images_impl(
    ctx: ProjectContext,
    scene_ids: list[str] | None = None,
    *,
    model: MagicImageModel | None = None,
    image_resolution: MagicImageResolution | None = None,
    image_style_tool: MagicImageStyleTool = "general",
) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    selected_ids = scene_ids_for(plan, scene_ids)
    scenes = [
        scene.model_copy(update={"image_prompt": provider_image_prompt(plan, scene)})
        for scene in plan.scenes
        if scene.id in selected_ids
    ]
    image_ctx = context_with_magic_image_settings(
        ctx,
        model=model or ctx.image_model,
        image_resolution=image_resolution or ctx.image_resolution,
        image_style_tool=image_style_tool or ctx.image_style_tool,
    )
    image_results = await asyncio.gather(
        *(generate_image_asset(image_ctx, scene) for scene in scenes),
        return_exceptions=True,
    )
    existing_images = read_json_artifact(ctx, "images", [])
    existing_failures = read_json_artifact(ctx, "failed_scenes", [])
    images: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for scene, result in zip(scenes, image_results):
        if isinstance(result, Exception):
            logger.warning(
                "Scene image generation failed for %s",
                scene.id,
                exc_info=(type(result), result, result.__traceback__),
            )
            failures.append({"scene_id": scene.id, "stage": "image_generation", "error": str(result)})
        else:
            images.append(result)

    merged_images = ordered_scene_assets(plan, upsert_scene_assets(existing_images, images))
    failure_free_images = {image["scene_id"] for image in images}
    updated_failures = clear_scene_failures(existing_failures, failure_free_images, {"image_generation"})
    updated_failures = record_scene_failures(updated_failures, failures)
    write_json_artifact(ctx, "images", merged_images)
    write_json_artifact(ctx, "failed_scenes", updated_failures)
    update_project_state(
        ctx,
        provider_settings={
            "image_model": image_ctx.image_model,
            "image_resolution": image_ctx.image_resolution,
            "image_style_tool": image_ctx.image_style_tool,
        },
        images=merged_images,
        failures=updated_failures,
        status={"stage": "images_generated", "progress": 45, "message": "Scene images generated."},
        decision={
            "tool": "generate_scene_images",
            "decision": f"Generated {len(images)} scene image(s).",
            "metadata": {
                "requested_scene_ids": [scene.id for scene in scenes],
                "failed_scene_ids": [failure["scene_id"] for failure in failures],
            },
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "images_generated",
        "images": [with_media_url(image) for image in merged_images],
        "failed_scenes": updated_failures,
        "next_tools": ["animate_scene_videos"],
    }


async def animate_scene_videos_impl(
    ctx: ProjectContext,
    scene_ids: list[str] | None = None,
    *,
    model: MagicVideoModel | None = None,
    resolution: Resolution | None = None,
    audio: bool | None = None,
) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    selected_ids = scene_ids_for(plan, scene_ids)
    selected_scenes = [scene for scene in plan.scenes if scene.id in selected_ids]
    video_ctx = context_with_magic_video_settings(
        ctx,
        model=model or ctx.video_model,
        resolution=resolution or ctx.resolution,
        audio=ctx.video_audio if audio is None else audio,
        scenes=selected_scenes,
    )
    existing_images = read_json_artifact(ctx, "images", [])
    image_by_scene = {image["scene_id"]: image for image in existing_images}
    video_scene_pairs = [
        (scene, image_by_scene[scene.id])
        for scene in selected_scenes
        if scene.id in image_by_scene
    ]
    missing_image_failures = [
        {"scene_id": scene.id, "stage": "video_generation", "error": "No image asset exists for this scene."}
        for scene in selected_scenes
        if scene.id not in image_by_scene
    ]
    if not video_scene_pairs:
        existing_failures = read_json_artifact(ctx, "failed_scenes", [])
        updated_failures = record_scene_failures(existing_failures, missing_image_failures)
        write_json_artifact(ctx, "failed_scenes", updated_failures)
        update_project_state(
            ctx,
            failures=updated_failures,
            status={"stage": "video_generation_blocked", "progress": 65, "message": "No scene images are ready for animation."},
        )
        raise RuntimeError("No scene images completed, so no videos can be animated.")

    video_results = await generate_video_assets_batch(video_ctx, video_scene_pairs)
    existing_videos = read_json_artifact(ctx, "videos", [])
    existing_failures = read_json_artifact(ctx, "failed_scenes", [])
    videos: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = [*missing_image_failures]

    for (scene, _image), result in zip(video_scene_pairs, video_results):
        if isinstance(result, Exception):
            logger.warning(
                "Scene video generation failed for %s",
                scene.id,
                exc_info=(type(result), result, result.__traceback__),
            )
            failures.append({"scene_id": scene.id, "stage": "video_generation", "error": str(result)})
        else:
            videos.append(result)

    merged_videos = ordered_scene_assets(plan, upsert_scene_assets(existing_videos, videos))
    successful_video_ids = {video["scene_id"] for video in videos}
    updated_failures = clear_scene_failures(existing_failures, successful_video_ids, {"video_generation"})
    updated_failures = record_scene_failures(updated_failures, failures)
    write_json_artifact(ctx, "videos", merged_videos)
    write_json_artifact(ctx, "failed_scenes", updated_failures)
    update_project_state(
        ctx,
        provider_settings={
            "video_model": video_ctx.video_model,
            "video_resolution": video_ctx.resolution,
            "video_audio": video_ctx.video_audio,
        },
        videos=merged_videos,
        failures=updated_failures,
        status={"stage": "videos_animated", "progress": 70, "message": "Scene videos animated."},
        decision={
            "tool": "animate_scene_videos",
            "decision": f"Animated {len(videos)} scene video(s).",
            "metadata": {
                "requested_scene_ids": [scene.id for scene, _image in video_scene_pairs],
                "failed_scene_ids": [failure["scene_id"] for failure in failures],
            },
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "videos_animated",
        "videos": [with_media_url(video) for video in merged_videos],
        "failed_scenes": updated_failures,
        "next_tools": ["stitch_final_video", "retry_scene"],
    }


async def stitch_final_video_impl(ctx: ProjectContext, token_output: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    images = ordered_scene_assets(plan, read_json_artifact(ctx, "images", []))
    videos = ordered_scene_assets(plan, read_json_artifact(ctx, "videos", []))
    voiceover = read_json_artifact(ctx, "voiceover")
    failed_scenes = read_json_artifact(ctx, "failed_scenes", [])
    if not videos:
        failures = "; ".join(
            f"{failure['scene_id']} {failure['stage']}: {failure['error']}"
            for failure in failed_scenes
        )
        detail = f" Failures: {failures}" if failures else ""
        raise RuntimeError(f"No scene videos completed, so no final MP4 can be stitched.{detail}")
    if not voiceover:
        raise RuntimeError("No voiceover asset found. Call generate_voiceover before stitching.")

    final_video = await stitch_assets(ctx, videos, voiceover)
    return build_video_manifest(
        plan,
        ctx,
        images=images,
        videos=videos,
        voiceover=voiceover,
        failed_scenes=failed_scenes,
        token_output=token_output or pending_token_output(ctx, ENV.get("OPENAI_MODEL", "gpt-5.5")),
        final_video=final_video,
    )


async def inspect_render_status_impl(ctx: ProjectContext) -> dict[str, Any]:
    artifacts = {
        name: artifact_path(ctx, name).exists()
        for name in ("plan", "voiceover", "images", "videos", "manifest")
    }
    plan_payload = read_json_artifact(ctx, "plan")
    images = read_json_artifact(ctx, "images", [])
    videos = read_json_artifact(ctx, "videos", [])
    failed_scenes = read_json_artifact(ctx, "failed_scenes", [])
    scene_ids: list[str] = []
    missing_images: list[str] = []
    missing_videos: list[str] = []

    if plan_payload:
        plan = VideoPlan.model_validate(plan_payload)
        scene_ids = [scene.id for scene in plan.scenes]
        image_ids = {image["scene_id"] for image in images}
        video_ids = {video["scene_id"] for video in videos}
        missing_images = [scene_id for scene_id in scene_ids if scene_id not in image_ids]
        missing_videos = [scene_id for scene_id in scene_ids if scene_id not in video_ids]

    next_tools = []
    if not artifacts["plan"]:
        next_tools.append("draft_video_plan")
    else:
        if not artifacts["voiceover"]:
            next_tools.append("generate_voiceover")
        if missing_images:
            next_tools.append("generate_scene_images")
        if missing_videos and not missing_images:
            next_tools.append("animate_scene_videos")
        if videos and artifacts["voiceover"] and not artifacts["manifest"]:
            next_tools.append("stitch_final_video")
        if failed_scenes:
            next_tools.append("retry_scene")

    return {
        "project_id": ctx.project_id,
        "project_state": read_project_state(ctx),
        "artifacts": artifacts,
        "scene_ids": scene_ids,
        "completed_scene_count": len(videos),
        "failed_scene_count": len(failed_scenes),
        "missing_images": missing_images,
        "missing_videos": missing_videos,
        "failed_scenes": failed_scenes,
        "next_tools": next_tools,
    }


async def retry_scene_impl(ctx: ProjectContext, scene_id: str, stage: str = "video") -> dict[str, Any]:
    return await retry_scene_with_models_impl(ctx, scene_id, stage=stage)


async def retry_scene_with_models_impl(
    ctx: ProjectContext,
    scene_id: str,
    stage: str = "video",
    *,
    image_model: MagicImageModel | None = None,
    image_resolution: MagicImageResolution | None = None,
    image_style_tool: MagicImageStyleTool | None = None,
    video_model: MagicVideoModel | None = None,
    video_resolution: Resolution | None = None,
    video_audio: bool | None = None,
) -> dict[str, Any]:
    if stage not in {"image", "video", "all"}:
        raise ValueError("stage must be one of: image, video, all")
    plan = load_video_plan(ctx)
    scene = next((candidate for candidate in plan.scenes if candidate.id == scene_id), None)
    if scene is None:
        raise ValueError(f"Unknown scene id: {scene_id}")
    image_ctx = context_with_magic_image_settings(
        ctx,
        model=image_model or ctx.image_model,
        image_resolution=image_resolution or ctx.image_resolution,
        image_style_tool=image_style_tool or ctx.image_style_tool,
    )
    video_ctx = context_with_magic_video_settings(
        ctx,
        model=video_model or ctx.video_model,
        resolution=video_resolution or ctx.resolution,
        audio=ctx.video_audio if video_audio is None else video_audio,
        scenes=[scene],
    )

    images = read_json_artifact(ctx, "images", [])
    videos = read_json_artifact(ctx, "videos", [])
    failures = read_json_artifact(ctx, "failed_scenes", [])
    image_by_scene = {image["scene_id"]: image for image in images}
    new_images: list[dict[str, Any]] = []
    new_videos: list[dict[str, Any]] = []

    if stage in {"image", "all"} or scene_id not in image_by_scene:
        image = await generate_image_asset(image_ctx, scene)
        new_images.append(image)
        image_by_scene[scene_id] = image
        failures = clear_scene_failures(failures, {scene_id}, {"image_generation"})

    if stage in {"video", "all"}:
        image = image_by_scene.get(scene_id)
        if not image:
            raise RuntimeError(f"No image asset exists for {scene_id}; retry with stage='all'.")
        video = await generate_video_asset(video_ctx, scene, image)
        new_videos.append(video)
        failures = clear_scene_failures(failures, {scene_id}, {"video_generation"})

    merged_images = ordered_scene_assets(plan, upsert_scene_assets(images, new_images))
    merged_videos = ordered_scene_assets(plan, upsert_scene_assets(videos, new_videos))
    write_json_artifact(ctx, "images", merged_images)
    write_json_artifact(ctx, "videos", merged_videos)
    write_json_artifact(ctx, "failed_scenes", failures)
    update_project_state(
        ctx,
        provider_settings={
            "image_model": image_ctx.image_model,
            "image_resolution": image_ctx.image_resolution,
            "image_style_tool": image_ctx.image_style_tool,
            "video_model": video_ctx.video_model,
            "video_resolution": video_ctx.resolution,
            "video_audio": video_ctx.video_audio,
        },
        images=merged_images,
        videos=merged_videos,
        failures=failures,
        status={"stage": "scene_retried", "progress": 75, "message": f"Retried {scene_id}."},
        decision={
            "tool": "retry_scene",
            "decision": f"Retried {stage} asset(s) for {scene_id}.",
            "scene_id": scene_id,
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "scene_retried",
        "retried_scene_id": scene_id,
        "images": [with_media_url(image) for image in merged_images],
        "videos": [with_media_url(video) for video in merged_videos],
        "failed_scenes": failures,
        "next_tools": ["stitch_final_video", "inspect_render_status"],
    }


async def record_project_decision_impl(
    ctx: ProjectContext,
    decision: str,
    rationale: str = "",
    scene_id: str | None = None,
) -> dict[str, Any]:
    entry = append_project_decision(
        ctx,
        decision=decision,
        rationale=rationale,
        scene_id=scene_id,
        tool="record_project_decision",
    )
    return {
        "project_id": ctx.project_id,
        "stage": "decision_recorded",
        "decision": entry,
        "decision_count": len(read_project_state(ctx)["decisions"]),
    }


async def regenerate_scene_impl(
    ctx: ProjectContext,
    scene_id: str,
    *,
    narration: str | None = None,
    image_prompt: str | None = None,
    video_prompt: str | None = None,
    duration_seconds: int | None = None,
    regenerate_image: bool = True,
    image_model: MagicImageModel | None = None,
    image_resolution: MagicImageResolution | None = None,
    image_style_tool: MagicImageStyleTool | None = None,
    video_model: MagicVideoModel | None = None,
    video_resolution: Resolution | None = None,
    video_audio: bool | None = None,
) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    plan, scene = patch_scene_in_plan(
        plan,
        scene_id,
        narration=narration,
        image_prompt=image_prompt,
        video_prompt=video_prompt,
        duration_seconds=duration_seconds,
    )
    save_video_plan(ctx, plan)
    invalidate_final_artifacts(ctx)

    images = read_json_artifact(ctx, "images", [])
    videos = read_json_artifact(ctx, "videos", [])
    failures = read_json_artifact(ctx, "failed_scenes", [])
    image_by_scene = {image["scene_id"]: image for image in images}
    image_ctx = context_with_magic_image_settings(
        ctx,
        model=image_model or ctx.image_model,
        image_resolution=image_resolution or ctx.image_resolution,
        image_style_tool=image_style_tool or ctx.image_style_tool,
    )
    video_ctx = context_with_magic_video_settings(
        ctx,
        model=video_model or ctx.video_model,
        resolution=video_resolution or ctx.resolution,
        audio=ctx.video_audio if video_audio is None else video_audio,
        scenes=[scene],
    )

    if regenerate_image or scene_id not in image_by_scene:
        image = await generate_image_asset(image_ctx, scene)
    else:
        image = image_by_scene[scene_id]
    video = await generate_video_asset(video_ctx, scene, image)

    merged_images = ordered_scene_assets(plan, upsert_scene_assets(images, [image]))
    merged_videos = ordered_scene_assets(plan, upsert_scene_assets(videos, [video]))
    failures = clear_scene_failures(failures, {scene_id}, {"image_generation", "video_generation"})
    write_json_artifact(ctx, "images", merged_images)
    write_json_artifact(ctx, "videos", merged_videos)
    write_json_artifact(ctx, "failed_scenes", failures)
    update_project_state(
        ctx,
        current_plan=plan.model_dump(mode="json"),
        provider_settings={
            "image_model": image_ctx.image_model,
            "image_resolution": image_ctx.image_resolution,
            "image_style_tool": image_ctx.image_style_tool,
            "video_model": video_ctx.video_model,
            "video_resolution": video_ctx.resolution,
            "video_audio": video_ctx.video_audio,
        },
        images=merged_images,
        videos=merged_videos,
        failures=failures,
        final_video_path=None,
        manifest_path=None,
        status={"stage": "scene_regenerated", "progress": 78, "message": f"Regenerated {scene_id}."},
        decision={
            "tool": "regenerate_scene",
            "decision": f"Regenerated assets for {scene_id}.",
            "scene_id": scene_id,
            "metadata": {
                "regenerated_image": regenerate_image or scene_id not in image_by_scene,
                "patched_fields": [
                    field
                    for field, value in {
                        "narration": narration,
                        "image_prompt": image_prompt,
                        "video_prompt": video_prompt,
                        "duration_seconds": duration_seconds,
                    }.items()
                    if value is not None
                ],
            },
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "scene_regenerated",
        "scene": scene.model_dump(mode="json"),
        "images": [with_media_url(asset) for asset in merged_images],
        "videos": [with_media_url(asset) for asset in merged_videos],
        "failed_scenes": failures,
        "next_tools": ["inspect_render_status", "restitch_video"],
    }


async def revise_narration_impl(
    ctx: ProjectContext,
    narration: str,
    scene_narration_updates: list[SceneNarrationRevision] | None = None,
) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    plan = plan.model_copy(update={"narration": narration})
    plan = revise_scene_narrations(plan, scene_narration_updates or [])
    save_video_plan(ctx, plan)
    invalidate_final_artifacts(ctx, voiceover=True)
    update_project_state(
        ctx,
        current_plan=plan.model_dump(mode="json"),
        status={"stage": "narration_revised", "progress": 35, "message": "Narration revised; voiceover is stale."},
        decision={
            "tool": "revise_narration",
            "decision": "Revised narration and invalidated the previous voiceover.",
            "metadata": {"scene_ids": [revision.scene_id for revision in scene_narration_updates or []]},
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "narration_revised",
        "plan": plan.model_dump(mode="json"),
        "next_tools": ["replace_voiceover", "restitch_video"],
    }


async def replace_voiceover_impl(ctx: ProjectContext, narration: str | None = None) -> dict[str, Any]:
    plan = load_video_plan(ctx)
    if narration is not None:
        plan = plan.model_copy(update={"narration": narration})
        save_video_plan(ctx, plan)
    invalidate_final_artifacts(ctx, voiceover=True)
    voiceover = await generate_voiceover_asset(ctx, plan.narration, plan_duration_seconds(plan))
    write_json_artifact(ctx, "voiceover", voiceover)
    update_project_state(
        ctx,
        current_plan=plan.model_dump(mode="json"),
        voiceover=voiceover,
        final_video_path=None,
        manifest_path=None,
        status={"stage": "voiceover_replaced", "progress": 55, "message": "Voiceover replaced."},
        decision={
            "tool": "replace_voiceover",
            "decision": "Replaced the voiceover audio from the current narration.",
            "metadata": {"target_duration_seconds": voiceover.get("target_duration_seconds")},
        },
    )
    return {
        "project_id": ctx.project_id,
        "stage": "voiceover_replaced",
        "voiceover": with_media_url(voiceover),
        "next_tools": ["restitch_video"],
    }


async def restitch_video_impl(
    ctx: ProjectContext,
    token_output: dict[str, Any] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    manifest = await stitch_final_video_impl(ctx, token_output)
    update_project_state(
        ctx,
        status={"stage": "video_restitched", "progress": 95, "message": "Final video restitched."},
        decision={
            "tool": "restitch_video",
            "decision": "Restitched the final video from current scene videos and voiceover.",
            **({"rationale": reason} if reason else {}),
        },
    )
    return manifest


async def render_plan(
    plan: VideoPlan,
    ctx: ProjectContext,
    token_output: dict[str, Any],
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    await draft_video_plan_impl(ctx, plan.title, plan.narration, plan.scenes, plan.visual_bible, normalize_scene_ids=False)
    voice_task = asyncio.create_task(generate_voiceover_impl(ctx))
    try:
        if on_progress:
            await on_progress("voiceover_images", 30, "Generating the voiceover and scene images.")
        await generate_scene_images_impl(ctx)
        if on_progress:
            await on_progress("video_generation", 65, "Animating scene videos.")
        await animate_scene_videos_impl(ctx)
        await voice_task
        if on_progress:
            await on_progress("stitching", 90, "Stitching the final edit.")
        return await stitch_final_video_impl(ctx, token_output)
    except Exception:
        if not voice_task.done():
            voice_task.cancel()
            with suppress(asyncio.CancelledError):
                await voice_task
        raise


@function_tool(defer_loading=True)
async def draft_video_plan(
    ctx: RunContextWrapper[ProjectContext],
    title: str,
    narration: str,
    scenes: list[Scene],
    visual_bible: str = "",
) -> dict[str, Any]:
    """
    Persist the complete creative plan before making provider calls.

    Args:
        title: Concise title for the finished video.
        narration: Full voiceover script for the complete edit.
        visual_bible: Compact continuity notes for subject, palette, lens language, and environment.
        scenes: Ordered scene plan with narration, image prompts, motion prompts, and durations.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="planning",
        progress=15,
        message="Creative plan drafted by the agent.",
    )
    return await draft_video_plan_impl(ctx.context, title, narration, scenes, visual_bible)


@function_tool(defer_loading=True)
async def generate_voiceover(ctx: RunContextWrapper[ProjectContext]) -> dict[str, Any]:
    """Generate the voiceover audio for the current saved plan."""
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="voiceover",
        progress=30,
        message="Generating the voiceover.",
    )
    return await generate_voiceover_impl(ctx.context)


@function_tool(defer_loading=True)
async def generate_scene_images(
    ctx: RunContextWrapper[ProjectContext],
    model: MagicImageModel = DEFAULT_MAGIC_HOUR_IMAGE_MODEL,
    image_resolution: MagicImageResolution = "1k",
    image_style_tool: MagicImageStyleTool = "general",
    scene_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate still images for all scenes, or for selected scene ids.

    Args:
        model: Magic Hour image model. Default to seedream-v4 unless the user explicitly selected a different model or the prompt clearly needs a model-specific capability. Do not use Magic Hour's default model unless the user explicitly asks for it.
        image_resolution: Magic Hour image resolution. Choose a value supported by the selected image model: 640px, 1k, 2k, or 4k.
        image_style_tool: Magic Hour image style category. Use general unless a specific image domain such as ai-photo-generator, ai-character-generator, ai-landscape-generator, or movie-poster-generator clearly fits.
        scene_ids: Optional scene ids to generate. Omit to generate every scene in the saved plan.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="image_generation",
        progress=45,
        message=f"Generating scene images with {model}.",
    )
    return await generate_scene_images_impl(
        ctx.context,
        scene_ids,
        model=model,
        image_resolution=image_resolution,
        image_style_tool=image_style_tool,
    )


@function_tool(defer_loading=True)
async def animate_scene_videos(
    ctx: RunContextWrapper[ProjectContext],
    model: MagicVideoModel = DEFAULT_MAGIC_HOUR_VIDEO_MODEL,
    resolution: Resolution = "720p",
    audio: bool = False,
    scene_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Animate scene videos from generated images.

    Args:
        model: Magic Hour image-to-video model. Default to ltx-2.3 unless the user explicitly selected a different model or the prompt clearly needs a model-specific capability. Use seedance-2.0 for consistency, kling-2.5 for motion/camera control, kling-3.0 for cinematic storytelling, veo3.1 for realism/prompt adherence, or sora-2 for story-first creative motion only when that tradeoff is intentional.
        resolution: Output video resolution supported by the selected video model.
        audio: Whether Magic Hour should generate provider audio. Usually false because the final edit uses Fish Audio voiceover.
        scene_ids: Optional scene ids to animate. Omit to animate every scene with an image.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="video_generation",
        progress=70,
        message=f"Animating scene videos with {model}.",
    )
    return await animate_scene_videos_impl(
        ctx.context,
        scene_ids,
        model=model,
        resolution=resolution,
        audio=audio,
    )


@function_tool(defer_loading=True)
async def stitch_final_video(ctx: RunContextWrapper[ProjectContext]) -> dict[str, Any]:
    """Stitch completed scene videos with the voiceover into the final MP4."""
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="stitching",
        progress=90,
        message="Stitching the final edit.",
    )
    return await stitch_final_video_impl(
        ctx.context,
        pending_token_output(ctx.context, ENV.get("OPENAI_MODEL", "gpt-5.5")),
    )


@function_tool(defer_loading=True)
async def inspect_render_status(ctx: RunContextWrapper[ProjectContext]) -> dict[str, Any]:
    """Inspect saved plan, project_state.json, media artifacts, failures, and recommended next tools."""
    return await inspect_render_status_impl(ctx.context)


@function_tool(defer_loading=True)
async def record_project_decision(
    ctx: RunContextWrapper[ProjectContext],
    decision: str,
    rationale: str = "",
    scene_id: str | None = None,
) -> dict[str, Any]:
    """
    Persist an important creative, retry, or user-preference decision.

    Args:
        decision: Short statement of the choice being made.
        rationale: Optional reason for the choice.
        scene_id: Optional scene id when the decision is scene-specific.
    """
    return await record_project_decision_impl(ctx.context, decision, rationale, scene_id)


@function_tool(defer_loading=True)
async def regenerate_scene(
    ctx: RunContextWrapper[ProjectContext],
    scene_id: str,
    narration: str | None = None,
    image_prompt: str | None = Field(default=None, description="Optional replacement still-image prompt. Write a stable keyframe for image-to-video: concrete visible subject, action pose, foreground/background, lighting, lens/framing, palette, and continuity. Do not include text/logos/UI or anything that must be invented later."),
    video_prompt: str | None = Field(default=None, description="Optional replacement image-to-video motion prompt. Use one camera move and at most one subject motion; only animate what already exists in the still image. No cuts, new objects, scene changes, transformations, or ungrounded events."),
    duration_seconds: int | None = None,
    regenerate_image: bool = True,
    image_model: MagicImageModel | None = None,
    image_resolution: MagicImageResolution | None = None,
    image_style_tool: MagicImageStyleTool | None = None,
    video_model: MagicVideoModel | None = None,
    video_resolution: Resolution | None = None,
    video_audio: bool | None = None,
) -> dict[str, Any]:
    """
    Patch one scene and regenerate only that scene's media assets.

    Args:
        scene_id: Saved scene id, such as scene_2.
        narration: Optional replacement narration for this scene.
        image_prompt: Optional replacement still-image prompt. Write a stable keyframe for image-to-video with visible subject, action pose, foreground/background, lighting, lens/framing, palette, and continuity. Avoid text/logos/UI and anything the video model must invent.
        video_prompt: Optional replacement motion prompt. Use one camera move and at most one subject motion; only animate what already exists in the still image. No cuts, new objects, scene changes, transformations, or ungrounded events.
        duration_seconds: Optional replacement scene duration.
        regenerate_image: Whether to regenerate the image before animating the scene.
        image_model: Optional Magic Hour image model for the regenerated keyframe.
        image_resolution: Optional Magic Hour image resolution for the regenerated keyframe.
        image_style_tool: Optional Magic Hour image style tool.
        video_model: Optional Magic Hour image-to-video model for the regenerated scene.
        video_resolution: Optional Magic Hour video resolution for the regenerated scene.
        video_audio: Optional provider-audio toggle; usually false because final stitching uses Fish Audio.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="regenerate_scene",
        progress=78,
        message=f"Regenerating {scene_id}.",
    )
    return await regenerate_scene_impl(
        ctx.context,
        scene_id,
        narration=narration,
        image_prompt=image_prompt,
        video_prompt=video_prompt,
        duration_seconds=duration_seconds,
        regenerate_image=regenerate_image,
        image_model=image_model,
        image_resolution=image_resolution,
        image_style_tool=image_style_tool,
        video_model=video_model,
        video_resolution=video_resolution,
        video_audio=video_audio,
    )


@function_tool(defer_loading=True)
async def revise_narration(
    ctx: RunContextWrapper[ProjectContext],
    narration: str,
    scene_narration_updates: list[SceneNarrationRevision] | None = None,
) -> dict[str, Any]:
    """
    Patch the saved narration and invalidate stale voiceover/final video artifacts.

    Args:
        narration: Replacement full voiceover narration.
        scene_narration_updates: Optional per-scene narration replacements.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="revise_narration",
        progress=35,
        message="Revising narration.",
    )
    return await revise_narration_impl(ctx.context, narration, scene_narration_updates)


@function_tool(defer_loading=True)
async def replace_voiceover(
    ctx: RunContextWrapper[ProjectContext],
    narration: str | None = None,
) -> dict[str, Any]:
    """
    Replace the voiceover audio from the current saved narration or a new narration.

    Args:
        narration: Optional full narration to save before generating audio.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="replace_voiceover",
        progress=55,
        message="Replacing voiceover.",
    )
    return await replace_voiceover_impl(ctx.context, narration)


@function_tool(defer_loading=True)
async def restitch_video(
    ctx: RunContextWrapper[ProjectContext],
    reason: str = "",
) -> dict[str, Any]:
    """
    Rebuild the final MP4 from the current scene videos and voiceover.

    Args:
        reason: Optional reason for restitching after a revision.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="restitching",
        progress=95,
        message="Restitching the final edit.",
    )
    return await restitch_video_impl(
        ctx.context,
        pending_token_output(ctx.context, ENV.get("OPENAI_MODEL", "gpt-5.5")),
        reason,
    )


@function_tool(defer_loading=True)
async def retry_scene(
    ctx: RunContextWrapper[ProjectContext],
    scene_id: str,
    stage: Literal["image", "video", "all"] = "video",
    image_model: MagicImageModel | None = None,
    image_resolution: MagicImageResolution | None = None,
    image_style_tool: MagicImageStyleTool | None = None,
    video_model: MagicVideoModel | None = None,
    video_resolution: Resolution | None = None,
    video_audio: bool | None = None,
) -> dict[str, Any]:
    """
    Retry one scene without restarting the whole project.

    Args:
        scene_id: Saved scene id, such as scene_1.
        stage: Retry image, video, or all scene assets.
        image_model: Optional Magic Hour image model when retrying image/all.
        image_resolution: Optional Magic Hour image resolution when retrying image/all.
        image_style_tool: Optional Magic Hour image style tool when retrying image/all.
        video_model: Optional Magic Hour image-to-video model when retrying video/all.
        video_resolution: Optional Magic Hour video resolution when retrying video/all.
        video_audio: Optional provider-audio toggle; usually false because final stitching uses Fish Audio.
    """
    await update_project_status(
        ctx.context.project_id,
        status="running",
        stage="retry_scene",
        progress=75,
        message=f"Retrying {scene_id}.",
    )
    return await retry_scene_with_models_impl(
        ctx.context,
        scene_id,
        stage,
        image_model=image_model,
        image_resolution=image_resolution,
        image_style_tool=image_style_tool,
        video_model=video_model,
        video_resolution=video_resolution,
        video_audio=video_audio,
    )


VIDEO_STUDIO_TOOLS = tool_namespace(
    name="video_studio",
    description="Professional cinematic video generation and post-production tools.",
    tools=[
        draft_video_plan,
        generate_voiceover,
        generate_scene_images,
        animate_scene_videos,
        stitch_final_video,
        inspect_render_status,
        record_project_decision,
        regenerate_scene,
        revise_narration,
        replace_voiceover,
        restitch_video,
        retry_scene,
    ],
)


# Legacy direct planner retained for focused plan/token tests; run_project uses video_agent.
planning_agent = Agent(
    name="Fast Video Planning Agent",
    model=ENV.get("OPENAI_MODEL", "gpt-5.5"),
    instructions=PLANNING_INSTRUCTIONS,
    tools=[],
    output_type=VideoPlan,
    model_settings=ModelSettings(
        reasoning={"effort": ENV.get("OPENAI_REASONING_EFFORT", "low")},
        verbosity=ENV.get("OPENAI_VERBOSITY", "low"),
        parallel_tool_calls=False,
    ),
)

# The production path: the agent owns planning, provider-tool sequencing, retries, and stitching.
video_agent = Agent(
    name="Autonomous Video Art Director",
    model=ENV.get("OPENAI_MODEL", "gpt-5.4"),
    instructions=INSTRUCTIONS,
    tools=[*VIDEO_STUDIO_TOOLS, ToolSearchTool()],
    model_settings=ModelSettings(
        reasoning={"effort": ENV.get("OPENAI_REASONING_EFFORT", "low")},
        verbosity=ENV.get("OPENAI_VERBOSITY", "low"),
        parallel_tool_calls=True,
    ),
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    missing_config = missing_configuration()
    missing_dependencies = missing_system_dependencies()
    return {
        "status": "ok" if not missing_config and not missing_dependencies else "missing_config",
        "output_dir": str(OUTPUT_DIR),
        "missing_config": missing_config,
        "missing_dependencies": missing_dependencies,
        "active_projects": sum(1 for project in PROJECTS.values() if project["status"] in {"queued", "running"}),
    }


async def run_project(project_id: str, request: CreateProjectRequest) -> None:
    ctx = context(project_id, request)
    ensure_project_state(ctx, request)

    try:
        await update_project_status(
            project_id,
            status="running",
            stage="planning",
            progress=10,
            message="Planning the timed script and scene continuity.",
        )

        result = await Runner.run(
            video_agent,
            input=build_generation_brief(request, ctx),
            context=ctx,
            max_turns=configured_agent_max_turns(),
        )
        token_output = write_token_output(ctx, result.context_wrapper.usage, model=video_agent.model)
        manifest = merge_token_output_into_manifest(ctx, token_output)
        failed_count = int(manifest.get("failed_scene_count") or 0)
        await update_project_status(
            project_id,
            status="succeeded",
            stage="complete",
            progress=100,
            message="Video is ready." if failed_count == 0 else f"Partial video is ready with {failed_count} failed scene(s).",
            manifest=manifest,
        )
    except Exception as exc:
        logger.exception("Project generation failed")
        current = PROJECTS.get(project_id, {})
        await update_project_status(
            project_id,
            status="failed",
            stage="failed",
            progress=int(current.get("progress", 0)),
            message="Generation failed.",
            error=str(exc),
        )


async def run_project_message(project_id: str, message: str) -> None:
    ctx = context_for_existing_project(project_id)
    previous_status = read_project_status(project_id) or {}
    previous_manifest = previous_status.get("manifest")

    try:
        await update_project_status(
            project_id,
            status="running",
            stage="message_running",
            progress=25,
            message="Agent is handling the project message.",
            manifest=previous_manifest,
        )
        result = await Runner.run(
            video_agent,
            input=build_project_message_brief(project_id, message, ctx),
            context=ctx,
            max_turns=configured_agent_max_turns(),
        )
        token_output = write_token_output(ctx, result.context_wrapper.usage, model=video_agent.model)
        manifest = merge_token_output_into_manifest(ctx, token_output) if artifact_path(ctx, "manifest").exists() else None
        response_text = agent_response_content(result.final_output)
        append_project_message(
            ctx,
            role="assistant",
            content=response_text,
            metadata={
                "model": video_agent.model,
                "token_output_path": token_output["token_output_path"],
            },
        )
        await update_project_status(
            project_id,
            status="succeeded",
            stage="message_complete",
            progress=100,
            message=response_text[:240],
            manifest=manifest,
        )
    except Exception as exc:
        logger.exception("Project message handling failed")
        append_project_message(
            ctx,
            role="assistant",
            content=f"Agent turn failed: {exc}",
            metadata={"error": str(exc)},
        )
        current = PROJECTS.get(project_id, {})
        await update_project_status(
            project_id,
            status="failed",
            stage="message_failed",
            progress=int(current.get("progress", 0)),
            message="Project message failed.",
            error=str(exc),
            manifest=previous_manifest,
        )


@app.post("/api/projects", status_code=202)
async def create_project(request: CreateProjectRequest) -> dict[str, Any]:
    assert_runtime_ready()
    project_id = uuid.uuid4().hex
    initialize_project_state(context(project_id, request), request)
    payload = await update_project_status(
        project_id,
        status="queued",
        stage="queued",
        progress=0,
        message="Project queued locally.",
    )
    asyncio.create_task(run_project(project_id, request))
    return payload


@app.post("/api/projects/{project_id}/messages", status_code=202)
async def create_project_message(project_id: str, request: ProjectMessageRequest) -> dict[str, Any]:
    assert_runtime_ready()
    try:
        existing_status = read_project_status(project_id)
        ctx = context_for_existing_project(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if existing_status is None:
        raise HTTPException(status_code=404, detail="Project not found")

    append_project_message(ctx, role="user", content=request.message)
    payload = await update_project_status(
        project_id,
        status="queued",
        stage="message_queued",
        progress=int(existing_status.get("progress", 0)),
        message="Project message queued for the agent.",
        manifest=existing_status.get("manifest"),
    )
    payload["project_state"] = read_project_state(ctx)
    asyncio.create_task(run_project_message(project_id, request.message))
    return payload


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    try:
        status = read_project_status(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc
    if status is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return status
