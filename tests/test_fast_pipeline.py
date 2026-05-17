import asyncio
import json
import threading
from pathlib import Path

import pytest
from agents import ToolSearchTool
from agents.usage import Usage
from fastapi.testclient import TestClient
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from backend.app import main
from backend.app.tools import media


def test_main_agent_uses_split_namespaced_video_studio_tools() -> None:
    tool_names = {getattr(tool, "name", None) for tool in main.video_agent.tools}
    expected_tools = {
        "draft_video_plan",
        "generate_voiceover",
        "generate_scene_images",
        "animate_scene_videos",
        "stitch_final_video",
        "inspect_render_status",
        "record_project_decision",
        "retry_scene",
        "regenerate_scene",
        "revise_narration",
        "replace_voiceover",
        "restitch_video",
    }

    assert expected_tools.issubset(tool_names)
    assert "execute_video_batch" not in tool_names
    assert any(isinstance(tool, ToolSearchTool) for tool in main.video_agent.tools)
    for tool in main.video_agent.tools:
        if getattr(tool, "name", None) in expected_tools:
            assert getattr(tool, "defer_loading") is True
            assert getattr(tool, "_tool_namespace") == "video_studio"
    assert main.video_agent.model_settings.reasoning.effort == "low"


def test_planner_instructions_prioritize_fast_good_i2v_prompts() -> None:
    for phrase in {
        "draft_video_plan",
        "generate_voiceover",
        "generate_scene_images",
        "animate_scene_videos",
        "stitch_final_video",
        "inspect_render_status",
        "retry_scene",
        "record_project_decision",
        "regenerate_scene",
        "revise_narration",
        "replace_voiceover",
        "restitch_video",
        "art director",
        "Use 3-5 scenes",
        "Image prompts should be concrete",
        "Video prompts should describe camera motion",
        "prefer 4-6 second scenes",
    }:
        assert phrase in main.INSTRUCTIONS


def test_planning_instructions_treat_narration_as_spoken_story_not_visual_prompt() -> None:
    for phrase in {
        "Narration is spoken voiceover copy",
        "Do not write narration as image prompt",
        "not camera direction",
        "not a production note",
    }:
        assert phrase in main.PLANNING_INSTRUCTIONS
        assert phrase in main.build_generation_brief(main.CreateProjectRequest(prompt="make a video"), main.context("a" * 32, main.CreateProjectRequest(prompt="make a video")))


def test_magic_hour_requirement_supports_ltx_23_sdk_validator() -> None:
    requirement = Path("requirements.txt").read_text(encoding="utf-8")

    assert "magic-hour>=0.63.0" in requirement


def test_installed_magic_hour_sdk_accepts_ltx_23_model_literal() -> None:
    import inspect

    from magic_hour.resources.v1.image_to_video.client import ImageToVideoClient

    model_annotation = str(inspect.signature(ImageToVideoClient.create).parameters["model"].annotation)

    assert "ltx-2.3" in model_annotation


def test_video_poll_interval_defaults_to_lower_noise_provider_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAGIC_HOUR_POLL_INTERVAL", raising=False)

    assert media.video_poll_interval_seconds() == 2.0


def test_context_defaults_video_model_to_ltx_23_when_env_is_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.delitem(main.ENV, "MAGIC_HOUR_VIDEO_MODEL", raising=False)

    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("a" * 32, request)

    assert ctx.video_model == "ltx-2.3"


def test_context_coerces_magic_hour_default_video_model_to_ltx_23(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(main.ENV, "MAGIC_HOUR_VIDEO_MODEL", "default")

    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("a" * 32, request)

    assert ctx.video_model == "ltx-2.3"


def test_context_defaults_image_model_to_seedream_when_env_is_unset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.delitem(main.ENV, "MAGIC_HOUR_IMAGE_MODEL", raising=False)

    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("a" * 32, request)

    assert ctx.image_model == "seedream-v4"


def test_context_coerces_magic_hour_default_image_model_to_seedream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setitem(main.ENV, "MAGIC_HOUR_IMAGE_MODEL", "default")

    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("a" * 32, request)

    assert ctx.image_model == "seedream-v4"


def test_project_context_defaults_use_explicit_magic_hour_models(tmp_path: Path) -> None:
    ctx = main.ProjectContext(project_id="defaults", project_dir=tmp_path / "defaults", aspect_ratio="9:16", resolution="720p")

    assert ctx.image_model == "seedream-v4"
    assert ctx.video_model == "ltx-2.3"


def test_generation_brief_tells_agent_to_default_to_ltx_23_for_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.delitem(main.ENV, "MAGIC_HOUR_VIDEO_MODEL", raising=False)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)
    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("b" * 32, request)

    brief = main.build_generation_brief(request, ctx)

    assert "Default image-to-video model: ltx-2.3" in brief
    assert "User-selected image-to-video model: agent chooses" not in brief


def test_generation_brief_tells_agent_to_default_to_seedream_for_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.delitem(main.ENV, "MAGIC_HOUR_IMAGE_MODEL", raising=False)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)
    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context("b" * 32, request)

    brief = main.build_generation_brief(request, ctx)

    assert "Default image model: seedream-v4" in brief
    assert "User-selected image model: agent chooses" not in brief


def test_project_message_brief_tells_agent_to_default_to_ltx_23_for_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.delitem(main.ENV, "MAGIC_HOUR_VIDEO_MODEL", raising=False)
    project_id = "b" * 32
    request = main.CreateProjectRequest(prompt="make a video")
    ctx = main.context(project_id, request)
    main.initialize_project_state(ctx, request)
    main.write_json_artifact(ctx, "status", {"project_id": project_id, "status": "succeeded"})

    brief = main.build_project_message_brief(project_id, "make a new video", ctx)

    assert "Default image-to-video model: ltx-2.3" in brief
    assert "Default image model: seedream-v4" in brief


def test_scene_schema_describes_keyframe_and_i2v_prompt_contracts() -> None:
    schema = main.draft_video_plan.params_json_schema
    scene_properties = schema["$defs"]["Scene"]["properties"]
    image_description = scene_properties["image_prompt"]["description"]
    video_description = scene_properties["video_prompt"]["description"]

    assert "stable cinematic keyframe" in image_description
    assert "only what is visible" in image_description
    assert "one camera move" in video_description
    assert "only animate what already exists" in video_description
    assert "no cuts" in video_description


def test_agent_media_tools_expose_model_selection_contract() -> None:
    tool_by_name = {getattr(tool, "name", None): tool for tool in main.video_agent.tools}
    image_schema = tool_by_name["generate_scene_images"].params_json_schema["properties"]
    video_schema = tool_by_name["animate_scene_videos"].params_json_schema["properties"]
    regenerate_schema = tool_by_name["regenerate_scene"].params_json_schema["properties"]

    assert image_schema["model"]["enum"] == list(main.MAGIC_IMAGE_MODELS)
    assert image_schema["model"]["default"] == "seedream-v4"
    assert "Default to seedream-v4" in image_schema["model"]["description"]
    assert image_schema["image_resolution"]["enum"] == list(main.MAGIC_IMAGE_RESOLUTIONS)
    assert video_schema["model"]["enum"] == list(main.MAGIC_VIDEO_MODELS)
    assert video_schema["model"]["default"] == "ltx-2.3"
    assert "Default to ltx-2.3" in video_schema["model"]["description"]
    assert video_schema["resolution"]["enum"] == ["480p", "720p", "1080p"]
    assert "stable keyframe" in regenerate_schema["image_prompt"]["description"]
    assert "only animate what already exists" in regenerate_schema["video_prompt"]["description"]
    assert regenerate_schema["image_model"]["anyOf"][0]["enum"] == list(main.MAGIC_IMAGE_MODELS)
    assert regenerate_schema["video_model"]["anyOf"][0]["enum"] == list(main.MAGIC_VIDEO_MODELS)


def test_generation_brief_uses_tts_budget_and_crossfade_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "old-render"
    voiceover_path = project_dir / "voiceover" / "voiceover.mp3"
    voiceover_path.parent.mkdir(parents=True)
    voiceover_path.write_bytes(b"voice")
    words = " ".join(f"word{i}" for i in range(30))
    (project_dir / "manifest.json").write_text(
        json.dumps(
            {
                "audio_model": "s2-pro",
                "voiceover": {"path": str(voiceover_path)},
                "plan": {"narration": words},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)

    request = main.CreateProjectRequest(
        prompt="rainy alley story",
        duration_seconds=20,
        scene_count=4,
        aspect_ratio="9:16",
        resolution="720p",
    )
    ctx = main.ProjectContext(
        project_id="project",
        project_dir=tmp_path / "project",
        aspect_ratio="9:16",
        resolution="720p",
        audio_model="s2-pro",
    )

    brief = main.build_generation_brief(request, ctx)

    assert "Estimated Fish Audio pace: 3.00 words/second" in brief
    assert "Narration budget: 54-60 spoken words" in brief
    assert "Scene duration total: 21.5 seconds" in brief
    assert "Fish Audio S2 expression cues" in brief
    assert "[whispers softly]" in brief
    assert "(softly)" not in brief
    assert "Magic Hour image models" in brief
    assert "seedream-v4: detailed cinematic keyframes" in brief
    assert "Magic Hour image-to-video models" in brief
    assert "kling-3.0: cinematic multi-shot storytelling" in brief
    assert "Supported I2V durations" in brief


def test_generation_brief_lets_prompt_constraints_override_ui_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)
    ctx = main.ProjectContext(project_id="project", project_dir=tmp_path / "project", aspect_ratio="9:16", resolution="720p")

    wifi = main.CreateProjectRequest(
        prompt='Create a 15-second informational video explaining "How WiFi moves through walls." You cannot show a router or a computer. Use 3 scenes and a visual metaphor involving "water" or "light" to explain signal diffraction and absorption.',
        duration_seconds=20,
        scene_count=4,
    )
    wifi_brief = main.build_generation_brief(wifi, ctx)

    assert "Prompt duration constraint: 15 seconds" in wifi_brief
    assert "Scene count constraint: exactly 3 scenes" in wifi_brief
    assert "Requested scene count: 4" not in wifi_brief
    assert "Target final runtime: 15 seconds" in wifi_brief

    money = main.CreateProjectRequest(
        prompt='Generate a 15-second visual history of "The Evolution of Money," starting from "Bartering Cattle" and ending with "Digital Code." You must include exactly 4 scenes, and each scene must have a unique lighting style representing its era.',
    )
    money_brief = main.build_generation_brief(money, ctx)

    assert "Scene count constraint: exactly 4 scenes" in money_brief
    assert "Prompt duration constraint: 15 seconds" in money_brief


def test_generation_brief_handles_minimum_and_agent_decided_scene_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)
    ctx = main.ProjectContext(project_id="project", project_dir=tmp_path / "project", aspect_ratio="9:16", resolution="720p")

    solar = main.CreateProjectRequest(
        prompt="Create a 15-second informational video explaining how a solar panel turns sunlight into electricity. The video must show at least three distinct stages of the process to be educational.",
        duration_seconds=20,
        scene_count=4,
    )
    solar_brief = main.build_generation_brief(solar, ctx)

    assert "Scene count constraint: at least 3 scenes or stages" in solar_brief
    assert "Prompt duration constraint: 15 seconds" in solar_brief

    purchasing_power = main.CreateProjectRequest(
        prompt="Explain why $100 bought a grocery cart full of food in 1970 but only a few items today. You decide the number of scenes and the visual style. The goal is to make the viewer feel the loss of purchasing power in 15 seconds.",
        duration_seconds=20,
        scene_count=4,
    )
    purchasing_brief = main.build_generation_brief(purchasing_power, ctx)

    assert "Scene count constraint: agent decides" in purchasing_brief
    assert "Requested scene count: 4" not in purchasing_brief
    assert "Prompt duration constraint: 15 seconds" in purchasing_brief


def test_generation_brief_handles_under_duration_and_director_scene_phrases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "probe_media_duration", lambda path: 10.0)
    ctx = main.ProjectContext(project_id="project", project_dir=tmp_path / "project", aspect_ratio="9:16", resolution="720p")

    quantum = main.CreateProjectRequest(
        prompt="Explain 'Quantum Entanglement' using a visual metaphor. You are the director: decide the number of scenes and the sequence of events to show that when one particle spins, the other reacts instantly, no matter the distance. Keep it under 15 seconds.",
        duration_seconds=20,
        scene_count=4,
    )
    quantum_brief = main.build_generation_brief(quantum, ctx)

    assert "Prompt duration constraint: under 15 seconds" in quantum_brief
    assert "Scene count constraint: agent decides" in quantum_brief
    assert "Requested scene count: 4" not in quantum_brief

    mitosis = main.CreateProjectRequest(
        prompt="Create an informational short on 'How a Cell Divides.' You must decide how many scenes are needed to show the most critical steps of mitosis. Ensure the transition between scenes feels like one continuous biological event.",
        duration_seconds=20,
        scene_count=4,
    )
    mitosis_brief = main.build_generation_brief(mitosis, ctx)

    assert "Scene count constraint: agent decides" in mitosis_brief
    assert "Requested scene count: 4" not in mitosis_brief


@pytest.mark.asyncio
async def test_image_generation_tool_uses_agent_selected_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="models", project_dir=tmp_path / "models", aspect_ratio="9:16", resolution="720p")
    await main.draft_video_plan_impl(
        ctx,
        "Model Choice",
        "Narration.",
        [main.Scene(id="scene_1", narration="one", image_prompt="image", video_prompt="motion", duration_seconds=5)],
    )
    seen: dict[str, str] = {}

    async def fake_image(ctx_arg: main.ProjectContext, scene: main.Scene) -> dict:
        seen["image_model"] = ctx_arg.image_model
        seen["image_resolution"] = ctx_arg.image_resolution
        seen["image_style_tool"] = ctx_arg.image_style_tool
        path = ctx_arg.project_dir / "image.jpg"
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": ctx_arg.image_model}

    monkeypatch.setattr(main, "generate_image_asset", fake_image)

    payload = await main.generate_scene_images_impl(
        ctx,
        model="z-image-turbo",
        image_resolution="640px",
        image_style_tool="ai-photo-generator",
    )
    state = main.read_project_state(ctx)

    assert seen == {
        "image_model": "z-image-turbo",
        "image_resolution": "640px",
        "image_style_tool": "ai-photo-generator",
    }
    assert payload["images"][0]["model"] == "z-image-turbo"
    assert state["provider_settings"]["image_model"] == "z-image-turbo"
    assert state["provider_settings"]["image_resolution"] == "640px"
    assert state["provider_settings"]["image_style_tool"] == "ai-photo-generator"


@pytest.mark.asyncio
async def test_image_generation_tool_sends_visual_bible_with_each_scene_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="continuity", project_dir=tmp_path / "continuity", aspect_ratio="9:16", resolution="720p")
    await main.draft_video_plan_impl(
        ctx,
        "Continuity",
        "Narration.",
        [
            main.Scene(
                id="scene_1",
                narration="one",
                image_prompt="Same woman tightens a motorcycle bolt at sunset.",
                video_prompt="motion",
                duration_seconds=5,
            )
        ],
        visual_bible="Same late-40s woman mechanic: tan skin, silver-streaked tied-back curls, rectangular glasses, green work shirt, red bandana.",
    )
    seen: dict[str, str] = {}

    async def fake_image(ctx_arg: main.ProjectContext, scene: main.Scene) -> dict:
        seen["prompt"] = scene.image_prompt
        path = ctx_arg.project_dir / "image.jpg"
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": ctx_arg.image_model}

    monkeypatch.setattr(main, "generate_image_asset", fake_image)

    payload = await main.generate_scene_images_impl(ctx)

    assert "Continuity bible for every scene:" in seen["prompt"]
    assert "silver-streaked tied-back curls" in seen["prompt"]
    assert "Same woman tightens a motorcycle bolt" in seen["prompt"]
    assert payload["images"][0]["prompt"] == seen["prompt"]


@pytest.mark.asyncio
async def test_video_generation_tool_uses_agent_selected_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="video-models", project_dir=tmp_path / "video-models", aspect_ratio="9:16", resolution="720p")
    await main.draft_video_plan_impl(
        ctx,
        "Video Model Choice",
        "Narration.",
        [main.Scene(id="scene_1", narration="one", image_prompt="image", video_prompt="motion", duration_seconds=5)],
    )
    main.write_json_artifact(ctx, "images", [{"scene_id": "scene_1", "path": str(ctx.project_dir / "image.jpg")}])
    seen: dict[str, object] = {}

    async def fake_video_batch(ctx_arg: main.ProjectContext, pairs: list[tuple[main.Scene, dict]]) -> list[dict]:
        seen["video_model"] = ctx_arg.video_model
        seen["resolution"] = ctx_arg.resolution
        seen["video_audio"] = ctx_arg.video_audio
        scene = pairs[0][0]
        path = ctx_arg.project_dir / "video.mp4"
        path.write_bytes(b"video")
        return [{"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": ctx_arg.video_model, "duration_seconds": scene.duration_seconds}]

    async def fake_stitch(ctx_arg: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        path = ctx_arg.project_dir / "final.mp4"
        path.write_bytes(b"final")
        return str(path)

    monkeypatch.setattr(main, "generate_video_assets_batch", fake_video_batch)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)

    payload = await main.animate_scene_videos_impl(
        ctx,
        model="kling-3.0",
        resolution="1080p",
        audio=True,
    )
    main.write_json_artifact(ctx, "voiceover", {"path": str(ctx.project_dir / "voice.mp3"), "model": "voice-model", "duration_seconds": 5})
    manifest = await main.stitch_final_video_impl(ctx, main.pending_token_output(ctx, "gpt-5.5"))
    state = main.read_project_state(ctx)

    assert seen == {"video_model": "kling-3.0", "resolution": "1080p", "video_audio": True}
    assert payload["videos"][0]["model"] == "kling-3.0"
    assert state["provider_settings"]["video_model"] == "kling-3.0"
    assert state["provider_settings"]["video_resolution"] == "1080p"
    assert state["provider_settings"]["video_audio"] is True
    assert manifest["video_model"] == "kling-3.0"
    assert manifest["video_resolution"] == "1080p"
    assert manifest["video_audio"] is True


@pytest.mark.asyncio
async def test_draft_video_plan_clears_stale_render_outputs(tmp_path: Path) -> None:
    ctx = main.ProjectContext(project_id="stale", project_dir=tmp_path / "stale", aspect_ratio="9:16", resolution="720p")
    main.initialize_project_state(ctx, main.CreateProjectRequest(prompt="old prompt"))
    for artifact in ("voiceover", "images", "videos", "manifest"):
        main.write_json_artifact(ctx, artifact, {"old": True})
    stale_files = [
        ctx.project_dir / "voiceover" / "old.mp3",
        ctx.project_dir / "images" / "scene_1" / "old.png",
        ctx.project_dir / "videos" / "scene_1" / "old.mp4",
        ctx.project_dir / "final.mp4",
        ctx.project_dir / "merged.mp4",
    ]
    for path in stale_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old")

    await main.draft_video_plan_impl(
        ctx,
        "WiFi",
        "Signals bend and fade.",
        [main.Scene(id="scene_1", narration="one", image_prompt="new image", video_prompt="new video", duration_seconds=5)],
    )

    assert not (ctx.project_dir / "voiceover").exists()
    assert not (ctx.project_dir / "images").exists()
    assert not (ctx.project_dir / "videos").exists()
    assert not (ctx.project_dir / "final.mp4").exists()
    assert not (ctx.project_dir / "merged.mp4").exists()
    assert not (ctx.project_dir / "manifest.json").exists()
    assert main.read_json_artifact(ctx, "images") is None
    assert main.read_project_state(ctx)["scene_assets"]["images"] == []


@pytest.mark.asyncio
async def test_image_generation_ignores_stale_files_in_scene_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeImageGenerator:
        def generate(self, **kwargs):
            return type(
                "Result",
                (),
                {
                    "id": "new-image-job",
                    "downloaded_paths": [],
                    "downloads": [type("Download", (), {"url": "https://example.test/new-image.png"})()],
                },
            )()

    class FakeMagicHourClient:
        def __init__(self, token: str):
            self.v1 = type("V1", (), {"ai_image_generator": FakeImageGenerator()})()

    class FakeResponse:
        content = b"new image"

        def raise_for_status(self) -> None:
            return None

    class FakeHttpClient:
        def __init__(self, timeout: int):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            assert url == "https://example.test/new-image.png"
            return FakeResponse()

    monkeypatch.setattr(media, "MagicHourClient", FakeMagicHourClient)
    monkeypatch.setattr(media.httpx, "Client", FakeHttpClient)
    ctx = main.ProjectContext(project_id="provider", project_dir=tmp_path, aspect_ratio="9:16", resolution="720p")
    scene = main.Scene(id="scene_1", narration="one", image_prompt="new image", video_prompt="new motion", duration_seconds=5)
    stale = tmp_path / "images" / "scene_1" / "old-dragon.png"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"old dragon")

    result = await main.generate_image_asset(ctx, scene)

    assert Path(result["path"]) == tmp_path / "images" / "scene_1" / "new-image.png"
    assert Path(result["path"]).read_bytes() == b"new image"
    assert not stale.exists()


def test_count_spoken_words_ignores_fish_audio_bracket_expression_cues() -> None:
    text = "[speaks softly] Rain taps the glass. [whispers] Listen closely. [pause]"

    assert main.count_spoken_words(text) == 6
    assert main.fish_audio_expression_cues(text) == ["speaks softly", "whispers", "pause"]


def test_normalize_plan_preserves_original_image_and_video_prompts() -> None:
    plan = main.VideoPlan(
        title="Continuity",
        narration="One. Two.",
        visual_bible="Same rainy alley, mustard raincoat, black cat, teal and amber palette, 35mm lens.",
        scenes=[
            main.Scene(
                id="a",
                narration="[speaks softly] The woman leaves the bakery.",
                image_prompt="Bakery doorway.",
                video_prompt="Slow push in.",
                duration_seconds=5,
            ),
            main.Scene(
                id="b",
                narration="The cat steps from the awning.",
                image_prompt="Cat near awning.",
                video_prompt="Tilt down.",
                duration_seconds=5,
            ),
        ],
    )

    normalized = main.normalize_plan(plan)

    assert normalized.scenes[0].id == "scene_1"
    assert normalized.scenes[0].image_prompt == "Bakery doorway."
    assert normalized.scenes[0].video_prompt == "Slow push in."
    assert normalized.scenes[1].image_prompt == "Cat near awning."
    assert normalized.scenes[1].video_prompt == "Tilt down."


def test_provider_image_prompt_prepends_visual_bible_for_independent_stills() -> None:
    plan = main.VideoPlan(
        title="Mechanic",
        narration="One.",
        visual_bible="Same late-40s woman mechanic: tan skin, silver-streaked tied-back curls, rectangular glasses, oil-stained green shirt, faded jeans, brown tool belt, red bandana.",
        scenes=[
            main.Scene(
                id="scene_1",
                narration="one",
                image_prompt="Same woman tightens a bolt beside the motorcycle at sunset.",
                video_prompt="Slow push in.",
                duration_seconds=5,
            )
        ],
    )

    prompt = main.provider_image_prompt(plan, plan.scenes[0])

    assert prompt.startswith("Continuity bible for every scene:")
    assert "late-40s woman mechanic" in prompt
    assert "red bandana" in prompt
    assert "Same woman tightens a bolt" in prompt


def test_provider_image_prompt_does_not_duplicate_existing_visual_bible() -> None:
    scene_prompt = "Same late-40s woman mechanic: tan skin, silver-streaked tied-back curls. She tightens a bolt."
    plan = main.VideoPlan(
        title="Mechanic",
        narration="One.",
        visual_bible="Same late-40s woman mechanic: tan skin, silver-streaked tied-back curls.",
        scenes=[main.Scene(id="scene_1", narration="one", image_prompt=scene_prompt, video_prompt="Slow push in.", duration_seconds=5)],
    )

    assert main.provider_image_prompt(plan, plan.scenes[0]) == scene_prompt


def test_normalize_plan_replaces_unsafe_scene_ids() -> None:
    plan = main.VideoPlan(
        title="Unsafe",
        narration="Narration.",
        scenes=[
            main.Scene(id="../one", narration="one", image_prompt="image", video_prompt="motion", duration_seconds=2),
            main.Scene(id="two/three", narration="two", image_prompt="image", video_prompt="motion", duration_seconds=2),
        ],
    )

    normalized = main.normalize_plan(plan)

    assert [scene.id for scene in normalized.scenes] == ["scene_1", "scene_2"]


def test_download_picker_uses_existing_file_over_stale_result_path(tmp_path: Path) -> None:
    real_file = tmp_path / "output-0.jpg"
    real_file.write_bytes(b"image")
    stale_file = tmp_path / "output-1.jpg"
    result = type("Result", (), {"downloaded_paths": [str(stale_file)]})()

    assert main.pick_download(result, tmp_path) == real_file


def test_token_output_payload_includes_gpt_usage_and_cost() -> None:
    payload = main.token_output_payload(
        "project",
        "gpt-5.4",
        Usage(
            requests=1,
            input_tokens=1000,
            input_tokens_details=InputTokensDetails(cached_tokens=200),
            output_tokens=500,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=50),
            total_tokens=1500,
        ),
    )

    assert payload["usage"]["input_tokens"] == 1000
    assert payload["usage"]["cached_input_tokens"] == 200
    assert payload["usage"]["uncached_input_tokens"] == 800
    assert payload["usage"]["output_tokens"] == 500
    assert payload["usage"]["reasoning_tokens"] == 50
    assert payload["cost"]["total_usd"] == 0.00955


@pytest.mark.asyncio
async def test_plan_video_writes_token_output_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_run(*args, **kwargs):
        assert "Narration budget" in kwargs["input"]
        assert "Fish Audio S2 expression cues" in kwargs["input"]
        usage = Usage(requests=1, input_tokens=1000, output_tokens=500, total_tokens=1500)
        plan = main.VideoPlan(
            title="Token Test",
            narration="Narration.",
            visual_bible="same subject",
            scenes=[
                main.Scene(
                    id="scene_1",
                    narration="one",
                    image_prompt="image",
                    video_prompt="slow push-in",
                    duration_seconds=1,
                )
            ],
        )
        return type("Result", (), {"final_output": plan, "context_wrapper": type("Wrapper", (), {"usage": usage})()})()

    monkeypatch.setattr(main.Runner, "run", fake_run)
    ctx = main.ProjectContext(project_id="project", project_dir=tmp_path, aspect_ratio="9:16", resolution="720p")

    plan, token_output = await main.plan_video(main.CreateProjectRequest(prompt="make a video"), ctx)

    data = json.loads((tmp_path / "token_output.json").read_text(encoding="utf-8"))
    assert plan.title == "Token Test"
    assert data["token_output_path"] == str(tmp_path / "token_output.json")
    assert token_output["cost"]["total_usd"] == data["cost"]["total_usd"]


@pytest.mark.asyncio
async def test_run_project_uses_video_agent_as_orchestrator(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    runner_calls = []

    async def fake_run(agent, input, *, context, **kwargs):
        runner_calls.append(agent)
        assert agent is main.video_agent
        assert "Narration budget" in input
        assert "Fish Audio S2 expression cues" in input
        pending_token_output = main.pending_token_output(context, main.video_agent.model)
        manifest = {
            "project_id": context.project_id,
            "title": "Agent Render",
            "final_video_path": str(context.project_dir / "final.mp4"),
            "manifest_path": str(context.project_dir / "manifest.json"),
            "failed_scene_count": 0,
            "token_output": pending_token_output,
            "token_output_path": pending_token_output["token_output_path"],
            "gpt_cost_usd": 0,
        }
        context.project_dir.mkdir(parents=True, exist_ok=True)
        (context.project_dir / "final.mp4").write_bytes(b"final")
        (context.project_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        usage = Usage(requests=1, input_tokens=1000, output_tokens=500, total_tokens=1500)
        return type("Result", (), {"final_output": "rendered", "context_wrapper": type("Wrapper", (), {"usage": usage})()})()

    async def direct_render_call(*args, **kwargs):
        raise AssertionError("run_project should let video_agent call render tools")

    monkeypatch.setattr(main, "render_plan", direct_render_call)
    monkeypatch.setattr(main.Runner, "run", fake_run)

    await main.run_project("a" * 32, main.CreateProjectRequest(prompt="make a video"))

    status = json.loads((tmp_path / ("a" * 32) / "status.json").read_text(encoding="utf-8"))
    manifest = status["manifest"]
    token_output = json.loads((tmp_path / ("a" * 32) / "token_output.json").read_text(encoding="utf-8"))
    assert runner_calls == [main.video_agent]
    assert status["status"] == "succeeded"
    assert manifest["title"] == "Agent Render"
    assert manifest["token_output"]["usage"]["input_tokens"] == 1000
    assert token_output["usage"]["input_tokens"] == 1000


def test_project_state_persists_request_preferences_and_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    project_id = "b" * 32
    request = main.CreateProjectRequest(
        prompt="make a quiet product teaser",
        duration_seconds=24,
        scene_count=5,
        aspect_ratio="16:9",
        resolution="1080p",
    )
    ctx = main.context(project_id, request)

    state = main.initialize_project_state(ctx, request)
    main.update_project_state(ctx, status={"stage": "queued", "progress": 0, "message": "queued"})
    reloaded = main.read_project_state(ctx)

    assert state["project_id"] == project_id
    assert reloaded["user_preferences"]["prompt"] == "make a quiet product teaser"
    assert reloaded["user_preferences"]["duration_seconds"] == 24
    assert reloaded["user_preferences"]["scene_count"] == 5
    assert reloaded["user_preferences"]["aspect_ratio"] == "16:9"
    assert reloaded["user_preferences"]["resolution"] == "1080p"
    assert reloaded["provider_settings"]["image_model"] == ctx.image_model
    assert reloaded["status"]["stage"] == "queued"
    assert (ctx.project_dir / "project_state.json").exists()


@pytest.mark.asyncio
async def test_read_project_status_includes_persistent_state_after_memory_clear(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    project_id = "c" * 32
    request = main.CreateProjectRequest(prompt="state survives restart")
    ctx = main.context(project_id, request)
    main.initialize_project_state(ctx, request)

    await main.update_project_status(
        project_id,
        status="running",
        stage="planning",
        progress=10,
        message="Planning.",
    )
    main.PROJECTS.pop(project_id, None)

    status = main.read_project_status(project_id)

    assert status is not None
    assert status["status"] == "running"
    assert status["project_state"]["user_preferences"]["prompt"] == "state survives restart"
    assert status["project_state"]["status"]["stage"] == "planning"


def test_main_agent_exposes_project_decision_memory_tool() -> None:
    tool_names = {getattr(tool, "name", None) for tool in main.video_agent.tools}

    assert "record_project_decision" in tool_names


@pytest.mark.asyncio
async def test_split_render_tools_persist_artifacts_and_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="split", project_dir=tmp_path / "split", aspect_ratio="9:16", resolution="720p")
    main.initialize_project_state(
        ctx,
        main.CreateProjectRequest(prompt="split render prompt", duration_seconds=20, scene_count=2),
    )
    scenes = [
        main.Scene(id="unsafe/one", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=2),
        main.Scene(id="custom-two", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=3),
    ]

    async def fake_voice(ctx_arg: main.ProjectContext, narration: str, duration_seconds: int) -> dict:
        path = ctx_arg.project_dir / "voiceover" / "voiceover.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"voice")
        return {"path": str(path), "model": "voice-model", "duration_seconds": 5, "target_duration_seconds": duration_seconds}

    async def fake_image(ctx_arg: main.ProjectContext, scene: main.Scene) -> dict:
        path = ctx_arg.project_dir / "images" / scene.id / "output.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": "image-model"}

    async def fake_video_batch(ctx_arg: main.ProjectContext, pairs: list[tuple[main.Scene, dict]]) -> list[dict]:
        videos = []
        for scene, _image in pairs:
            path = ctx_arg.project_dir / "videos" / scene.id / "output.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
            videos.append({"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": "video-model", "duration_seconds": scene.duration_seconds})
        return videos

    async def fake_stitch(ctx_arg: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        assert [video["scene_id"] for video in videos] == ["scene_1", "scene_2"]
        assert Path(voiceover["path"]).exists()
        path = ctx_arg.project_dir / "final.mp4"
        path.write_bytes(b"final")
        return str(path)

    monkeypatch.setattr(main, "generate_voiceover_asset", fake_voice)
    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_assets_batch", fake_video_batch)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)

    plan_payload = await main.draft_video_plan_impl(ctx, "Split Render", "One. Two.", scenes)
    voiceover = await main.generate_voiceover_impl(ctx)
    images_payload = await main.generate_scene_images_impl(ctx)
    videos_payload = await main.animate_scene_videos_impl(ctx)
    status = await main.inspect_render_status_impl(ctx)
    await main.record_project_decision_impl(ctx, "Keep the first completed take.", rationale="The asset is clean.", scene_id="scene_1")
    manifest = await main.stitch_final_video_impl(ctx, main.pending_token_output(ctx, "gpt-5.5"))
    state = main.read_project_state(ctx)

    assert plan_payload["plan"]["scenes"][0]["id"] == "scene_1"
    assert voiceover["voiceover"]["target_duration_seconds"] == 5
    assert [image["scene_id"] for image in images_payload["images"]] == ["scene_1", "scene_2"]
    assert [video["scene_id"] for video in videos_payload["videos"]] == ["scene_1", "scene_2"]
    assert status["artifacts"]["plan"] is True
    assert status["project_state"]["current_plan"]["title"] == "Split Render"
    assert status["completed_scene_count"] == 2
    assert manifest["render_status"] == "complete"
    assert manifest["final_video_url"] == "/media/split/final.mp4"
    assert state["current_plan"]["title"] == "Split Render"
    assert [image["scene_id"] for image in state["scene_assets"]["images"]] == ["scene_1", "scene_2"]
    assert [video["scene_id"] for video in state["scene_assets"]["videos"]] == ["scene_1", "scene_2"]
    assert state["scene_assets"]["voiceover"]["target_duration_seconds"] == 5
    assert state["scene_assets"]["final_video_path"].endswith("final.mp4")
    assert state["failures"] == []
    assert any(decision["decision"] == "Keep the first completed take." for decision in state["decisions"])
    assert (ctx.project_dir / "plan.json").exists()
    assert (ctx.project_dir / "project_state.json").exists()
    assert (ctx.project_dir / "voiceover.json").exists()
    assert (ctx.project_dir / "images.json").exists()
    assert (ctx.project_dir / "videos.json").exists()
    assert (ctx.project_dir / "manifest.json").exists()


@pytest.mark.asyncio
async def test_retry_scene_regenerates_bounded_scene_assets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="retry", project_dir=tmp_path / "retry", aspect_ratio="9:16", resolution="720p")
    scenes = [
        main.Scene(id="scene_1", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=2),
        main.Scene(id="scene_2", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=3),
    ]
    await main.draft_video_plan_impl(ctx, "Retry Render", "One. Two.", scenes)
    main.write_json_artifact(ctx, "images", [{"scene_id": "scene_2", "path": str(ctx.project_dir / "old.jpg")}])
    main.write_json_artifact(ctx, "videos", [{"scene_id": "scene_1", "path": str(ctx.project_dir / "scene_1.mp4")}])
    main.write_json_artifact(
        ctx,
        "failed_scenes",
        [{"scene_id": "scene_2", "stage": "video_generation", "error": "provider timeout"}],
    )

    async def fake_image(ctx_arg: main.ProjectContext, scene: main.Scene) -> dict:
        raise AssertionError("video retry should reuse the existing image")

    async def fake_video(ctx_arg: main.ProjectContext, scene: main.Scene, image: dict) -> dict:
        assert scene.id == "scene_2"
        assert image["path"].endswith("old.jpg")
        path = ctx_arg.project_dir / "videos" / scene.id / "retry.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": "video-model", "duration_seconds": scene.duration_seconds}

    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_asset", fake_video)

    payload = await main.retry_scene_impl(ctx, "scene_2", stage="video")
    state = main.read_project_state(ctx)

    assert payload["retried_scene_id"] == "scene_2"
    assert [video["scene_id"] for video in payload["videos"]] == ["scene_1", "scene_2"]
    assert main.read_json_artifact(ctx, "failed_scenes", []) == []
    assert state["failures"] == []
    assert state["scene_assets"]["videos"][1]["scene_id"] == "scene_2"


@pytest.mark.asyncio
async def test_regenerate_scene_patches_one_scene_and_restitches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="revise-scene", project_dir=tmp_path / "revise-scene", aspect_ratio="9:16", resolution="720p")
    main.initialize_project_state(ctx, main.CreateProjectRequest(prompt="revise one scene", scene_count=2))
    scenes = [
        main.Scene(id="scene_1", narration="one", image_prompt="old image one", video_prompt="old video one", duration_seconds=2),
        main.Scene(id="scene_2", narration="two", image_prompt="old image two", video_prompt="old video two", duration_seconds=3),
    ]
    await main.draft_video_plan_impl(ctx, "Revision Test", "One. Two.", scenes)
    old_image = {"scene_id": "scene_1", "path": str(ctx.project_dir / "images" / "scene_1" / "old.jpg"), "prompt": "old image one", "model": "image-model"}
    old_video = {"scene_id": "scene_1", "path": str(ctx.project_dir / "videos" / "scene_1" / "old.mp4"), "prompt": "old video one", "model": "video-model", "duration_seconds": 2}
    main.write_json_artifact(ctx, "images", [old_image])
    main.write_json_artifact(ctx, "videos", [old_video])
    main.write_json_artifact(ctx, "voiceover", {"path": str(ctx.project_dir / "voice.mp3"), "model": "voice-model", "duration_seconds": 5})

    async def fake_image(ctx_arg: main.ProjectContext, scene: main.Scene) -> dict:
        assert scene.id == "scene_2"
        assert scene.image_prompt == "new image two"
        path = ctx_arg.project_dir / "images" / scene.id / "new.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": "image-model"}

    async def fake_video(ctx_arg: main.ProjectContext, scene: main.Scene, image: dict) -> dict:
        assert scene.id == "scene_2"
        assert scene.video_prompt == "new video two"
        assert image["prompt"] == "new image two"
        path = ctx_arg.project_dir / "videos" / scene.id / "new.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": "video-model", "duration_seconds": scene.duration_seconds}

    async def fake_stitch(ctx_arg: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        assert [video["scene_id"] for video in videos] == ["scene_1", "scene_2"]
        path = ctx_arg.project_dir / "final_revised.mp4"
        path.write_bytes(b"final")
        return str(path)

    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_asset", fake_video)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)

    patched = await main.regenerate_scene_impl(
        ctx,
        "scene_2",
        image_prompt="new image two",
        video_prompt="new video two",
        narration="revised two",
    )
    manifest = await main.restitch_video_impl(ctx, main.pending_token_output(ctx, "gpt-5.5"), reason="scene_2 looked flat")
    plan = main.load_video_plan(ctx)
    state = main.read_project_state(ctx)

    assert plan.scenes[1].narration == "revised two"
    assert plan.scenes[1].image_prompt == "new image two"
    assert patched["scene"]["video_prompt"] == "new video two"
    assert [image["scene_id"] for image in patched["images"]] == ["scene_1", "scene_2"]
    assert patched["videos"][0]["path"] == old_video["path"]
    assert patched["videos"][1]["prompt"] == "new video two"
    assert manifest["final_video_path"].endswith("final_revised.mp4")
    assert state["scene_assets"]["final_video_path"].endswith("final_revised.mp4")
    assert any(decision["tool"] == "regenerate_scene" for decision in state["decisions"])
    assert any(decision["tool"] == "restitch_video" for decision in state["decisions"])


@pytest.mark.asyncio
async def test_revise_narration_invalidates_and_replaces_voiceover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    ctx = main.ProjectContext(project_id="revise-audio", project_dir=tmp_path / "revise-audio", aspect_ratio="9:16", resolution="720p")
    main.initialize_project_state(ctx, main.CreateProjectRequest(prompt="revise narration", scene_count=2))
    await main.draft_video_plan_impl(
        ctx,
        "Narration Revision",
        "Old full narration.",
        [
            main.Scene(id="scene_1", narration="old one", image_prompt="image one", video_prompt="video one", duration_seconds=2),
            main.Scene(id="scene_2", narration="old two", image_prompt="image two", video_prompt="video two", duration_seconds=3),
        ],
    )
    main.write_json_artifact(ctx, "voiceover", {"path": str(ctx.project_dir / "old_voice.mp3"), "model": "voice-model", "duration_seconds": 5})

    async def fake_voice(ctx_arg: main.ProjectContext, narration: str, duration_seconds: int) -> dict:
        assert narration == "New full narration."
        path = ctx_arg.project_dir / "voiceover" / "replacement.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"voice")
        return {"path": str(path), "model": "voice-model", "duration_seconds": 5, "target_duration_seconds": duration_seconds}

    monkeypatch.setattr(main, "generate_voiceover_asset", fake_voice)

    revised = await main.revise_narration_impl(
        ctx,
        "New full narration.",
        [main.SceneNarrationRevision(scene_id="scene_2", narration="new two")],
    )
    assert not (ctx.project_dir / "voiceover.json").exists()

    voiceover = await main.replace_voiceover_impl(ctx)
    plan = main.load_video_plan(ctx)
    state = main.read_project_state(ctx)

    assert revised["plan"]["narration"] == "New full narration."
    assert plan.scenes[1].narration == "new two"
    assert voiceover["voiceover"]["path"].endswith("replacement.mp3")
    assert state["scene_assets"]["voiceover"]["path"].endswith("replacement.mp3")
    assert any(decision["tool"] == "revise_narration" for decision in state["decisions"])
    assert any(decision["tool"] == "replace_voiceover" for decision in state["decisions"])


@pytest.mark.asyncio
async def test_image_tool_creates_download_directory_before_sdk_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeGenerator:
        def generate(self, **kwargs):
            download_dir = Path(kwargs["download_directory"])
            assert download_dir.exists()
            file_path = download_dir / "output-0.jpg"
            file_path.write_bytes(b"image")
            return type("Result", (), {"id": "job1", "downloads": [], "downloaded_paths": [str(file_path)]})()

    class FakeV1:
        ai_image_generator = FakeGenerator()

    class FakeMagicHourClient:
        def __init__(self, token: str):
            self.v1 = FakeV1()

    monkeypatch.setattr(media, "MagicHourClient", FakeMagicHourClient)
    scene = main.Scene(
        id="scene_1",
        narration="one",
        image_prompt="image prompt",
        video_prompt="video prompt",
        duration_seconds=1,
    )
    ctx = main.ProjectContext(
        project_id="project",
        project_dir=tmp_path,
        aspect_ratio="9:16",
        resolution="720p",
        magic_hour_api_key="key",
    )

    result = await main.generate_image_asset(ctx, scene)

    path = Path(result["path"])
    assert path.exists()
    assert path.parent == tmp_path / "images" / "scene_1"


@pytest.mark.asyncio
async def test_video_tool_keeps_download_inside_scene_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeVideoGenerator:
        def generate(self, **kwargs):
            download_dir = Path(kwargs["download_directory"])
            assert download_dir.exists()
            file_path = download_dir / "output-0.mp4"
            file_path.write_bytes(b"video")
            return type("Result", (), {"id": "job2", "downloads": [], "downloaded_paths": [str(file_path)]})()

    class FakeV1:
        image_to_video = FakeVideoGenerator()

    class FakeMagicHourClient:
        def __init__(self, token: str):
            self.v1 = FakeV1()

    monkeypatch.setattr(media, "MagicHourClient", FakeMagicHourClient)
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"image")
    scene = main.Scene(
        id="scene_1",
        narration="one",
        image_prompt="image prompt",
        video_prompt="slow push-in",
        duration_seconds=1,
    )
    ctx = main.ProjectContext(
        project_id="project",
        project_dir=tmp_path,
        aspect_ratio="9:16",
        resolution="720p",
        magic_hour_api_key="key",
    )

    result = await main.generate_video_asset(ctx, scene, {"path": str(image_path)})

    path = Path(result["path"])
    assert path.exists()
    assert path.parent == tmp_path / "videos" / "scene_1"


@pytest.mark.asyncio
async def test_video_tool_downloads_provider_url_when_sdk_does_not_write_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeVideoGenerator:
        def generate(self, **kwargs):
            download_dir = Path(kwargs["download_directory"])
            assert download_dir.exists()
            return type(
                "Result",
                (),
                {
                    "id": "job2",
                    "status": "complete",
                    "error": None,
                    "downloads": [
                        type("Download", (), {"url": "https://example.test/render/output.mp4"})(),
                    ],
                    "downloaded_paths": [],
                },
            )()

    class FakeV1:
        image_to_video = FakeVideoGenerator()

    class FakeMagicHourClient:
        def __init__(self, token: str):
            self.v1 = FakeV1()

    class FakeResponse:
        content = b"video"

        def raise_for_status(self) -> None:
            return None

    class FakeHttpClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str):
            assert url == "https://example.test/render/output.mp4"
            return FakeResponse()

    monkeypatch.setattr(media, "MagicHourClient", FakeMagicHourClient)
    monkeypatch.setattr(media.httpx, "Client", FakeHttpClient)
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"image")
    scene = main.Scene(
        id="scene_1",
        narration="one",
        image_prompt="image prompt",
        video_prompt="slow push-in",
        duration_seconds=1,
    )
    ctx = main.ProjectContext(
        project_id="project",
        project_dir=tmp_path,
        aspect_ratio="9:16",
        resolution="720p",
        magic_hour_api_key="key",
    )

    result = await main.generate_video_asset(ctx, scene, {"path": str(image_path)})

    path = Path(result["path"])
    assert path == tmp_path / "videos" / "scene_1" / "output.mp4"
    assert path.read_bytes() == b"video"


@pytest.mark.asyncio
async def test_video_batch_submits_all_jobs_before_polling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scenes = [
        main.Scene(id="scene1", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=1),
        main.Scene(id="scene2", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=1),
        main.Scene(id="scene3", narration="three", image_prompt="image three", video_prompt="video three", duration_seconds=1),
    ]
    ctx = main.ProjectContext(project_id="batch", project_dir=tmp_path, aspect_ratio="9:16", resolution="720p")
    pairs = [(scene, {"path": str(tmp_path / f"{scene.id}.jpg")}) for scene in scenes]
    submitted: list[str] = []
    polled: list[str] = []
    all_submitted = asyncio.Event()

    async def fake_submit(ctx_arg: main.ProjectContext, scene: main.Scene, image: dict) -> media.VideoAssetJob:
        submitted.append(scene.id)
        if len(submitted) == len(scenes):
            all_submitted.set()
        return media.VideoAssetJob(
            scene=scene,
            image=image,
            out_dir=ctx_arg.project_dir / "videos" / scene.id,
            provider_job_id=f"job-{scene.id}",
            prompt=scene.video_prompt,
            model=ctx_arg.video_model,
            resolution=ctx_arg.resolution,
            audio=ctx_arg.video_audio,
            duration_seconds=scene.duration_seconds,
            submitted_status="queued",
        )

    async def fake_poll(ctx_arg: main.ProjectContext, job: media.VideoAssetJob) -> dict:
        await asyncio.wait_for(all_submitted.wait(), timeout=1)
        polled.append(job.scene.id)
        output = job.out_dir / "output.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"video")
        return {"scene_id": job.scene.id, "path": str(output), "provider_job_id": job.provider_job_id}

    monkeypatch.setattr(media, "submit_video_asset_job", fake_submit)
    monkeypatch.setattr(media, "poll_video_asset_job", fake_poll)

    results = await media.generate_video_assets_batch(ctx, pairs)

    assert submitted == ["scene1", "scene2", "scene3"]
    assert polled == ["scene1", "scene2", "scene3"]
    assert [result["scene_id"] for result in results] == ["scene1", "scene2", "scene3"]


@pytest.mark.asyncio
async def test_stitch_assets_pads_short_voiceover_to_planned_final_duration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[tuple[str, list[str]]] = []

    async def fake_run_ffmpeg(cmd: list[str], label: str) -> None:
        commands.append((label, cmd))

    async def fake_probe(path: str | Path) -> float:
        return 12.0 if str(path).endswith("voice.mp3") else 10.0

    monkeypatch.setattr(media, "_run_ffmpeg", fake_run_ffmpeg)
    monkeypatch.setattr(media, "_run_ffprobe_duration", fake_probe)
    ctx = main.ProjectContext(project_id="timed", project_dir=tmp_path, aspect_ratio="9:16", resolution="720p")
    videos = [
        {"scene_id": "scene_1", "path": str(tmp_path / "scene_1.mp4"), "duration_seconds": 10},
        {"scene_id": "scene_2", "path": str(tmp_path / "scene_2.mp4"), "duration_seconds": 10},
    ]
    voiceover = {"path": str(tmp_path / "voice.mp3"), "duration_seconds": 12}

    final = await media.stitch_assets(ctx, videos, voiceover)

    final_cmd = next(cmd for label, cmd in commands if label == "voiceover mux ffmpeg")
    assert final == str(tmp_path / "final.mp4")
    assert "-shortest" not in final_cmd
    assert final_cmd[final_cmd.index("-af") + 1] == "apad"
    assert final_cmd[final_cmd.index("-t") + 1] == "19.500"


@pytest.mark.asyncio
async def test_generation_runs_images_and_videos_in_parallel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = main.VideoPlan(
        title="Parallel Test",
        narration="Fast narration.",
        scenes=[
            main.Scene(id="scene1", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=1),
            main.Scene(id="scene2", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=1),
            main.Scene(id="scene3", narration="three", image_prompt="image three", video_prompt="video three", duration_seconds=1),
        ],
    )
    image_started: list[str] = []
    video_started: list[str] = []
    all_images_started = asyncio.Event()
    all_videos_started = asyncio.Event()

    async def fake_image(ctx: main.ProjectContext, scene: main.Scene) -> dict:
        image_started.append(scene.id)
        if len(image_started) == 3:
            all_images_started.set()
        await asyncio.wait_for(all_images_started.wait(), timeout=1)
        path = ctx.project_dir / f"{scene.id}.png"
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path)}

    async def fake_video(ctx: main.ProjectContext, scene: main.Scene, image: dict) -> dict:
        video_started.append(scene.id)
        if len(video_started) == 3:
            all_videos_started.set()
        await asyncio.wait_for(all_videos_started.wait(), timeout=1)
        path = ctx.project_dir / f"{scene.id}.mp4"
        path.write_bytes(b"video")
        return {"scene_id": scene.id, "path": str(path)}

    async def fake_video_batch(ctx: main.ProjectContext, pairs: list[tuple[main.Scene, dict]]) -> list[dict]:
        return await asyncio.gather(*(fake_video(ctx, scene, image) for scene, image in pairs))

    async def fake_voice(ctx: main.ProjectContext, narration: str, duration_seconds: int) -> dict:
        path = ctx.project_dir / "voice.mp3"
        path.write_bytes(b"voice")
        return {"path": str(path)}

    async def fake_stitch(ctx: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        path = ctx.project_dir / "final.mp4"
        path.write_bytes(b"final")
        return str(path)

    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_assets_batch", fake_video_batch)
    monkeypatch.setattr(main, "generate_voiceover_asset", fake_voice)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)

    manifest = await main.render_plan(
        plan,
        main.ProjectContext(
            project_id="parallel",
            project_dir=tmp_path / "parallel",
            aspect_ratio="9:16",
            resolution="720p",
        ),
        {
            "token_output_path": str(tmp_path / "token_output.json"),
            "cost": {"total_usd": 0.01},
        },
    )

    assert image_started == ["scene1", "scene2", "scene3"]
    assert video_started == ["scene1", "scene2", "scene3"]
    assert Path(manifest["final_video_path"]).exists()
    assert manifest["gpt_cost_usd"] == 0.01


@pytest.mark.asyncio
async def test_render_plan_stitches_successful_segments_when_one_video_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = main.VideoPlan(
        title="Partial Test",
        narration="Fast narration.",
        scenes=[
            main.Scene(id="scene1", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=1),
            main.Scene(id="scene2", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=1),
            main.Scene(id="scene3", narration="three", image_prompt="image three", video_prompt="video three", duration_seconds=1),
        ],
    )

    async def fake_image(ctx: main.ProjectContext, scene: main.Scene) -> dict:
        path = ctx.project_dir / "images" / scene.id / "output.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": "image-model"}

    async def fake_video(ctx: main.ProjectContext, scene: main.Scene, image: dict) -> dict:
        if scene.id == "scene3":
            raise RuntimeError("provider did not return the final segment")
        path = ctx.project_dir / "videos" / scene.id / "output.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": "video-model", "duration_seconds": scene.duration_seconds}

    async def fake_video_batch(ctx: main.ProjectContext, pairs: list[tuple[main.Scene, dict]]) -> list[dict | Exception]:
        return await asyncio.gather(
            *(fake_video(ctx, scene, image) for scene, image in pairs),
            return_exceptions=True,
        )

    async def fake_voice(ctx: main.ProjectContext, narration: str, duration_seconds: int) -> dict:
        path = ctx.project_dir / "voiceover" / "voiceover.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"voice")
        return {"path": str(path), "model": "voice-model", "duration_seconds": duration_seconds}

    async def fake_stitch(ctx: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        assert [video["scene_id"] for video in videos] == ["scene1", "scene2"]
        path = ctx.project_dir / "final.mp4"
        path.write_bytes(b"partial final")
        return str(path)

    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_assets_batch", fake_video_batch)
    monkeypatch.setattr(main, "generate_voiceover_asset", fake_voice)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)
    ctx = main.ProjectContext(project_id="partial", project_dir=tmp_path / "partial", aspect_ratio="9:16", resolution="720p")

    manifest = await main.render_plan(
        plan,
        ctx,
        {
            "token_output_path": str(ctx.project_dir / "token_output.json"),
            "cost": {"total_usd": 0.01},
        },
    )

    assert manifest["render_status"] == "partial"
    assert [scene["scene_id"] for scene in manifest["failed_scenes"]] == ["scene3"]
    assert Path(manifest["final_video_path"]).exists()


def test_backend_health() -> None:
    response = TestClient(main.app).get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "missing_config"}
    assert body["output_dir"] == str(main.OUTPUT_DIR)
    assert "missing_config" in body


def test_public_media_path_is_relative_to_output_dir(tmp_path: Path) -> None:
    project_file = main.OUTPUT_DIR / "project-1" / "final.mp4"

    assert main.public_media_path(project_file) == "/media/project-1/final.mp4"


def test_public_media_path_rejects_files_outside_output_dir(tmp_path: Path) -> None:
    outside_file = tmp_path / "final.mp4"

    with pytest.raises(ValueError, match="outside output directory"):
        main.public_media_path(outside_file)


def test_project_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Invalid project id"):
        main.project_dir_for("../manifest")


@pytest.mark.asyncio
async def test_create_project_returns_pollable_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "missing_configuration", lambda: [])
    monkeypatch.setattr(main, "missing_system_dependencies", lambda: [])
    started = threading.Event()

    async def fake_run_project(project_id: str, request: main.CreateProjectRequest) -> None:
        started.set()
        await main.update_project_status(
            project_id,
            status="succeeded",
            stage="complete",
            progress=100,
            message="Ready.",
            manifest={"project_id": project_id, "title": "Done"},
        )

    monkeypatch.setattr(main, "run_project", fake_run_project)

    with TestClient(main.app) as client:
        response = client.post("/api/projects", json={"prompt": "make a polished launch video"})
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["status_url"] == f"/api/projects/{body['project_id']}"

        assert started.wait(timeout=1)
        status = client.get(body["status_url"]).json()

    assert status["status"] == "succeeded"
    assert status["manifest"]["title"] == "Done"
    assert (tmp_path / body["project_id"] / "status.json").exists()


@pytest.mark.asyncio
async def test_message_endpoint_queues_agent_turn_for_existing_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "missing_configuration", lambda: [])
    monkeypatch.setattr(main, "missing_system_dependencies", lambda: [])
    project_id = "d" * 32
    request = main.CreateProjectRequest(prompt="initial render", scene_count=2)
    ctx = main.context(project_id, request)
    main.initialize_project_state(ctx, request)
    await main.update_project_status(
        project_id,
        status="succeeded",
        stage="complete",
        progress=100,
        message="Video is ready.",
        manifest={"project_id": project_id, "title": "Initial"},
    )
    started = threading.Event()

    async def fake_run_project_message(project_id_arg: str, message: str) -> None:
        assert project_id_arg == project_id
        assert message == "make scene_2 brighter"
        started.set()
        await main.update_project_status(
            project_id_arg,
            status="succeeded",
            stage="message_complete",
            progress=100,
            message="Handled message.",
        )

    monkeypatch.setattr(main, "run_project_message", fake_run_project_message)

    with TestClient(main.app) as client:
        response = client.post(
            f"/api/projects/{project_id}/messages",
            json={"message": "make scene_2 brighter"},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["stage"] == "message_queued"
        assert body["project_state"]["messages"][-1]["role"] == "user"
        assert body["project_state"]["messages"][-1]["content"] == "make scene_2 brighter"

        assert started.wait(timeout=1)
        status = client.get(f"/api/projects/{project_id}").json()

    assert status["stage"] == "message_complete"


@pytest.mark.asyncio
async def test_run_project_message_uses_video_agent_with_persistent_project_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    project_id = "e" * 32
    request = main.CreateProjectRequest(prompt="initial render", scene_count=2, aspect_ratio="16:9")
    ctx = main.context(project_id, request)
    main.initialize_project_state(ctx, request)
    await main.update_project_status(
        project_id,
        status="succeeded",
        stage="complete",
        progress=100,
        message="Video is ready.",
        manifest={"project_id": project_id, "title": "Initial"},
    )
    main.append_project_message(ctx, role="user", content="make scene_2 brighter")
    runner_inputs: list[str] = []

    async def fake_run(agent, input, *, context, **kwargs):
        runner_inputs.append(input)
        assert agent is main.video_agent
        assert context.project_id == project_id
        assert context.aspect_ratio == "16:9"
        assert kwargs["max_turns"] > 10
        assert "make scene_2 brighter" in input
        assert "project_state.json" in input
        assert '"messages"' in input
        usage = Usage(requests=1, input_tokens=700, output_tokens=100, total_tokens=800)
        return type(
            "Result",
            (),
            {
                "final_output": "I brightened scene_2 and restitched the edit.",
                "context_wrapper": type("Wrapper", (), {"usage": usage})(),
            },
        )()

    monkeypatch.setattr(main.Runner, "run", fake_run)

    await main.run_project_message(project_id, "make scene_2 brighter")

    status = main.read_project_status(project_id)
    state = main.read_project_state(ctx)
    token_output = json.loads((ctx.project_dir / "token_output.json").read_text(encoding="utf-8"))
    assert runner_inputs
    assert status is not None
    assert status["stage"] == "message_complete"
    assert state["messages"][-1]["role"] == "assistant"
    assert state["messages"][-1]["content"] == "I brightened scene_2 and restitched the edit."
    assert token_output["usage"]["input_tokens"] == 700


@pytest.mark.asyncio
async def test_run_project_uses_extended_agent_turn_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    project_id = "c" * 32
    request = main.CreateProjectRequest(prompt="make a solar explainer")
    seen: dict[str, int] = {}

    async def fake_run(agent, input, *, context, **kwargs):
        seen["max_turns"] = kwargs["max_turns"]
        assert agent is main.video_agent
        usage = Usage(requests=1, input_tokens=700, output_tokens=100, total_tokens=800)
        ctx = context
        main.write_json_artifact(
            ctx,
            "manifest",
            {
                "project_id": ctx.project_id,
                "title": "Done",
                "token_output_path": str(ctx.project_dir / "token_output.json"),
                "cost": {"total_usd": 0},
            },
        )
        return type(
            "Result",
            (),
            {
                "final_output": "Done",
                "context_wrapper": type("Wrapper", (), {"usage": usage})(),
            },
        )()

    monkeypatch.setattr(main.Runner, "run", fake_run)

    await main.run_project(project_id, request)

    assert seen["max_turns"] > 10


@pytest.mark.asyncio
async def test_run_project_message_does_not_reuse_previous_manifest_after_new_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    project_id = "f" * 32
    request = main.CreateProjectRequest(prompt="dragon video", scene_count=2)
    ctx = main.context(project_id, request)
    main.initialize_project_state(ctx, request)
    previous_manifest = {"project_id": project_id, "title": "Dragon", "images": [{"scene_id": "scene_1", "path": "dragon.png"}]}
    main.write_json_artifact(ctx, "manifest", previous_manifest)
    await main.update_project_status(
        project_id,
        status="succeeded",
        stage="complete",
        progress=100,
        message="Video is ready.",
        manifest=previous_manifest,
    )
    main.append_project_message(ctx, role="user", content="now make a wifi video")

    async def fake_run(agent, input, *, context, **kwargs):
        await main.draft_video_plan_impl(
            context,
            "WiFi",
            "WiFi waves bend and fade.",
            [main.Scene(id="scene_1", narration="one", image_prompt="wifi image", video_prompt="wifi motion", duration_seconds=5)],
        )
        usage = Usage(requests=1, input_tokens=700, output_tokens=100, total_tokens=800)
        return type(
            "Result",
            (),
            {
                "final_output": "I drafted a WiFi plan and queued fresh rendering.",
                "context_wrapper": type("Wrapper", (), {"usage": usage})(),
            },
        )()

    monkeypatch.setattr(main.Runner, "run", fake_run)

    await main.run_project_message(project_id, "now make a wifi video")

    status = main.read_project_status(project_id)
    state = main.read_project_state(ctx)
    assert status is not None
    assert status["stage"] == "message_complete"
    assert status.get("manifest") is None
    assert state["current_plan"]["title"] == "WiFi"
    assert state["scene_assets"]["images"] == []


@pytest.mark.asyncio
async def test_render_plan_emits_progress_and_media_urls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plan = main.VideoPlan(
        title="Progress Test",
        narration="Fast narration.",
        scenes=[
            main.Scene(id="scene1", narration="one", image_prompt="image one", video_prompt="video one", duration_seconds=1),
            main.Scene(id="scene2", narration="two", image_prompt="image two", video_prompt="video two", duration_seconds=1),
        ],
    )
    progress: list[tuple[str, int]] = []

    async def fake_image(ctx: main.ProjectContext, scene: main.Scene) -> dict:
        path = ctx.project_dir / "images" / scene.id / "output.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.image_prompt, "model": "image-model"}

    async def fake_video(ctx: main.ProjectContext, scene: main.Scene, image: dict) -> dict:
        path = ctx.project_dir / "videos" / scene.id / "output.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"video")
        return {"scene_id": scene.id, "path": str(path), "prompt": scene.video_prompt, "model": "video-model", "duration_seconds": scene.duration_seconds}

    async def fake_video_batch(ctx: main.ProjectContext, pairs: list[tuple[main.Scene, dict]]) -> list[dict]:
        return await asyncio.gather(*(fake_video(ctx, scene, image) for scene, image in pairs))

    async def fake_voice(ctx: main.ProjectContext, narration: str, duration_seconds: int) -> dict:
        path = ctx.project_dir / "voiceover" / "voiceover.mp3"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"voice")
        return {"path": str(path), "model": "voice-model", "duration_seconds": duration_seconds}

    async def fake_stitch(ctx: main.ProjectContext, videos: list[dict], voiceover: dict) -> str:
        path = ctx.project_dir / "final.mp4"
        path.write_bytes(b"final")
        return str(path)

    async def capture(stage: str, progress_value: int, message: str) -> None:
        progress.append((stage, progress_value))

    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(main, "generate_image_asset", fake_image)
    monkeypatch.setattr(main, "generate_video_assets_batch", fake_video_batch)
    monkeypatch.setattr(main, "generate_voiceover_asset", fake_voice)
    monkeypatch.setattr(main, "stitch_assets", fake_stitch)
    ctx = main.ProjectContext(project_id="progress", project_dir=tmp_path / "progress", aspect_ratio="9:16", resolution="720p")

    manifest = await main.render_plan(
        plan,
        ctx,
        {
            "token_output_path": str(ctx.project_dir / "token_output.json"),
            "cost": {"total_usd": 0.01},
        },
        on_progress=capture,
    )

    assert progress == [
        ("voiceover_images", 30),
        ("video_generation", 65),
        ("stitching", 90),
    ]
    assert manifest["final_video_url"] == "/media/progress/final.mp4"
    assert manifest["images"][0]["url"] == "/media/progress/images/scene1/output.jpg"
    assert manifest["videos"][0]["url"] == "/media/progress/videos/scene1/output.mp4"
