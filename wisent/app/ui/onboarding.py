"""First-use journey presentation and Gradio callback wiring."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import gradio as gr

from wisent.app.onboarding import JourneyRuntime

_TITLES = {
    "welcome.title": "See what representations reveal",
    "result.title": "Create and inspect a representation result",
}
_BODIES = {
    "welcome.body": (
        "Wisent works with model representations: it can generate contrastive data, "
        "extract activations, build steering directions, compare geometry, and visualize "
        "how a direction changes model behavior. Start with a visualization so the result "
        "is visible alongside the command output."
    ),
    "result.body": (
        "The **Steering → steering-viz** operation turns activation-space effects into "
        "a visual result. Configure the model and required inputs, then run it. This journey "
        "finishes only after Gradio has rendered real result text or a generated visualization."
    ),
}
_RESULT_COMMAND_EXCLUSIONS = frozenset(
    {"agent", "inference-config", "optimization-cache", "tasks"}
)
_BROWSER_SUBJECT_JS = """
(current) => {
  const key = "wisent.onboarding.wisent-gradio.subject";
  let subject = window.localStorage.getItem(key);
  if (!subject) {
    if (window.crypto && window.crypto.randomUUID) {
      subject = window.crypto.randomUUID();
    } else {
      const bytes = new Uint8Array(16);
      window.crypto.getRandomValues(bytes);
      bytes[6] = (bytes[6] & 15) | 64;
      bytes[8] = (bytes[8] & 63) | 128;
      const hex = Array.from(bytes, b => b.toString(16).padStart(2, "0")).join("");
      subject = `${hex.slice(0,8)}-${hex.slice(8,12)}-${hex.slice(12,16)}-${hex.slice(16,20)}-${hex.slice(20)}`;
    }
    window.localStorage.setItem(key, subject);
  }
  return subject;
}
"""


def build_onboarding_panel() -> Dict[str, Any]:
    """Build product-owned first-use content above the normal command tabs."""
    subject = gr.Textbox(value="", visible=False, elem_id="onboarding-subject")
    with gr.Accordion("First-use representation journey", open=True) as panel:
        title = gr.Markdown("### Preparing your first-use journey…")
        body = gr.Markdown(
            "Wisent will guide you from representation operations to a rendered result."
        )
        progress = gr.Markdown("Loading saved progress…")
        action = gr.Button("Open Steering visualization", variant="primary")
    return {
        "subject": subject,
        "panel": panel,
        "title": title,
        "body": body,
        "progress": progress,
        "action": action,
    }


def _view(runtime: JourneyRuntime):
    progress = runtime.progress
    screen = runtime.screen
    completed = progress.get("status") == "completed"
    if completed:
        title = "### First representation result observed"
        body = (
            "Your rendered command result is the first-success evidence for this journey. "
            "You can continue exploring generation, steering, evaluation, and analysis operations below."
        )
        status = "**Journey complete** · result evidence saved for this browser device."
        action = gr.update(value="Open Steering visualization", visible=True)
        return title, body, status, action
    screens = runtime.bundle["definition"]["screens"]
    position = next(
        (index for index, item in enumerate(screens, start=1) if item["screen_id"] == screen["screen_id"]),
        1,
    )
    title = f"### {_TITLES[screen['title_key']]}"
    body = _BODIES[screen["body_key"]]
    status = f"**Step {position} of {len(screens)}** · progress is saved automatically."
    action = gr.update(value="Open Steering visualization", visible=True)
    return title, body, status, action


def load_journey(browser_subject: str):
    """Load or resume the device-scoped journey when the Gradio page opens."""
    return browser_subject, *_view(JourneyRuntime(browser_subject).start())


def primary_action(browser_subject: str):
    """Advance the explanation and route the normal UI to steering visualization."""
    runtime = JourneyRuntime(browser_subject).open_existing().primary_action()
    return (*_view(runtime), gr.Tabs(selected="Steering"), gr.Tabs(selected="steering-viz"))


def _has_rendered_result(text: Any, images: Any, detail: Any) -> bool:
    if isinstance(detail, str) and detail.strip():
        return False
    if images:
        return True
    if not isinstance(text, str) or not text.strip():
        return False
    normalized = text.strip().lower()
    rejected_prefixes = (
        "argument parsing failed",
        "unknown command:",
        "handler not found:",
        "command completed successfully (no output).",
        "--- stderr ---",
    )
    return not normalized.startswith(rejected_prefixes)


def observe_rendered_result(
    browser_subject: str,
    command_name: str,
    text: Any,
    images: Any,
    detail: Any,
):
    """Record first success only after a command's output components were updated."""
    runtime = JourneyRuntime(browser_subject).open_existing()
    if command_name not in _RESULT_COMMAND_EXCLUSIONS and _has_rendered_result(text, images, detail):
        runtime.observe_representation_result(command_name)
    return _view(runtime)


def view_outputs(components: Mapping[str, Any]) -> Iterable[Any]:
    return (
        components["title"],
        components["body"],
        components["progress"],
        components["action"],
    )


def wire_page_load(app: gr.Blocks, components: Mapping[str, Any]) -> None:
    """Resolve the stable browser subject before loading persisted progress."""
    app.load(
        fn=load_journey,
        inputs=[components["subject"]],
        outputs=[components["subject"], *view_outputs(components)],
        js=_BROWSER_SUBJECT_JS,
        show_progress="hidden",
    )


def wire_primary_action(
    components: Mapping[str, Any],
    outer_tabs: gr.Tabs,
    steering_tabs: gr.Tabs,
) -> None:
    components["action"].click(
        fn=primary_action,
        inputs=[components["subject"]],
        outputs=[*view_outputs(components), outer_tabs, steering_tabs],
        show_progress="hidden",
    )
