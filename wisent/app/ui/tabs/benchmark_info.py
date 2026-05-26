"""Benchmark info panel — detailed metadata for the debug tab."""

import inspect
from typing import Any

import gradio as gr

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    TEST_EXTRACTOR_EVALUATOR_DEFAULT_LIMIT,
)


def get_contrastive_pair_logic(extractor: Any) -> str:
    """Extract the contrastive pair generation approach from source."""
    cls = type(extractor)
    doc = cls.extract_contrastive_pairs.__doc__
    if doc:
        return doc.strip()
    # Try to find _extract_pair_from_doc or _create_incorrect
    for method_name in ("_extract_pair_from_doc",
                        "_create_incorrect_answer",
                        "_create_incorrect_response"):
        method = getattr(cls, method_name, None)
        if method and method.__doc__:
            return f"{method_name}: {method.__doc__.strip()}"
    return "No documentation available"


def get_evaluator_logic(evaluator_name: str) -> str:
    """Get the evaluator methodology description."""
    from wisent.core.reading.evaluators.core.atoms import BaseEvaluator
    import wisent.core.reading.evaluators.core.benchmark_specific  # noqa: F401
    try:
        cls = BaseEvaluator.get(evaluator_name)
        parts = []
        if cls.__doc__:
            parts.append(cls.__doc__.strip())
        if hasattr(cls, "requires_judge"):
            parts.append(f"Requires judge: {cls.requires_judge}")
        evaluate_doc = cls.evaluate.__doc__
        if evaluate_doc:
            parts.append(f"evaluate(): {evaluate_doc.strip()}")
        return "\n".join(parts) if parts else "No documentation"
    except Exception as exc:
        return f"Error: {exc}"


def get_all_pairs(extractor: Any) -> list[dict]:
    """Every contrastive pair for the task, untruncated, in extraction
    order (the same order the pos_<i>/neg_<i> activation tensors use)."""
    try:
        pairs = extractor.extract_contrastive_pairs()
    except TypeError:
        try:
            pairs = extractor.extract_contrastive_pairs(lm_eval_task_data=None)
        except Exception:
            return []
    except Exception:
        return []
    return [{
        "prompt": pair.prompt,
        "correct": pair.positive_response.model_response,
        "incorrect": pair.negative_response.model_response,
    } for pair in pairs]


def _provenance_block(task_name: str) -> str:
    """Exact HF source of this task's pairs and activations. The <model>,
    <prompt_format> and <L> placeholders are fixed by the model/layer
    selectors elsewhere in the tab."""
    from wisent.app.ui.tabs.debug.benchmark_artifacts import _category
    cat = _category(task_name)
    pt = (f"pair_texts/{cat}/{task_name}.json" if cat
          else f"pair_texts/{task_name}.json")
    return (
        "\n**Provenance — exact source on HuggingFace:**\n"
        f"- Pairs: `wisent-ai/activations` -> `{pt}` "
        "(pid key == pos_<pid>/neg_<pid> in the tensors)\n"
        "- Prompt formats: `chat`, `mc_balanced`, `role_play`\n"
        "- Activations: `wisent-ai/activations` -> "
        f"`raw_activations/<model>/{task_name}/<prompt_format>/"
        "layer_<L>_chunk_<C>.safetensors`"
    )


def format_pairs_by_format(task_name: str, model_name: str, count: int) -> str:
    """Render the first `count` contrastive pairs for a benchmark in each
    prompt format (chat / mc_balanced / role_play) — the actual text fed to
    the model — using the selected model's tokenizer chat template. Empty
    model => can't render chat templates, so we say so rather than guess."""
    from wisent.core.primitives.model_interface.core.activations import (
        ExtractionStrategy, build_extraction_texts,
    )
    from wisent.app.ui.tabs.debug.benchmark_tokens import source_pairs
    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]
    if not model_name:
        return "_Select a model — the chat/role_play formats need its tokenizer template._"
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:
        return f"Error loading tokenizer for `{model_name}`: {exc}"
    n = int(count) if count else None  # empty Number => all pairs
    try:
        pairs = source_pairs(task_name, n)
    except Exception as exc:
        return f"_Could not generate pairs for `{task_name}`: {exc}_"
    fmt_strat = [
        ("chat", ExtractionStrategy.CHAT_LAST),
        ("mc_balanced", ExtractionStrategy.MC_BALANCED),
        ("role_play", ExtractionStrategy.ROLE_PLAY),
    ]
    lines = [f"**Pairs by format — `{task_name}` via `{model_name}` "
             f"(source build_contrastive_pairs, showing {len(pairs)}; "
             f"pid == activation tensor index):**"]
    for pid, (prompt, pos, neg) in enumerate(pairs):
        lines.append(f"\n### Pair {pid}\n- raw prompt: {prompt}\n"
                     f"- raw positive: {pos!r}\n- raw negative: {neg!r}")
        for label, strat in fmt_strat:
            try:
                full, _ans, _po = build_extraction_texts(
                    strat, prompt, pos, tok, other_response=neg, is_positive=True,
                )
                lines.append(f"\n**{label}** — full text fed to the model (positive arm):")
                lines.append(f"```\n{full}\n```")
            except Exception as exc:
                lines.append(f"\n**{label}**: render error — {exc}")
    return "\n".join(lines)


def format_full_info(task_name: str) -> str:
    """Build complete benchmark info string."""
    from wisent.extractors.lm_eval.lm_extractor_registry import (
        get_extractor, _REGISTRY,
    )
    from wisent.core.utils.services.benchmarks.registry.benchmark_registry import (
        get_working_benchmarks_with_categories,
    )

    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]

    all_names = sorted(_REGISTRY.keys())
    categories = get_working_benchmarks_with_categories()
    cat = categories.get(task_name, "uncategorized")

    lines = [f"**{task_name}** — `{cat}`\n"]

    try:
        ext = get_extractor(task_name)
    except Exception as exc:
        return f"**{task_name}**\n\nError loading extractor: {exc}"

    ev_name = getattr(ext, "evaluator_name", None)
    lines.append(f"**Extractor:** `{type(ext).__name__}` "
                 f"(`{type(ext).__module__}`)")
    lines.append(f"**Evaluator:** `{ev_name}`")

    # Aliases
    ref = _REGISTRY.get(task_name)
    aliases = [n for n in all_names if _REGISTRY.get(n) == ref and n != task_name]
    if aliases:
        lines.append(f"**Aliases:** {', '.join(aliases)}")

    # Subtasks
    prefix = task_name + "_"
    subtasks = [n for n in all_names if n.startswith(prefix)]
    if subtasks:
        lines.append(f"**Subtasks:** {len(subtasks)}")

    # Contrastive pair logic
    lines.append(f"\n**Contrastive Pair Logic:**")
    lines.append(f"```\n{get_contrastive_pair_logic(ext)}\n```")

    # Evaluator logic
    if ev_name:
        lines.append(f"\n**Evaluator Logic ({ev_name}):**")
        lines.append(f"```\n{get_evaluator_logic(ev_name)}\n```")

    # Provenance: exact HF source of pairs + activations
    lines.append(_provenance_block(task_name))

    # All contrastive pairs (untruncated, full set — not just 3 examples)
    pairs = get_all_pairs(ext)
    if pairs:
        lines.append(f"\n**All Contrastive Pairs ({len(pairs)}):**")
        for i, ex in enumerate(pairs):
            lines.append(f"\n*Pair {i}:*")
            lines.append(f"- Prompt: {ex['prompt']}")
            lines.append(f"- Correct: {ex['correct']}")
            lines.append(f"- Incorrect: {ex['incorrect']}")

    return "\n".join(lines)


def format_activations_summary(task_name: str, model_name: str, layer) -> str:
    """Inspect the raw per-token activation shards for a benchmark+model:
    which prompt formats + layers exist, and for the selected layer (or the
    first available) the per-token tensor shapes [seq_len, hidden] + L2 norms,
    reading the exact raw_activations the extraction wrote."""
    if " (" in task_name and task_name.endswith(")"):
        task_name = task_name.split(" (")[INDEX_FIRST]
    if not model_name:
        return "_Select a model to inspect activations._"
    from wisent.app.ui.tabs.debug.benchmark_artifacts import (
        summarize_raw_activations,
    )
    try:
        return summarize_raw_activations(task_name, model_name, layer)
    except Exception as exc:
        return f"Error loading activations: {exc}"


def build_activation_inspector(task_dd, model_dd, layer_dd):
    """Add a 'Check Activations' section reusing the tab's benchmark/model/
    layer selectors: button -> shapes, per-pair norms, HF provenance; plus a
    per-token view (one row per token, pos/neg L2 norm) for a chosen pair."""
    from wisent.app.ui.tabs.debug.benchmark_tokens import inspect_pair_tokens
    gr.Markdown("---\n**Check Activations** (shapes, per-pair norms, HF source)")
    btn = gr.Button("Inspect Activations", variant="secondary")
    out = gr.Markdown(value="Select Benchmark + Model + Layer above, then click.")
    btn.click(fn=format_activations_summary,
              inputs=[task_dd, model_dd, layer_dd], outputs=[out])
    gr.Markdown("**Per-token view** — one row per token (pos/neg L2 norm)")
    with gr.Row():
        pf_dd = gr.Dropdown(label="Prompt format",
                            choices=["chat", "mc_balanced", "role_play"],
                            value="chat", interactive=True)
        pid_num = gr.Number(label="Pair id (pid)", value=0,
                            precision=INDEX_FIRST)
    tok_btn = gr.Button("Show Token-by-Token", variant="secondary")
    tok_out = gr.Markdown(value="Pick prompt format + pair id, then click.")
    tok_btn.click(fn=inspect_pair_tokens,
                  inputs=[task_dd, model_dd, pf_dd, layer_dd, pid_num],
                  outputs=[tok_out])
    from wisent.app.ui.tabs.debug.benchmark_tokens import summarize_strategies
    strat_btn = gr.Button("Show All 7 Strategies (aggregations)", variant="secondary")
    strat_out = gr.Markdown(value="Per-pair vectors under all 7 strategies (norms).")
    strat_btn.click(fn=summarize_strategies, inputs=[task_dd, model_dd, layer_dd, pid_num], outputs=[strat_out])
    from wisent.app.ui.tabs.debug.benchmark_legacy import (
        list_inventory, inspect_inventory,
    )
    gr.Markdown("---\n**Activation Inventory** — every (model, task) on HF "
                "(`[raw]`=per-token, `[legacy]`=aggregated, `[both]`)")
    inv_dd = gr.Dropdown(label="All activations", choices=[], value=None,
                         interactive=True, allow_custom_value=True)
    with gr.Row():
        inv_load = gr.Button("Load inventory", variant="secondary")
        inv_btn = gr.Button("Inspect Selected", variant="secondary")
    inv_out = gr.Markdown(value="Load inventory, pick an entry (uses Layer above), then Inspect Selected.")
    inv_load.click(fn=lambda: gr.update(choices=list_inventory()), outputs=[inv_dd])
    inv_btn.click(fn=inspect_inventory, inputs=[inv_dd, layer_dd], outputs=[inv_out])
