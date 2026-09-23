"""Loaders for the canonical published extraction artifacts on the
wisent-ai/activations HF dataset (pair_texts + raw_activations), for the
Benchmark Debug inspectors. The pid keys in pair_texts are the same ids
used in the pos_<pid>/neg_<pid> activation tensors, so the two map cleanly.
"""
import json

from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
    HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name, safe_name_to_model,
)

PAIR_CHUNK_SIZE = 25  # matches the raw extractor's per-chunk pair count
# The artifact summary prints the first few pairs of a chunk.
SHOWN_PAIRS = 6
from wisent.core.reading.modules.utilities.data.sources.hf.hf_loaders import (
    _hf_hub_download, _load_safetensors_file, _get_hf_token,
)
from wisent.app import failure
from wisent.app.failure import ArtifactUnavailable


def _reject_absence(failure_point: str, error, reason: str) -> None:
    """Let a real "not there" through; refuse to call anything else empty.

    A 404 from the Hub is an answer: nothing was published. A timeout, a 5xx, a
    missing token or dead DNS is not an answer, and reporting it as an empty
    inventory is how an outage ends up looking like a benchmark nobody ever
    extracted. Those raise `ArtifactUnavailable` instead.
    """
    if error is None:
        return
    classification = failure.report(
        failure_point,
        service=failure.SERVICE_HUGGINGFACE,
        error=error,
        reason=reason,
    )
    if classification.code != failure.CODE_NOT_FOUND:
        raise ArtifactUnavailable(classification, error) from error


def _category(task_name: str):
    """Benchmark category (e.g. 'commonsense' for hellaswag) used to build
    the pair_texts/<category>/<task>.json path. None if uncategorized."""
    from wisent.core.utils.services.benchmarks.registry.benchmark_registry import (
        get_working_benchmarks_with_categories,
    )
    return get_working_benchmarks_with_categories().get(task_name)


def load_pair_texts(task_name: str):
    """Load pair_texts/<category>/<task>.json from HF.

    Returns (pairs, hf_path) where pairs maps str pid -> {prompt, positive,
    negative}. The package's pair_texts_hf_path bakes in a stale flat
    pair_texts/<task>.json path, so we resolve the category-nested path here
    (falling back to the flat path for any task that was published flat).
    Raises FileNotFoundError if neither path exists.
    """
    cat = _category(task_name)
    candidates = []
    if cat:
        candidates.append(f"pair_texts/{cat}/{task_name}.json")
    candidates.append(f"pair_texts/{task_name}.json")
    last_exc = None
    for path in candidates:
        try:
            local = _hf_hub_download(path)
        except Exception as exc:
            last_exc = exc
            continue
        with open(local, "r") as f:
            return json.load(f), path
    _reject_absence(
        "hf.pair_texts", last_exc, f"pair_texts for {task_name} (tried {candidates})"
    )
    raise FileNotFoundError(
        f"no pair_texts published on HF for {task_name} (tried {candidates})"
    )


def _normalize_model(task_name: str, model_name: str) -> str:
    """Reconcile the tab's model-dropdown value with the raw_activations
    layout. The dropdown is populated by discover_available_models, which
    reads a stale index.json where model ids were munged as
    <model>__<category> (so a selected value arrives as e.g.
    'meta-llama/Llama-3.2-1B-Instruct/commonsense'). raw_activations stores
    the clean model id, so strip a trailing path segment that is exactly
    this task's category. Deterministic: only strips the category, nothing
    else."""
    cat = _category(task_name)
    if cat and model_name.endswith("/" + cat):
        return model_name[: -(len(cat) + 1)]
    return model_name


def discover_raw_models(task_name: str) -> list:
    """Clean model ids that actually have raw_activations for this task.

    Lists raw_activations/<safe_model>/ and keeps the models whose <task>
    subdir exists, returning the clean (un-munged) model ids. This is the
    canonical source for the tab's model dropdown — unlike the legacy
    index.json discovery, which munges model ids as <model>__<category> and
    points at the old activations/ layout.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=_get_hf_token())
    try:
        top = api.list_repo_tree(
            HF_REPO_ID, path_in_repo="raw_activations",
            repo_type=HF_REPO_TYPE, recursive=False,
        )
    except Exception as exc:
        _reject_absence(
            "hf.raw_models", exc, f"listing raw_activations for {task_name}"
        )
        return []
    models = []
    for e in top:
        path = getattr(e, "path", "")
        if not path.startswith("raw_activations/"):
            continue
        safe = path.split("/", 1)[1]
        try:
            sub = api.list_repo_tree(
                HF_REPO_ID, path_in_repo=f"{path}/{task_name}",
                repo_type=HF_REPO_TYPE, recursive=False,
            )
            if any(True for _ in sub):
                models.append(safe_name_to_model(safe))
        except Exception as exc:
            # A model that never published this task answers 404 and simply is
            # not a candidate. Anything else means the list below is short for
            # a reason the operator cannot see — and a short dropdown looks
            # exactly like a correct one.
            _reject_absence("hf.raw_models", exc, f"probing {path}/{task_name}")
            continue
    return sorted(models)


def _list_raw_tree(task_name: str, model_name: str) -> dict:
    """Discover raw activation shards under raw_activations/<safe>/<task>/.

    Returns {prompt_format: {"layers": sorted[int],
    "chunks": {layer: sorted[int]}, "complete": bool}}.
    """
    from huggingface_hub import HfApi
    safe = model_to_safe_name(model_name)
    prefix = f"raw_activations/{safe}/{task_name}"
    api = HfApi(token=_get_hf_token())
    try:
        entries = api.list_repo_tree(
            HF_REPO_ID, path_in_repo=prefix, repo_type=HF_REPO_TYPE,
            recursive=True,
        )
    except Exception as exc:
        _reject_absence("hf.raw_tree", exc, f"listing {prefix}")
        return {}
    raw: dict = {}
    for e in entries:
        path = getattr(e, "path", "")
        if not path or not path.startswith(prefix + "/"):
            continue
        rel = path[len(prefix) + 1:]
        parts = rel.split("/")
        if len(parts) != 2:
            continue
        pf, fname = parts
        slot = raw.setdefault(pf, {"layers": set(), "chunks": {}, "complete": False})
        if fname == "_complete.json":
            slot["complete"] = True
            continue
        if not (fname.startswith("layer_") and fname.endswith(".safetensors")):
            continue
        try:
            body = fname[len("layer_"):-len(".safetensors")]
            l_str, c_str = body.split("_chunk_")
            layer, chunk = int(l_str), int(c_str)
        except Exception:
            continue
        slot["layers"].add(layer)
        slot["chunks"].setdefault(layer, set()).add(chunk)
    return {
        pf: {
            "layers": sorted(slot["layers"]),
            "chunks": {k: sorted(v) for k, v in slot["chunks"].items()},
            "complete": slot["complete"],
        }
        for pf, slot in raw.items()
    }


def summarize_raw_activations(task_name: str, model_name: str, layer=None) -> str:
    """Markdown summary of the raw per-token activation shards for a
    benchmark+model: which prompt formats + layers exist, and for the
    selected layer (or first available) the per-token tensor shapes
    [seq_len, hidden] + L2 norms for the first few pairs, verifying each pid
    maps to a pair_texts entry (surfacing missing mappings, not fabricating).
    """
    model_name = _normalize_model(task_name, model_name)
    safe = model_to_safe_name(model_name)
    try:
        tree = _list_raw_tree(task_name, model_name)
    except ArtifactUnavailable as exc:
        return failure.as_markdown(
            exc.classification, exc.cause, heading="Raw activations unknown"
        )
    if not tree:
        return (f"_No raw activations on HF for `{task_name}` / `{model_name}` "
                f"(looked under `raw_activations/{safe}/{task_name}/`)._")
    try:
        pairs, pt_path = load_pair_texts(task_name)
    except ArtifactUnavailable as exc:
        pairs, pt_path = {}, f"(unknown — {failure.summary(exc.classification)})"
    except FileNotFoundError:
        pairs, pt_path = {}, "(none published)"
    lines = [
        f"**Raw activations — `{task_name}` / `{model_name}`:**",
        f"- HF: `raw_activations/{safe}/{task_name}/<prompt_format>/"
        f"layer_<L>_chunk_<C>.safetensors`",
        f"- pair_texts for pid->text mapping: `{pt_path}`",
    ]
    for pf in sorted(tree):
        slot = tree[pf]
        layers = slot["layers"]
        if not layers:
            lines.append(f"\n### {pf}: no layer shards")
            continue
        sel = (int(layer) if layer not in (None, "") and int(layer) in layers
               else layers[0])
        nchunks = len(slot["chunks"].get(sel, []))
        lines.append(
            f"\n### {pf} — layers {layers[0]}..{layers[-1]} "
            f"({len(layers)} layers), complete={slot['complete']}; "
            f"inspecting layer {sel} (chunk 0 of {nchunks})"
        )
        path = (f"raw_activations/{safe}/{task_name}/{pf}/"
                f"layer_{sel}_chunk_0.safetensors")
        try:
            local = _hf_hub_download(path)
            tensors, meta = _load_safetensors_file(local)
        except Exception as exc:
            classification = failure.report(
                "hf.raw_shard",
                service=failure.SERVICE_HUGGINGFACE,
                error=exc,
                reason=path,
            )
            lines.append(f"  - {failure.summary(classification)}")
            continue
        pids = json.loads(meta.get("pair_ids", "[]"))
        lines.append(f"  - pairs in chunk 0: {len(pids)}")
        for pid in pids[:SHOWN_PAIRS]:
            pos = tensors.get(f"pos_{pid}")
            neg = tensors.get(f"neg_{pid}")
            if pos is None or neg is None:
                lines.append(f"  - pid {pid}: MISSING tensor in shard")
                continue
            mapped = str(pid) in pairs
            lines.append(
                f"  - pid {pid}: pos {tuple(pos.shape)} "
                f"||{float(pos.float().norm()):.1f}||  "
                f"neg {tuple(neg.shape)} ||{float(neg.float().norm()):.1f}||  "
                f"pair_texts={'yes' if mapped else 'MISSING'}"
            )
    return "\n".join(lines)


