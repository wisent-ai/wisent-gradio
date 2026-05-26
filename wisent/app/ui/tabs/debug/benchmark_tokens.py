"""Per-token activation view with verified token-string labels.

Reconstructs the exact ordered contrastive pairs the raw extractor consumed
— build_contrastive_pairs(task, limit, train_ratio=0.5), the deterministic
generator behind `wisent generate-pairs-from-task` that the extractor ran —
so activation shard index i maps to source pair i. Rebuilds each arm's full
text via build_extraction_texts (same as collect_single_raw), tokenizes
identically, and labels each tensor row with its token ONLY when the
reconstructed token count matches the shard's seq_len, so labels are
verified rather than fabricated.
"""
from .benchmark_artifacts import (
    PAIR_CHUNK_SIZE, _hf_hub_download, _list_raw_tree,
    _load_safetensors_file, _normalize_model, model_to_safe_name,
)

_PAIRS_CACHE: dict = {}
_TOK_CACHE: dict = {}
_PF2STRAT = {"chat": "CHAT_LAST", "mc_balanced": "MC_BALANCED",
             "role_play": "ROLE_PLAY"}


def source_pairs(task_name: str, limit: int):
    """The exact ordered pairs the extractor consumed (cached per task/limit)."""
    key = (task_name, limit)
    if key not in _PAIRS_CACHE:
        from wisent.extractors.lm_eval.lm_task_pairs_generation import (
            build_contrastive_pairs,
        )
        pairs = build_contrastive_pairs(task_name, limit=limit, train_ratio=0.5)
        out = []
        for p in pairs:
            d = p.to_dict() if hasattr(p, "to_dict") else p
            po = ((d.get("positive_response") or {}).get("model_response")
                  if "positive_response" in d else d.get("positive"))
            ne = ((d.get("negative_response") or {}).get("model_response")
                  if "negative_response" in d else d.get("negative"))
            out.append((d.get("prompt", ""), po or "", ne or ""))
        _PAIRS_CACHE[key] = out
    return _PAIRS_CACHE[key]


def _tokenizer(model_name: str):
    if model_name not in _TOK_CACHE:
        from transformers import AutoTokenizer
        _TOK_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOK_CACHE[model_name]


def _toks(tok, text: str):
    ids = tok(text, add_special_tokens=False, truncation=True,
              max_length=tok.model_max_length)["input_ids"]
    return tok.convert_ids_to_tokens(ids)


def _esc(t: str) -> str:
    return t.replace("|", "\\|").replace("`", "ʼ")


def _labels(task_name, model_name, pf, pid, limit, pos_len, neg_len):
    """(toks_pos, toks_neg, note). Empty token lists if reconstruction can't
    be verified against the shard seq_len — never guess an alignment."""
    from wisent.core.primitives.model_interface.core.activations import (
        ExtractionStrategy, build_extraction_texts,
    )
    strat = getattr(ExtractionStrategy, _PF2STRAT[pf])
    pairs = source_pairs(task_name, limit)
    if pid >= len(pairs):
        return [], [], f"pid {pid} >= reconstructed pairs ({len(pairs)})"
    prompt, pos, neg = pairs[pid]
    needs = strat in (ExtractionStrategy.MC_BALANCED,
                      ExtractionStrategy.MC_COMPLETION)
    tok = _tokenizer(model_name)
    fp = build_extraction_texts(strat, prompt, pos, tok,
                                other_response=(neg if needs else None),
                                is_positive=True)[0]
    fn = build_extraction_texts(strat, prompt, neg, tok,
                                other_response=(pos if needs else None),
                                is_positive=False)[0]
    tp, tn = _toks(tok, fp), _toks(tok, fn)
    if len(tp) == pos_len and len(tn) == neg_len:
        return tp, tn, (
            f"labels verified — reconstructed from build_contrastive_pairs"
            f"('{task_name}', limit={limit})[{pid}]; token counts match "
            f"shard seq_len")
    return [], [], (
        f"labels unverified — reconstructed pair {pid} tokenizes to "
        f"pos={len(tp)}/neg={len(tn)} vs shard pos={pos_len}/neg={neg_len}; "
        f"showing positions only")


def inspect_pair_tokens(task_name, model_name, prompt_format, layer, pid,
                        max_tokens: int = 400) -> str:
    """One row per token for a chosen pair: token string (verified) + the
    positive- and negative-arm L2 norm at that position."""
    if " (" in str(task_name) and str(task_name).endswith(")"):
        task_name = task_name.split(" (")[0]
    model_name = _normalize_model(task_name, model_name)
    safe = model_to_safe_name(model_name)
    tree = _list_raw_tree(task_name, model_name)
    if not tree:
        return f"_No raw activations on HF for `{task_name}` / `{model_name}`._"
    pf = prompt_format if prompt_format in tree else sorted(tree)[0]
    layers = tree[pf]["layers"]
    if not layers:
        return f"_No layer shards for prompt format `{pf}`._"
    sel = (int(layer) if layer not in (None, "") and int(layer) in layers
           else layers[0])
    pid = int(pid)
    chunks = tree[pf]["chunks"].get(sel, [])
    chunk = pid // PAIR_CHUNK_SIZE
    path = (f"raw_activations/{safe}/{task_name}/{pf}/"
            f"layer_{sel}_chunk_{chunk}.safetensors")
    try:
        local = _hf_hub_download(path)
        tensors, _meta = _load_safetensors_file(local)
    except Exception as exc:
        return f"Error loading `{path}`: {exc}"
    pos = tensors.get(f"pos_{pid}")
    neg = tensors.get(f"neg_{pid}")
    if pos is None or neg is None:
        return f"_pid {pid} not in chunk {chunk} for `{pf}` layer {sel}._"
    pn, nn = pos.float().norm(dim=1), neg.float().norm(dim=1)
    limit = (max(chunks) + 1) * PAIR_CHUNK_SIZE if chunks else PAIR_CHUNK_SIZE
    try:
        tp, tn, note = _labels(task_name, model_name, pf, pid, limit,
                               pos.shape[0], neg.shape[0])
    except Exception as exc:
        tp, tn, note = [], [], f"labels unavailable (reconstruction error: {exc})"
    lines = [
        f"**Per-token activations — `{task_name}` / `{model_name}` / "
        f"`{pf}` / layer {sel} / pair {pid}:**",
        f"- positive `{tuple(pos.shape)}`  negative `{tuple(neg.shape)}` "
        f"(rows = tokens, {pos.shape[-1]} hidden dims)",
        f"- {note}",
        "",
        "| token # | pos token | \\|\\|pos\\|\\| | neg token | \\|\\|neg\\|\\| |",
        "|--:|:--|--:|:--|--:|",
    ]
    rows = max(pos.shape[0], neg.shape[0])
    for i in range(min(int(max_tokens), rows)):
        pt = f"`{_esc(tp[i])}`" if i < len(tp) else ""
        nt = f"`{_esc(tn[i])}`" if i < len(tn) else ""
        pv = f"{float(pn[i]):.3f}" if i < pos.shape[0] else ""
        nv = f"{float(nn[i]):.3f}" if i < neg.shape[0] else ""
        lines.append(f"| {i} | {pt} | {pv} | {nt} | {nv} |")
    if rows > int(max_tokens):
        lines.append(f"\n_(showing first {max_tokens} of {rows} tokens)_")
    return "\n".join(lines)


def _pair_tensor(safe, task_name, pf, layer, pid):
    chunk = pid // PAIR_CHUNK_SIZE
    path = (f"raw_activations/{safe}/{task_name}/{pf}/"
            f"layer_{layer}_chunk_{chunk}.safetensors")
    tensors, _meta = _load_safetensors_file(_hf_hub_download(path))
    return tensors.get(f"pos_{pid}"), tensors.get(f"neg_{pid}")


def summarize_strategies(task_name, model_name, layer, pid) -> str:
    """For one pair, derive the aggregated vector under all 7 extraction
    strategies from the raw per-token shards (5 chat_* from the chat pass,
    mc_balanced + role_play from their passes) via the package's exact
    extract_activation reductions, and report ||pos||, ||neg|| and the
    contrastive ||pos-neg|| per strategy."""
    if " (" in str(task_name) and str(task_name).endswith(")"):
        task_name = task_name.split(" (")[0]
    model_name = _normalize_model(task_name, model_name)
    safe = model_to_safe_name(model_name)
    tree = _list_raw_tree(task_name, model_name)
    if not tree:
        return f"_No raw activations on HF for `{task_name}` / `{model_name}`._"
    anyfmt = sorted(tree)[0]
    layers = tree[anyfmt]["layers"]
    sel = (int(layer) if layer not in (None, "") and int(layer) in layers
           else layers[0])
    pid = int(pid)
    limit = (max(tree[anyfmt]["chunks"].get(sel, [0])) + 1) * PAIR_CHUNK_SIZE
    from wisent.core.primitives.model_interface.core.activations import (
        ExtractionStrategy, extract_activation,
    )
    try:
        _prompt, pos_text, neg_text = source_pairs(task_name, limit)[pid]
    except Exception as exc:
        return f"_Could not reconstruct pair {pid}: {exc}_"
    tok = _tokenizer(model_name)
    plan = [
        ("chat_last", "chat", ExtractionStrategy.CHAT_LAST),
        ("chat_mean", "chat", ExtractionStrategy.CHAT_MEAN),
        ("chat_first", "chat", ExtractionStrategy.CHAT_FIRST),
        ("chat_max_norm", "chat", ExtractionStrategy.CHAT_MAX_NORM),
        ("chat_weighted", "chat", ExtractionStrategy.CHAT_WEIGHTED),
        ("mc_balanced", "mc_balanced", ExtractionStrategy.MC_BALANCED),
        ("role_play", "role_play", ExtractionStrategy.ROLE_PLAY),
    ]
    lines = [
        f"**7 strategy aggregations — `{task_name}` / `{model_name}` / "
        f"layer {sel} / pair {pid}:**",
        "_Each row reduces the raw per-token tensor to one vector via "
        "wisent.extract_activation; ||pos-neg|| is the contrastive direction._",
        "",
        "| strategy | raw pass | \\|\\|pos\\|\\| | \\|\\|neg\\|\\| | \\|\\|pos-neg\\|\\| |",
        "|:--|:--|--:|--:|--:|",
    ]
    cache: dict = {}
    for name, pf, strat in plan:
        if pf not in tree:
            lines.append(f"| {name} | {pf} | _missing pass_ |  |  |")
            continue
        if pf not in cache:
            cache[pf] = _pair_tensor(safe, task_name, pf, sel, pid)
        pt, nt = cache[pf]
        if pt is None or nt is None:
            lines.append(f"| {name} | {pf} | _pid {pid} absent_ |  |  |")
            continue
        pv = extract_activation(strat, pt, pos_text, tok, 0)
        nv = extract_activation(strat, nt, neg_text, tok, 0)
        lines.append(f"| {name} | {pf} | {float(pv.norm()):.3f} | "
                     f"{float(nv.norm()):.3f} | {float((pv - nv).norm()):.3f} |")
    return "\n".join(lines)
