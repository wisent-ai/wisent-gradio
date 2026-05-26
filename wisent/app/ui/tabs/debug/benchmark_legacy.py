"""Inventory + aggregated-store inspection + Macro Check for Benchmark Debug.

Reads the actual HF repo trees (activations/ and raw_activations/) via
huggingface_hub.list_repo_tree. list_inventory() unions both trees;
summarize_legacy_activations() reports the pre-reduced shards; coverage_matrix
/ benchmark_sizes / missing_matrix back the Macro Check sub-tab.
"""
from .benchmark_artifacts import (
    HF_REPO_ID, HF_REPO_TYPE, safe_name_to_model,
    _get_hf_token, _hf_hub_download, _load_safetensors_file,
    summarize_raw_activations,
)

_INVENTORY_CACHE: dict = {}


def _subdirs(path: str) -> list:
    """Immediate child directory basenames under a repo path."""
    from huggingface_hub import HfApi
    api = HfApi(token=_get_hf_token())
    out = []
    try:
        for e in api.list_repo_tree(HF_REPO_ID, path_in_repo=path,
                                    repo_type=HF_REPO_TYPE, recursive=False):
            p = getattr(e, "path", "")
            if getattr(e, "tree_id", None) is not None or "." not in p.rsplit("/", 1)[-1]:
                out.append(p.rsplit("/", 1)[-1])
    except Exception:
        pass
    return out


def _layer_files(path: str) -> list:
    """layer_<N>.safetensors integers under a strategy path."""
    from huggingface_hub import HfApi
    api = HfApi(token=_get_hf_token())
    layers = []
    try:
        for e in api.list_repo_tree(HF_REPO_ID, path_in_repo=path,
                                    repo_type=HF_REPO_TYPE, recursive=False):
            name = getattr(e, "path", "").rsplit("/", 1)[-1]
            if name.startswith("layer_") and name.endswith(".safetensors"):
                try:
                    layers.append(int(name[len("layer_"):-len(".safetensors")]))
                except ValueError:
                    pass
    except Exception:
        pass
    return sorted(layers)


def list_inventory() -> list:
    """Every (model, task) with activations on HF: legacy aggregated (real
    activations/ tree) + raw per-token (raw_activations/ tree). Sorted choice
    strings tagged '[raw]', '[legacy]' or '[both]'. Cached per process."""
    if "list" in _INVENTORY_CACHE:
        return _INVENTORY_CACHE["list"]
    legacy = set()
    for m in _subdirs("activations"):
        for t in _subdirs(f"activations/{m}"):
            legacy.add(f"{m}/{t}")
    raw = set()
    from huggingface_hub import HfApi
    api = HfApi(token=_get_hf_token())
    try:
        for e in api.list_repo_tree(HF_REPO_ID, path_in_repo="raw_activations",
                                    repo_type=HF_REPO_TYPE, recursive=True):
            p = getattr(e, "path", "")
            if p.endswith("_chunk_0.safetensors"):
                parts = p[len("raw_activations/"):].split("/")
                if len(parts) >= 4:
                    raw.add(f"{parts[0]}/{'/'.join(parts[1:-2])}")
    except Exception:
        pass
    choices = []
    for mt in sorted(legacy | raw):
        store = ("both" if mt in legacy and mt in raw
                 else "raw" if mt in raw else "activations")
        choices.append(f"[{store}] {mt}")
    _INVENTORY_CACHE["list"] = choices
    return choices


def summarize_legacy_activations(safe_model: str, task: str, layer) -> str:
    """Legacy aggregated activations for a (model, task): one pre-reduced
    [num_pairs, hidden] vector per pair per strategy. Reports, per strategy,
    pair count + mean ||pos||, mean ||neg||, mean ||pos-neg|| at a layer."""
    base = f"activations/{safe_model}/{task}"
    strategies = sorted(_subdirs(base))
    if not strategies:
        return f"_No aggregated activations on HF for `{safe_model}` / `{task}`._"
    layers = _layer_files(f"{base}/{strategies[0]}")
    if not layers:
        return f"_No layer files under `{base}/{strategies[0]}`._"
    sel = (int(layer) if layer not in (None, "") and int(layer) in layers
           else layers[0])
    lines = [
        f"**Aggregated activations (`activations/`) — `{safe_model}` / "
        f"`{task}` / layer {sel}:**",
        f"- HF: `activations/{safe_model}/{task}/<strategy>/layer_{sel}"
        ".safetensors` (pre-reduced [num_pairs, hidden] per strategy)",
        f"- strategies: {len(strategies)}; layers: {layers[0]}..{layers[-1]}",
        "",
        "| strategy | pairs | mean \\|\\|pos\\|\\| | mean \\|\\|neg\\|\\| | "
        "mean \\|\\|pos-neg\\|\\| |",
        "|:--|--:|--:|--:|--:|",
    ]
    for strat in strategies:
        path = f"{base}/{strat}/layer_{sel}.safetensors"
        try:
            t, _meta = _load_safetensors_file(_hf_hub_download(path))
        except Exception as exc:
            lines.append(f"| {strat} | _err: {str(exc)[:40]}_ |  |  |  |")
            continue
        pos, neg = t.get("pos_activations"), t.get("neg_activations")
        if pos is None or neg is None:
            lines.append(f"| {strat} | _no tensors_ |  |  |  |")
            continue
        pf, nf = pos.float(), neg.float()
        lines.append(
            f"| {strat} | {pos.shape[0]} | {float(pf.norm(dim=1).mean()):.3f} "
            f"| {float(nf.norm(dim=1).mean()):.3f} | "
            f"{float((pf - nf).norm(dim=1).mean()):.3f} |")
    return "\n".join(lines)


def inspect_inventory(choice: str, layer) -> str:
    """Dispatch an inventory selection: raw/both -> per-token raw summary;
    legacy -> aggregated per-strategy summary."""
    if not choice:
        return "_Select an inventory entry, then click._"
    store, _, mt = choice.partition("] ")
    store = store.lstrip("[")
    safe_model, _, task = mt.partition("/")
    if store in ("raw", "both"):
        return summarize_raw_activations(task, safe_name_to_model(safe_model), layer)
    return summarize_legacy_activations(safe_model, task, layer)


def coverage_matrix():
    """Canonical benchmarks x models coverage. Subtasks roll up to their
    top-level benchmark (benchmark_tags.json) so rows = the ~380 top-level
    benchmarks, not the thousands of subtask names. Two ✓/— columns per
    model: `raw_activations/` and `activations/`."""
    from .benchmark_artifacts import canonical_benchmarks, rollup_to_canonical
    inv = list_inventory()
    canon = sorted(set(canonical_benchmarks()))
    canon_set = set(canon)
    cov: dict = {}
    models: set = set()
    for c in inv:
        store, _, mt = c.partition("] ")
        store = store.lstrip("[")
        sm, _, task = mt.partition("/")
        models.add(sm)
        s = rollup_to_canonical(task, canon_set)
        if not s:
            continue
        d = cov.setdefault((s, sm), [False, False])
        d[0] = d[0] or store in ("raw", "both")
        d[1] = d[1] or store in ("activations", "both")
    models = sorted(models)
    headers = ["benchmark"]
    for m in models:
        nm = safe_name_to_model(m)
        headers += [f"{nm} · raw", f"{nm} · agg"]
    rows = []
    for cb in canon:
        row = [cb]
        for m in models:
            d = cov.get((cb, m), (False, False))
            row += ["✓" if d[0] else "—", "✓" if d[1] else "—"]
        rows.append(row)
    raw_tot = sum(1 for d in cov.values() if d[0])
    agg_tot = sum(1 for d in cov.values() if d[1])
    summary = (
        f"**Coverage:** {len(canon)} canonical benchmarks (subtasks rolled "
        f"up) x {len(models)} models. (benchmark, model) cells with each "
        f"store: raw_activations={raw_tot}, activations={agg_tot}.")
    return headers, rows, summary


_SIZES_CACHE: dict = {}


def benchmark_sizes(model_safe: str = "meta-llama__Llama-3.2-1B-Instruct"):
    """Per TOP-LEVEL benchmark: sum of original (pair_texts_total_entries) +
    extracted (supabase_total_pairs) over the benchmark + all its subtasks.
    Reads coverage-JSON headers in parallel; cached per process."""
    if model_safe in _SIZES_CACHE:
        return _SIZES_CACHE[model_safe]
    import re
    import socket
    import concurrent.futures
    import requests
    from huggingface_hub import HfApi
    socket.setdefaulttimeout(25)
    sess = requests.Session()
    sess.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=64, pool_maxsize=64))
    tok = _get_hf_token()
    api = HfApi(token=tok)
    base = f"coverage/{model_safe}"
    tasks = []
    try:
        for e in api.list_repo_tree(HF_REPO_ID, path_in_repo=base,
                                    repo_type=HF_REPO_TYPE, recursive=False):
            p = getattr(e, "path", "")
            if p.endswith(".json"):
                tasks.append(p[len(base) + 1:-5])
    except Exception:
        pass
    resolve = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{base}"
    hdr = {"Range": "bytes=0-600"}
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    pat_o = re.compile(r'"pair_texts_total_entries":\s*(\d+)')
    pat_x = re.compile(r'"supabase_total_pairs":\s*(\d+)')

    def _one(task):
        try:
            txt = sess.get(f"{resolve}/{task}.json", headers=hdr).text
        except Exception:
            return (task, None, None)
        o = pat_o.search(txt)
        x = pat_x.search(txt)
        return (task, int(o.group(1)) if o else None,
                int(x.group(1)) if x else None)

    from .benchmark_artifacts import canonical_benchmarks, rollup_to_canonical
    canon = set(canonical_benchmarks())
    agg: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as pool:
        for task, o, x in pool.map(_one, tasks):
            cb = rollup_to_canonical(task, canon)
            if not cb:
                continue
            a = agg.setdefault(cb, [0, 0, 0])
            a[0] += o or 0; a[1] += x or 0; a[2] += 1
    rows = [[cb, agg[cb][0], agg[cb][1], agg[cb][2]]
            for cb in sorted(agg, key=lambda c: agg[c][0], reverse=True)]
    tot_o = sum(a[0] for a in agg.values())
    tot_x = sum(a[1] for a in agg.values())
    mx = rows[0] if rows else ["", 0, 0, 0]
    summary = (f"**Benchmark sizes (top-level; subtasks of any depth summed)** "
               f"— {len(rows)} benchmarks. Max original {mx[1]:,} (`{mx[0]}`); "
               f"totals: original {tot_o:,}, extracted {tot_x:,}.")
    _SIZES_CACHE[model_safe] = (
        ["benchmark", "original (sum)", "extracted (sum)", "#tasks"],
        rows, summary)
    return _SIZES_CACHE[model_safe]


def build_macro_check():
    """Macro Check sub-tab: a button that loads the full coverage matrix."""
    import gradio as gr
    gr.Markdown("**Macro Check** — activation coverage. Each model has two "
                "independent ✓/— columns: `· raw` = `raw_activations/` "
                "(per-token), `· agg` = `activations/` (aggregated). A "
                "benchmark in both stores shows ✓ in both.")
    btn = gr.Button("Load coverage matrix", variant="primary")
    summary = gr.Markdown(value="Click to load (enumerates the full HF "
                          "inventory; cached after first load).")
    df = gr.Dataframe(headers=["benchmark"], interactive=False, wrap=True)

    def _load():
        headers, rows, summ = coverage_matrix()
        return summ, gr.update(headers=headers, value=rows)

    btn.click(fn=_load, outputs=[summary, df])

    gr.Markdown("---\n**Benchmark sizes** — original pairs "
                "(`pair_texts_total_entries`) vs extracted "
                "(`supabase_total_pairs`), sorted by original, largest first")
    sz_btn = gr.Button("Load benchmark sizes", variant="secondary")
    sz_summary = gr.Markdown(value="Click to load (per pair-set; "
                             "model-independent; cached after first load).")
    sz_df = gr.Dataframe(
        headers=["benchmark", "original_pairs", "extracted_pairs"],
        interactive=False, wrap=True)

    def _load_sizes():
        h, r, s = benchmark_sizes()
        return s, gr.update(headers=h, value=r)

    sz_btn.click(fn=_load_sizes, outputs=[sz_summary, sz_df])

    from .benchmark_artifacts import missing_matrix
    gr.Markdown("---\n**What's missing** — canonical benchmarks not covered "
                "per model (`✓`=has activations, `—`=missing; most-missing "
                "first)")
    miss_btn = gr.Button("Load missing", variant="secondary")
    miss_summary = gr.Markdown(value="Click to load.")
    miss_df = gr.Dataframe(headers=["benchmark"], interactive=False, wrap=True)

    def _load_missing():
        h, r, s = missing_matrix(list_inventory())
        return s, gr.update(headers=h, value=r)

    miss_btn.click(fn=_load_missing, outputs=[miss_summary, miss_df])
