"""Missing artifact-store coverage by canonical benchmark."""

from . import canonical_benchmarks, rollup_to_canonical


def missing_matrix(inventory: list):
    """Return per-benchmark raw and aggregated store gaps."""
    canon = set(canonical_benchmarks())
    cov_raw: dict = {}
    cov_agg: dict = {}
    models: set = set()
    for choice in inventory:
        store, _, model_task = choice.partition("] ")
        store = store.lstrip("[")
        safe_model, _, task = model_task.partition("/")
        models.add(safe_model)
        benchmark = rollup_to_canonical(task, canon)
        if not benchmark:
            continue
        if store in ("raw", "both"):
            cov_raw.setdefault(benchmark, set()).add(safe_model)
        if store in ("activations", "both"):
            cov_agg.setdefault(benchmark, set()).add(safe_model)
    models = sorted(models)
    model_count = len(models)
    model_set = set(models)
    rows = []
    for benchmark in sorted(canon):
        raw_missing = model_count - len(cov_raw.get(benchmark, set()) & model_set)
        aggregate_missing = model_count - len(cov_agg.get(benchmark, set()) & model_set)
        rows.append([benchmark, raw_missing, aggregate_missing, raw_missing + aggregate_missing])
    rows.sort(key=lambda row: (row[3], row[2]), reverse=True)
    headers = [
        "benchmark",
        f"raw missing /{model_count}",
        f"agg missing /{model_count}",
        "#missing cells",
    ]
    raw_cells = sum(row[1] for row in rows)
    aggregate_cells = sum(row[2] for row in rows)
    no_aggregate = [row[0] for row in rows if row[2] == model_count]
    no_raw = sum(1 for row in rows if row[1] == model_count)
    summary = (
        f"**Missing by store** — {len(canon)} benchmarks x {model_count} models. "
        f"raw_activations missing: {raw_cells} cells; activations (agg) "
        f"missing: {aggregate_cells} cells. No agg for ANY model ({len(no_aggregate)}): "
        f"{', '.join(no_aggregate) or 'none'}. No raw for ANY model: {no_raw} "
        "benchmarks."
    )
    return headers, rows, summary
