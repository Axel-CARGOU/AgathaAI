import atexit
import asyncio
import csv
import io
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path


frozen_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
OUT_DIR = LOG_DIR / "graphs"

LLM_REQUEST_METRICS_FILE = OUT_DIR / f"llm_request_metrics_{frozen_ts}.csv"
LLM_REQUEST_SUMMARY_FILE = OUT_DIR / f"llm_request_summary_{frozen_ts}.csv"
VLLM_ENGINE_METRICS_FILE = OUT_DIR / f"vllm_engine_metrics_{frozen_ts}.csv"
NVIDIA_SMI_METRICS_FILE = OUT_DIR / f"nvidia_smi_metrics_{frozen_ts}.csv"

LLM_REQUEST_FIELDS = [
    "request_id",
    "timestamp",
    "source",
    "mode",
    "model",
    "success",
    "is_warmup",
    "message_count",
    "image_count",
    "prompt_chars",
    "completion_chars",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "duration_s",
    "time_to_first_token_s",
    "generation_duration_s",
    "queue_prefill_s",
    "decode_s",
    "ttft_share",
    "decode_share",
    "prompt_tok_s",
    "prompt_prefill_tok_s",
    "completion_tok_s",
    "total_tok_s",
    "completion_chars_s",
    "prompt_completion_ratio",
    "completion_chars_per_token",
    "chunk_count",
    "content_chunk_count",
    "finish_reason",
    "temperature",
    "top_p",
    "max_tokens",
    "error",
]

VLLM_ENGINE_FIELDS = [
    "timestamp",
    "prompt_tok_s",
    "gen_tok_s",
    "running",
    "waiting",
    "kv_cache",
    "prefix_hit",
    "mm_cache_hit",
]

NVIDIA_SMI_FIELDS = [
    "timestamp",
    "gpu_index",
    "name",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "temperature_c",
    "power_draw_w",
    "power_limit_w",
]

SUMMARY_FIELDS = [
    "metric",
    "count",
    "mean",
    "p50",
    "p95",
    "p99",
    "min",
    "max",
]

REQUEST_GRAPH_ROW_LIMIT = 160
ENGINE_GRAPH_ROW_LIMIT = 240
METRICS_AUTO_RENDER_INTERVAL_S = 5.0

_metrics_queue = queue.Queue()
_vllm_engine_queue = queue.Queue()
_writer_state_lock = threading.Lock()
_writer_thread = None
_vllm_engine_writer_thread = None
_vllm_engine_file_lock = threading.Lock()
_request_graph_condition = threading.Condition()
_engine_graph_condition = threading.Condition()
_request_graph_worker_thread = None
_engine_graph_worker_thread = None
_request_graph_requested = False
_engine_graph_requested = False
_request_graph_render_lock = threading.Lock()
_engine_graph_render_lock = threading.Lock()
_nvidia_graph_render_lock = threading.Lock()


def record_llm_request_metric(metric):
    row = {field: metric.get(field, "") for field in LLM_REQUEST_FIELDS}
    row["timestamp"] = row["timestamp"] or datetime.now().isoformat(timespec="milliseconds")
    _ensure_metrics_writer()
    _metrics_queue.put(row)


def flush_llm_request_metrics():
    _metrics_queue.join()


def record_vllm_engine_metric_line(line):
    row = _parse_vllm_engine_metric_line(line)
    if not row:
        return False

    _ensure_vllm_engine_writer()
    _vllm_engine_queue.put(row)
    return True


def flush_vllm_engine_metrics():
    _vllm_engine_queue.join()


def schedule_metrics_graph_refresh():
    return False


def schedule_all_metrics_graph_refresh():
    return False


def schedule_llm_request_graph_refresh():
    global _request_graph_requested

    _ensure_request_graph_refresh_worker()

    with _request_graph_condition:
        _request_graph_requested = True
        _request_graph_condition.notify()


def schedule_vllm_engine_graph_refresh():
    global _engine_graph_requested

    _ensure_engine_graph_refresh_worker()

    with _engine_graph_condition:
        _engine_graph_requested = True
        _engine_graph_condition.notify()


def _ensure_request_graph_refresh_worker():
    global _request_graph_worker_thread

    with _writer_state_lock:
        if _request_graph_worker_thread is not None and _request_graph_worker_thread.is_alive():
            return

        _request_graph_worker_thread = threading.Thread(
            target=_request_graph_refresh_worker,
            name="metrics-request-graph-renderer",
            daemon=True,
        )
        _request_graph_worker_thread.start()


def _ensure_engine_graph_refresh_worker():
    global _engine_graph_worker_thread

    with _writer_state_lock:
        if _engine_graph_worker_thread is not None and _engine_graph_worker_thread.is_alive():
            return

        _engine_graph_worker_thread = threading.Thread(
            target=_engine_graph_refresh_worker,
            name="metrics-engine-graph-renderer",
            daemon=True,
        )
        _engine_graph_worker_thread.start()


def _request_graph_refresh_worker():
    global _request_graph_requested

    while True:
        with _request_graph_condition:
            while not _request_graph_requested:
                _request_graph_condition.wait()
            _request_graph_requested = False

        try:
            _run_graph_render_subprocess("request")
        except Exception as exc:
            print(f"Unable to refresh request metrics graph: {exc}")


def _engine_graph_refresh_worker():
    global _engine_graph_requested

    while True:
        with _engine_graph_condition:
            while not _engine_graph_requested:
                _engine_graph_condition.wait()
            _engine_graph_requested = False

        try:
            _run_graph_render_subprocess("engine", include_logs=False)
        except Exception as exc:
            print(f"Unable to refresh vLLM engine metrics graph: {exc}")


async def refresh_metrics_graphs():
    return current_metrics_snapshot()


def refresh_metrics_graphs_sync(*, silent_missing=False):
    return current_metrics_snapshot()


def auto_renderer_command(*, interval=METRICS_AUTO_RENDER_INTERVAL_S):
    return [
        sys.executable,
        "-m",
        "src.metrics.auto_render_daemon",
        "--frozen-ts",
        frozen_ts,
        "--base-dir",
        str(BASE_DIR),
        "--log-dir",
        str(LOG_DIR),
        "--out-dir",
        str(OUT_DIR),
        "--request-csv",
        str(LLM_REQUEST_METRICS_FILE),
        "--request-summary-csv",
        str(LLM_REQUEST_SUMMARY_FILE),
        "--engine-csv",
        str(VLLM_ENGINE_METRICS_FILE),
        "--nvidia-csv",
        str(NVIDIA_SMI_METRICS_FILE),
        "--interval",
        str(float(interval)),
    ]


def _run_graph_render_subprocess(graph, *, include_logs=True):
    if graph == "request":
        flush_llm_request_metrics()
    elif graph == "engine":
        flush_vllm_engine_metrics()
    else:
        raise ValueError(f"Unknown metrics graph: {graph}")

    command = [
        sys.executable,
        "-m",
        "src.metrics.render_worker",
        "--graph",
        graph,
        "--frozen-ts",
        frozen_ts,
        "--base-dir",
        str(BASE_DIR),
        "--log-dir",
        str(LOG_DIR),
        "--out-dir",
        str(OUT_DIR),
        "--request-csv",
        str(LLM_REQUEST_METRICS_FILE),
        "--request-summary-csv",
        str(LLM_REQUEST_SUMMARY_FILE),
        "--engine-csv",
        str(VLLM_ENGINE_METRICS_FILE),
    ]
    if include_logs:
        command.append("--include-logs")

    render_lock = _request_graph_render_lock if graph == "request" else _engine_graph_render_lock
    with render_lock:
        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(detail)

    if result.stdout.strip():
        print(result.stdout.strip())


def current_metrics_snapshot(*, request_graph=None, engine_graph=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    request_graph = request_graph or OUT_DIR / f"llm_request_metrics_{frozen_ts}.png"
    engine_graph = engine_graph or OUT_DIR / f"vllm_engine_metrics_{frozen_ts}.png"
    nvidia_graph = OUT_DIR / f"nvidia_smi_metrics_{frozen_ts}.png"

    files = []
    for path in sorted(OUT_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.suffix.lower() not in {".png", ".csv"}:
            continue

        files.append(_metrics_file_info(path))

    return {
        "frozenTimestamp": frozen_ts,
        "requestGraph": _metrics_file_info(request_graph),
        "engineGraph": _metrics_file_info(engine_graph),
        "requestCsv": _metrics_file_info(LLM_REQUEST_METRICS_FILE),
        "requestSummaryCsv": _metrics_file_info(LLM_REQUEST_SUMMARY_FILE),
        "engineCsv": _metrics_file_info(VLLM_ENGINE_METRICS_FILE),
        "nvidiaGraph": _metrics_file_info(nvidia_graph),
        "nvidiaCsv": _metrics_file_info(NVIDIA_SMI_METRICS_FILE),
        "files": files[:30],
    }


def _metrics_file_info(path):
    path = Path(path)
    exists = path.exists()
    mtime = path.stat().st_mtime if exists else None

    return {
        "name": path.name,
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "updatedAt": datetime.fromtimestamp(mtime).isoformat(timespec="seconds") if mtime else "",
        "url": f"/metrics/graphs/{path.name}?v={int(mtime or 0)}" if exists else "",
    }


def _ensure_metrics_writer():
    global _writer_thread

    with _writer_state_lock:
        if _writer_thread is not None and _writer_thread.is_alive():
            return

        _writer_thread = threading.Thread(
            target=_metrics_writer_loop,
            name="llm-metrics-writer",
            daemon=True,
        )
        _writer_thread.start()


def _metrics_writer_loop():
    while True:
        row = _metrics_queue.get()
        try:
            _write_llm_metric_row(row)
        except Exception as exc:
            print(f"Unable to write LLM request metric: {exc}")
        finally:
            _metrics_queue.task_done()


def _ensure_vllm_engine_writer():
    global _vllm_engine_writer_thread

    with _writer_state_lock:
        if _vllm_engine_writer_thread is not None and _vllm_engine_writer_thread.is_alive():
            return

        _vllm_engine_writer_thread = threading.Thread(
            target=_vllm_engine_writer_loop,
            name="vllm-engine-metrics-writer",
            daemon=True,
        )
        _vllm_engine_writer_thread.start()


def _vllm_engine_writer_loop():
    while True:
        row = _vllm_engine_queue.get()
        try:
            _write_vllm_engine_metric_row(row)
        except Exception as exc:
            print(f"Unable to write vLLM engine metric: {exc}")
        finally:
            _vllm_engine_queue.task_done()


def _write_llm_metric_row(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = (
        not LLM_REQUEST_METRICS_FILE.exists()
        or LLM_REQUEST_METRICS_FILE.stat().st_size == 0
    )

    with LLM_REQUEST_METRICS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_REQUEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_vllm_engine_metric_row(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with _vllm_engine_file_lock:
        write_header = (
            not VLLM_ENGINE_METRICS_FILE.exists()
            or VLLM_ENGINE_METRICS_FILE.stat().st_size == 0
        )

        with VLLM_ENGINE_METRICS_FILE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=VLLM_ENGINE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def get_llm_request_metrics_sync():
    flush_llm_request_metrics()
    if not LLM_REQUEST_METRICS_FILE.exists():
        return []

    with LLM_REQUEST_METRICS_FILE.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


async def get_llm_request_metrics():
    return await asyncio.to_thread(get_llm_request_metrics_sync)


def get_vllm_metrics_sync(*, include_logs=True, rewrite_csv=True):
    log_rows = []
    if include_logs:
        log_files = sorted(
            LOG_DIR.glob("*logs-Y*.log"),
            key=lambda p: p.stat().st_mtime,
        )

        for log_file in log_files:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    row = _parse_vllm_engine_metric_line(line)
                    if row:
                        log_rows.append(row)

    flush_vllm_engine_metrics()

    with _vllm_engine_file_lock:
        rows = []
        if VLLM_ENGINE_METRICS_FILE.exists():
            with VLLM_ENGINE_METRICS_FILE.open("r", newline="", encoding="utf-8") as f:
                rows.extend(
                    row
                    for row in (_coerce_vllm_engine_row(row) for row in csv.DictReader(f))
                    if row is not None
                )

        rows.extend(log_rows)

        if not rows:
            return []

        rows = _dedupe_vllm_engine_rows(rows)
        rows.sort(key=lambda row: row["timestamp"])
        if rewrite_csv:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            with VLLM_ENGINE_METRICS_FILE.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=VLLM_ENGINE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

        return rows


async def get_vllm_metrics():
    return await asyncio.to_thread(get_vllm_metrics_sync)


def record_nvidia_smi_metric():
    rows = _query_nvidia_smi_rows()
    if not rows:
        return False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_header = (
        not NVIDIA_SMI_METRICS_FILE.exists()
        or NVIDIA_SMI_METRICS_FILE.stat().st_size == 0
    )
    with NVIDIA_SMI_METRICS_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=NVIDIA_SMI_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return True


def get_nvidia_smi_metrics_sync():
    if not NVIDIA_SMI_METRICS_FILE.exists():
        return []

    with NVIDIA_SMI_METRICS_FILE.open("r", newline="", encoding="utf-8") as f:
        return [
            row
            for row in (_coerce_nvidia_smi_row(row) for row in csv.DictReader(f))
            if row is not None
        ]


def _query_nvidia_smi_rows():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = []
    for parsed in csv.reader(io.StringIO(result.stdout)):
        if len(parsed) < 9:
            continue
        rows.append({
            "timestamp": timestamp,
            "gpu_index": parsed[0].strip(),
            "name": parsed[1].strip(),
            "gpu_util_pct": _optional_float(parsed[2]),
            "mem_util_pct": _optional_float(parsed[3]),
            "mem_used_mib": _optional_float(parsed[4]),
            "mem_total_mib": _optional_float(parsed[5]),
            "temperature_c": _optional_float(parsed[6]),
            "power_draw_w": _optional_float(parsed[7]),
            "power_limit_w": _optional_float(parsed[8]),
        })
    return rows


def _coerce_nvidia_smi_row(row):
    if not row or not row.get("timestamp"):
        return None

    coerced = {
        "timestamp": row.get("timestamp", ""),
        "gpu_index": str(row.get("gpu_index", "")),
        "name": row.get("name", ""),
    }
    try:
        for field in (
            "gpu_util_pct",
            "mem_util_pct",
            "mem_used_mib",
            "mem_total_mib",
            "temperature_c",
            "power_draw_w",
            "power_limit_w",
        ):
            coerced[field] = _optional_float(row.get(field))
    except (TypeError, ValueError):
        return None
    return coerced


def _optional_float(value):
    value = str(value or "").strip()
    if not value or value.upper() in {"N/A", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
        return None
    return float(value)


def _parse_vllm_engine_metric_line(line):
    if "Avg prompt throughput" not in line:
        return None

    metrics = {
        "prompt_tok_s": _metric_float(line, r"Avg prompt throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s"),
        "gen_tok_s": _metric_float(line, r"Avg generation throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s"),
        "running": _metric_int(line, r"Running:\s*([0-9]+)\s*reqs?"),
        "waiting": _metric_int(line, r"Waiting:\s*([0-9]+)\s*reqs?"),
        "kv_cache": _metric_float(line, r"GPU KV cache usage:\s*([0-9]+(?:\.[0-9]+)?)%"),
        "prefix_hit": _metric_float(line, r"Prefix cache hit rate:\s*([0-9]+(?:\.[0-9]+)?)%"),
        "mm_cache_hit": _metric_float(line, r"MM cache hit rate:\s*([0-9]+(?:\.[0-9]+)?)%"),
    }

    required = ("prompt_tok_s", "gen_tok_s", "running", "waiting", "kv_cache", "prefix_hit")
    if any(metrics[field] is None for field in required):
        return None

    metrics["timestamp"] = _parse_vllm_metric_timestamp(line)
    return metrics


def _coerce_vllm_engine_row(row):
    if not row or not row.get("timestamp"):
        return None

    coerced = {"timestamp": row.get("timestamp", "")}
    try:
        for field in ("prompt_tok_s", "gen_tok_s", "kv_cache", "prefix_hit", "mm_cache_hit"):
            value = row.get(field)
            coerced[field] = float(value) if value not in (None, "") else None

        for field in ("running", "waiting"):
            value = row.get(field)
            coerced[field] = int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None

    required = ("prompt_tok_s", "gen_tok_s", "running", "waiting", "kv_cache", "prefix_hit")
    if any(coerced[field] is None for field in required):
        return None

    return coerced


def _dedupe_vllm_engine_rows(rows):
    deduped = {}
    for row in rows:
        key = tuple(row.get(field) for field in VLLM_ENGINE_FIELDS)
        deduped[key] = row

    return list(deduped.values())


def _parse_vllm_metric_timestamp(line):
    current_year = datetime.now().year
    match = re.search(r"\bINFO\s+(\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", line)
    if match:
        return f"{current_year}-{match.group(1)} {match.group(2)}"

    match = re.search(r"\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})", line)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    return datetime.now().isoformat(timespec="seconds")


def _metric_float(line, pattern):
    match = re.search(pattern, line)
    return float(match.group(1)) if match else None


def _metric_int(line, pattern):
    match = re.search(pattern, line)
    return int(match.group(1)) if match else None


async def make_vllm_graph_on_shutdown():
    make_vllm_graph_on_shutdown_sync()


def make_vllm_graph_on_shutdown_sync():
    request_graph = make_llm_request_graph_on_shutdown_sync()
    engine_graph = make_vllm_engine_graph_on_shutdown_sync()
    nvidia_graph = make_nvidia_smi_graph_on_shutdown_sync(silent_missing=True)

    if not request_graph and not engine_graph and not nvidia_graph:
        print("No valid LLM metrics data found to plot.")


async def make_llm_request_graph_on_shutdown():
    return make_llm_request_graph_on_shutdown_sync()


def make_llm_request_graph_on_shutdown_sync(*, silent_missing=False):
    with _request_graph_render_lock:
        return _make_llm_request_graph(silent_missing=silent_missing)


def _make_llm_request_graph(*, silent_missing=False):
    data = get_llm_request_metrics_sync()
    if not data:
        if not silent_missing:
            print("No request-level LLM metrics found to plot.")
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["success"] = _to_bool_series(df.get("success"))
    df["is_warmup"] = _to_bool_series(df.get("is_warmup"))

    numeric_cols = [
        "message_count",
        "image_count",
        "prompt_chars",
        "completion_chars",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "duration_s",
        "time_to_first_token_s",
        "generation_duration_s",
        "queue_prefill_s",
        "decode_s",
        "ttft_share",
        "decode_share",
        "prompt_tok_s",
        "prompt_prefill_tok_s",
        "completion_tok_s",
        "total_tok_s",
        "completion_chars_s",
        "prompt_completion_ratio",
        "completion_chars_per_token",
        "chunk_count",
        "content_chunk_count",
        "temperature",
        "top_p",
        "max_tokens",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["success"] & ~df["is_warmup"]].copy()
    if df.empty:
        if not silent_missing:
            print("No successful non-warmup LLM request metrics found to plot.")
        return None

    df = df.sort_values("timestamp")
    plot_df = df.tail(REQUEST_GRAPH_ROW_LIMIT).copy()
    _setup_plot_theme(sns, plt)

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))

    _lineplot_or_note(
        sns,
        axes[0, 0],
        plot_df,
        "duration_s",
        "End-to-End Latency",
        "seconds",
    )
    _lineplot_or_note(
        sns,
        axes[0, 1],
        plot_df,
        "completion_tok_s",
        "Completion Throughput",
        "tokens/s",
    )
    _lineplot_or_note(
        sns,
        axes[1, 0],
        plot_df,
        "time_to_first_token_s",
        "Time To First Token",
        "seconds",
    )
    _plot_phase_breakdown(axes[1, 1], plot_df)
    _plot_token_counts(sns, axes[2, 0], plot_df)
    _plot_prompt_size_scatter(sns, axes[2, 1], plot_df)
    _plot_latency_distribution(axes[3, 0], plot_df)
    _lineplot_or_note(
        sns,
        axes[3, 1],
        plot_df,
        "prompt_tok_s",
        "Prompt Token Rate",
        "tokens/s",
    )

    _write_request_summary(df)

    plt.tight_layout()

    graph_file = OUT_DIR / f"llm_request_metrics_{frozen_ts}.png"
    plt.savefig(graph_file)
    plt.close(fig)

    print(f"Request-level LLM graph saved to {graph_file}")
    print(f"Request-level LLM CSV saved to {LLM_REQUEST_METRICS_FILE}")
    print(f"Request-level LLM summary saved to {LLM_REQUEST_SUMMARY_FILE}")
    return graph_file


async def make_vllm_engine_graph_on_shutdown():
    return make_vllm_engine_graph_on_shutdown_sync()


def make_vllm_engine_graph_on_shutdown_sync(*, silent_missing=False, include_logs=True, rewrite_csv=True):
    with _engine_graph_render_lock:
        return _make_vllm_engine_graph(
            silent_missing=silent_missing,
            include_logs=include_logs,
            rewrite_csv=rewrite_csv,
        )


def _make_vllm_engine_graph(*, silent_missing=False, include_logs=True, rewrite_csv=True):
    data = get_vllm_metrics_sync(include_logs=include_logs, rewrite_csv=rewrite_csv)
    if not data:
        if not silent_missing:
            print("No vLLM engine metrics found to plot.")
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    for col in VLLM_ENGINE_FIELDS:
        if col != "timestamp" and col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("timestamp").tail(ENGINE_GRAPH_ROW_LIMIT).copy()
    _setup_plot_theme(sns, plt)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    _lineplot_or_note(
        sns,
        axes[0, 0],
        df,
        "running",
        "Running Requests",
        "requests",
    )
    _lineplot_or_note(
        sns,
        axes[0, 1],
        df,
        "waiting",
        "Waiting Requests",
        "requests",
    )
    _lineplot_or_note(
        sns,
        axes[1, 0],
        df,
        "kv_cache",
        "GPU KV Cache Usage",
        "%",
    )
    _plot_cache_hit_rates(sns, axes[1, 1], df)

    plt.tight_layout()

    graph_file = OUT_DIR / f"vllm_engine_metrics_{frozen_ts}.png"
    plt.savefig(graph_file)
    plt.close(fig)

    print(f"vLLM engine graph saved to {graph_file}")
    print(f"vLLM engine CSV saved to {VLLM_ENGINE_METRICS_FILE}")
    return graph_file


def make_nvidia_smi_graph_on_shutdown_sync(*, silent_missing=False):
    with _nvidia_graph_render_lock:
        return _make_nvidia_smi_graph(silent_missing=silent_missing)


def _make_nvidia_smi_graph(*, silent_missing=False):
    data = get_nvidia_smi_metrics_sync()
    if not data:
        if not silent_missing:
            print("No NVIDIA SMI metrics found to plot.")
        return None

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").tail(ENGINE_GRAPH_ROW_LIMIT).copy()
    if df.empty:
        if not silent_missing:
            print("No valid NVIDIA SMI metrics found to plot.")
        return None

    for col in NVIDIA_SMI_FIELDS:
        if col not in {"timestamp", "gpu_index", "name"} and col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _setup_plot_theme(sns, plt)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    _nvidia_lineplot_or_note(
        sns,
        axes[0, 0],
        df,
        "gpu_util_pct",
        "GPU Utilization",
        "%",
    )
    _nvidia_lineplot_or_note(
        sns,
        axes[0, 1],
        df,
        "mem_used_mib",
        "VRAM Used",
        "MiB",
    )
    _nvidia_lineplot_or_note(
        sns,
        axes[1, 0],
        df,
        "temperature_c",
        "GPU Temperature",
        "C",
    )
    _nvidia_lineplot_or_note(
        sns,
        axes[1, 1],
        df,
        "power_draw_w",
        "Power Draw",
        "W",
    )

    plt.tight_layout()

    graph_file = OUT_DIR / f"nvidia_smi_metrics_{frozen_ts}.png"
    plt.savefig(graph_file)
    plt.close(fig)

    print(f"NVIDIA SMI graph saved to {graph_file}")
    print(f"NVIDIA SMI CSV saved to {NVIDIA_SMI_METRICS_FILE}")
    return graph_file


def _setup_plot_theme(sns, plt):
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor": "#111827",
        "axes.facecolor": "#1f2937",
        "axes.edgecolor": "#374151",
        "axes.labelcolor": "#f9fafb",
        "xtick.color": "#f9fafb",
        "ytick.color": "#f9fafb",
        "text.color": "#f9fafb",
        "grid.color": "#374151",
        "legend.facecolor": "#1f2937",
        "legend.edgecolor": "#374151",
    })


def _lineplot_or_note(sns, ax, df, col, title, ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if col not in df or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("timestamp")
        return

    sns.lineplot(x="timestamp", y=col, data=df, marker="o", ax=ax)
    ax.set_xlabel("timestamp")


def _nvidia_lineplot_or_note(sns, ax, df, col, title, ylabel):
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if col not in df or df[col].dropna().empty:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("timestamp")
        return

    plot_df = df.dropna(subset=[col])
    sns.lineplot(x="timestamp", y=col, hue="gpu_index", data=plot_df, marker="o", ax=ax)
    ax.set_xlabel("timestamp")


def _plot_token_counts(sns, ax, df):
    ax.set_title("Tokens Per Request")
    ax.set_ylabel("tokens")

    cols = [col for col in ("prompt_tokens", "completion_tokens", "total_tokens") if col in df]
    token_df = df[["timestamp", *cols]].melt(
        "timestamp",
        var_name="type",
        value_name="tokens",
    )
    token_df = token_df.dropna(subset=["tokens"])
    if token_df.empty:
        ax.text(0.5, 0.5, "No token usage samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("timestamp")
        return

    sns.lineplot(x="timestamp", y="tokens", hue="type", data=token_df, marker="o", ax=ax)
    ax.set_xlabel("timestamp")


def _plot_phase_breakdown(ax, df):
    ax.set_title("Request Phase Breakdown")
    ax.set_ylabel("seconds")

    cols = [col for col in ("queue_prefill_s", "decode_s") if col in df and not df[col].dropna().empty]
    if not cols:
        ax.text(0.5, 0.5, "No phase samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("request")
        return

    plot_df = df.reset_index(drop=True).tail(40).copy()
    x = list(range(len(plot_df)))
    labels = plot_df["timestamp"].dt.strftime("%H:%M:%S").tolist()
    queue_prefill = _series_or_zero(plot_df, "queue_prefill_s")
    decode = _series_or_zero(plot_df, "decode_s")

    ax.bar(x, queue_prefill, label="queue+prefill", color="#60a5fa")
    ax.bar(x, decode, bottom=queue_prefill, label="decode", color="#34d399")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("request")
    ax.legend()


def _plot_prompt_size_scatter(sns, ax, df):
    ax.set_title("Throughput vs Prompt Size")
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel("completion tokens/s")

    required = ["prompt_tokens", "completion_tok_s"]
    if any(col not in df or df[col].dropna().empty for col in required):
        ax.text(0.5, 0.5, "No token samples", ha="center", va="center", transform=ax.transAxes)
        return

    scatter_df = df.dropna(subset=required)
    if scatter_df.empty:
        ax.text(0.5, 0.5, "No token samples", ha="center", va="center", transform=ax.transAxes)
        return

    sns.scatterplot(
        x="prompt_tokens",
        y="completion_tok_s",
        size="completion_tokens",
        sizes=(35, 160),
        data=scatter_df,
        ax=ax,
        legend=True,
    )


def _plot_latency_distribution(ax, df):
    ax.set_title("Latency Distribution")
    ax.set_xlabel("seconds")
    ax.set_ylabel("requests")

    if "duration_s" not in df or df["duration_s"].dropna().empty:
        ax.text(0.5, 0.5, "No latency samples", ha="center", va="center", transform=ax.transAxes)
        return

    latency = df["duration_s"].dropna()
    ax.hist(latency, bins=min(max(len(latency), 3), 20), color="#60a5fa", alpha=0.8)

    for quantile, color in ((0.5, "#fbbf24"), (0.95, "#fb7185"), (0.99, "#c084fc")):
        value = latency.quantile(quantile)
        ax.axvline(value, color=color, linestyle="--", linewidth=1.6, label=f"P{int(quantile * 100)} {value:.2f}s")

    ax.legend()


def _write_request_summary(df):
    summary_metrics = [
        "duration_s",
        "time_to_first_token_s",
        "queue_prefill_s",
        "decode_s",
        "prompt_tok_s",
        "completion_tok_s",
        "total_tok_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ]
    rows = []

    for metric in summary_metrics:
        if metric not in df:
            continue

        values = df[metric].dropna()
        if values.empty:
            continue

        rows.append({
            "metric": metric,
            "count": int(values.count()),
            "mean": round(float(values.mean()), 4),
            "p50": round(float(values.quantile(0.5)), 4),
            "p95": round(float(values.quantile(0.95)), 4),
            "p99": round(float(values.quantile(0.99)), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        })

    if not rows:
        return

    with LLM_REQUEST_SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_cache_hit_rates(sns, ax, df):
    ax.set_title("Cache Hit Rates")
    ax.set_ylabel("%")

    cols = [col for col in ("prefix_hit", "mm_cache_hit") if col in df and not df[col].dropna().empty]
    if not cols:
        ax.text(0.5, 0.5, "No cache samples", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("timestamp")
        return

    cache_df = df[["timestamp", *cols]].melt(
        "timestamp",
        var_name="cache",
        value_name="hit_rate",
    )
    sns.lineplot(x="timestamp", y="hit_rate", hue="cache", data=cache_df, marker="o", ax=ax)
    ax.set_xlabel("timestamp")


def _to_bool_series(series):
    if series is None:
        return False

    return series.astype(str).str.lower().isin(("1", "true", "yes"))


def _series_or_zero(df, col):
    if col in df:
        return df[col].fillna(0)

    return [0] * len(df)


atexit.register(flush_llm_request_metrics)
atexit.register(flush_vllm_engine_metrics)
