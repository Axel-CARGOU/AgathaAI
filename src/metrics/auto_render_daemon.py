import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from src.metrics import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-ts", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--request-csv", required=True)
    parser.add_argument("--request-summary-csv", required=True)
    parser.add_argument("--engine-csv", required=True)
    parser.add_argument("--nvidia-csv", required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    _lower_process_priority()
    _configure_metrics_module(args)

    previous = {"request": None, "engine": None, "nvidia": None}
    interval = max(float(args.interval), 1.0)

    while True:
        _render_changed_graphs(previous)
        if args.once:
            return
        time.sleep(interval)


def _configure_metrics_module(args):
    metrics.frozen_ts = args.frozen_ts
    metrics.BASE_DIR = Path(args.base_dir)
    metrics.LOG_DIR = Path(args.log_dir)
    metrics.OUT_DIR = Path(args.out_dir)
    metrics.LLM_REQUEST_METRICS_FILE = Path(args.request_csv)
    metrics.LLM_REQUEST_SUMMARY_FILE = Path(args.request_summary_csv)
    metrics.VLLM_ENGINE_METRICS_FILE = Path(args.engine_csv)
    metrics.NVIDIA_SMI_METRICS_FILE = Path(args.nvidia_csv)


def _lower_process_priority():
    try:
        if hasattr(os, "nice"):
            os.nice(10)
    except OSError:
        pass


def _render_changed_graphs(previous):
    metrics.record_nvidia_smi_metric()

    request_signature = _file_signature(metrics.LLM_REQUEST_METRICS_FILE)
    if request_signature and request_signature != previous["request"]:
        previous["request"] = request_signature
        _render_request_graph()

    engine_signature = _file_signature(metrics.VLLM_ENGINE_METRICS_FILE)
    if engine_signature and engine_signature != previous["engine"]:
        previous["engine"] = engine_signature
        _render_engine_graph()

    nvidia_signature = _file_signature(metrics.NVIDIA_SMI_METRICS_FILE)
    if nvidia_signature and nvidia_signature != previous["nvidia"]:
        previous["nvidia"] = nvidia_signature
        _render_nvidia_graph()


def _file_signature(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    except OSError:
        return None

    if stat.st_size == 0:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def _render_request_graph():
    started = time.perf_counter()
    try:
        metrics.make_llm_request_graph_on_shutdown_sync(silent_missing=True)
    except Exception as exc:
        print(f"Unable to auto-render request metrics graph: {exc}", flush=True)
    finally:
        print(
            f"[PIPELINE:GRAPHS] graph=request duration_s={time.perf_counter() - started:.4f}",
            flush=True,
        )


def _render_engine_graph():
    started = time.perf_counter()
    try:
        metrics.make_vllm_engine_graph_on_shutdown_sync(
            silent_missing=True,
            include_logs=False,
            rewrite_csv=False,
        )
    except Exception as exc:
        print(f"Unable to auto-render vLLM engine metrics graph: {exc}", flush=True)
    finally:
        print(
            f"[PIPELINE:GRAPHS] graph=engine duration_s={time.perf_counter() - started:.4f}",
            flush=True,
        )


def _render_nvidia_graph():
    started = time.perf_counter()
    try:
        metrics.make_nvidia_smi_graph_on_shutdown_sync(silent_missing=True)
    except Exception as exc:
        print(f"Unable to auto-render NVIDIA SMI metrics graph: {exc}", flush=True)
    finally:
        print(
            f"[PIPELINE:GRAPHS] graph=nvidia duration_s={time.perf_counter() - started:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
