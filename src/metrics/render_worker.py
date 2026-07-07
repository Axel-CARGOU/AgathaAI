import argparse
from pathlib import Path

from src.metrics import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", choices=("request", "engine", "nvidia"), required=True)
    parser.add_argument("--frozen-ts", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--request-csv", required=True)
    parser.add_argument("--request-summary-csv", required=True)
    parser.add_argument("--engine-csv", required=True)
    parser.add_argument("--nvidia-csv", required=False)
    parser.add_argument("--include-logs", action="store_true")
    args = parser.parse_args()

    metrics.frozen_ts = args.frozen_ts
    metrics.BASE_DIR = Path(args.base_dir)
    metrics.LOG_DIR = Path(args.log_dir)
    metrics.OUT_DIR = Path(args.out_dir)
    metrics.LLM_REQUEST_METRICS_FILE = Path(args.request_csv)
    metrics.LLM_REQUEST_SUMMARY_FILE = Path(args.request_summary_csv)
    metrics.VLLM_ENGINE_METRICS_FILE = Path(args.engine_csv)
    if args.nvidia_csv:
        metrics.NVIDIA_SMI_METRICS_FILE = Path(args.nvidia_csv)

    if args.graph == "request":
        metrics.make_llm_request_graph_on_shutdown_sync(silent_missing=True)
    elif args.graph == "engine":
        metrics.make_vllm_engine_graph_on_shutdown_sync(
            silent_missing=True,
            include_logs=args.include_logs,
        )
    else:
        metrics.make_nvidia_smi_graph_on_shutdown_sync(silent_missing=True)


if __name__ == "__main__":
    main()
