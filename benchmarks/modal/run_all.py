"""Single-command benchmark orchestrator.

Usage:
    modal run benchmarks/modal/run_all.py --phase 1 --gpu H100
    python benchmarks/modal/run_all.py --local --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import time
from pathlib import Path

import yaml


def load_models(phase: int) -> list[dict]:
    config_path = Path(__file__).parent / "configs" / "models.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get(f"phase_{phase}", [])


def run_all_benchmarks(
    model_name: str,
    router_path: str,
    output_dir: str = "./results",
):
    """Run all 6 benchmarks for a given model."""
    from benchmarks.modal.bench_throughput import bench_throughput
    from benchmarks.modal.bench_latency import bench_latency
    from benchmarks.modal.bench_quality import bench_quality
    from benchmarks.modal.bench_reasoning import bench_reasoning
    from benchmarks.modal.bench_memory import bench_memory
    from benchmarks.modal.bench_exit_distribution import bench_exit_distribution

    benchmarks = [
        ("throughput", lambda: bench_throughput(model_name, router_path, output_dir=output_dir)),
        ("latency", lambda: bench_latency(model_name, router_path, output_dir=output_dir)),
        ("quality", lambda: bench_quality(model_name, router_path, output_dir=output_dir)),
        ("reasoning", lambda: bench_reasoning(model_name, router_path, output_dir=output_dir)),
        ("memory", lambda: bench_memory(model_name, router_path, output_dir=output_dir)),
        ("exit_distribution", lambda: bench_exit_distribution(model_name, router_path, output_dir=output_dir)),
    ]

    all_results = {}
    for name, bench_fn in benchmarks:
        print(f"\n{'='*60}")
        print(f"Running {name} benchmark for {model_name}")
        print(f"{'='*60}")
        try:
            result = bench_fn()
            all_results[name] = result
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[name] = {"error": str(e)}

    # Generate summary report
    safe_name = model_name.replace("/", "_")
    summary_path = f"{output_dir}/{safe_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    _generate_markdown_report(all_results, model_name, output_dir)
    return all_results


def _generate_markdown_report(results: dict, model_name: str, output_dir: str):
    """Generate a Markdown summary report."""
    safe_name = model_name.replace("/", "_")
    lines = [
        f"# TIDE Benchmark Report: {model_name}",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if "throughput" in results and "entries" in results["throughput"]:
        lines.append("## Throughput")
        lines.append("| Batch | Threshold | TIDE tok/s | HF tok/s | Speedup | Exit Rate |")
        lines.append("|-------|-----------|-----------|---------|---------|-----------|")
        for e in results["throughput"]["entries"]:
            lines.append(
                f"| {e['batch_size']} | {e['threshold']} | "
                f"{e['tide_tokens_per_sec']:.0f} | {e['hf_tokens_per_sec']:.0f} | "
                f"{e['speedup']:.2f}x | {e['exit_rate']:.1%} |"
            )
        lines.append("")

    if "memory" in results and "entries" in results["memory"]:
        lines.append("## Memory")
        lines.append("| Batch | HF Peak MB | TIDE Peak MB | Overhead |")
        lines.append("|-------|-----------|-------------|----------|")
        for e in results["memory"]["entries"]:
            lines.append(
                f"| {e['batch_size']} | {e['hf_peak_mb']:.0f} | "
                f"{e['tide_peak_mb']:.0f} | {e['overhead_pct']:+.1f}% |"
            )
        lines.append("")

    report_path = f"{output_dir}/{safe_name}_report.md"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="TIDE Benchmark Orchestrator")
    parser.add_argument("--phase", type=int, default=1, help="Model phase (1-3)")
    parser.add_argument("--model", type=str, default=None, help="Specific model name")
    parser.add_argument("--router-dir", type=str, default="./routers", help="Router weights directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--local", action="store_true", help="Run locally instead of Modal")
    args = parser.parse_args()

    if args.model:
        models = [{"name": args.model, "short": args.model.split("/")[-1]}]
    else:
        models = load_models(args.phase)

    for model_info in models:
        model_name = model_info["name"]
        safe_name = model_name.replace("/", "_")
        router_path = f"{args.router_dir}/{safe_name}_router.pt"

        print(f"\n{'#'*60}")
        print(f"# Benchmarking: {model_name}")
        print(f"{'#'*60}")

        run_all_benchmarks(model_name, router_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
