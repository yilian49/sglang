"""DFLASH GSM8K sweep benchmark.

Two modes:
1. Default: Compare baseline vs DFLASH (fused KV enabled by default)
2. --compare-fused-kv: Compare baseline, DFLASH unfused, DFLASH fused

Example usage:
  # Compare baseline vs DFLASH
  python benchmark/dflash/bench_dflash_gsm8k_sweep.py --output-md dflash_gsm8k_sweep.md

  # Compare all three modes
  python benchmark/dflash/bench_dflash_gsm8k_sweep.py --compare-fused-kv --output-md full_sweep.md
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    find_available_port,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl

INVALID = -9999999


def _is_blackwell() -> bool:
    if envs.IS_BLACKWELL.get():
        return True
    return get_device_sm() >= 100


def _get_one_example(lines, i: int, include_answer: bool) -> str:
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def _get_few_shot_examples(lines, k: int) -> str:
    ret = ""
    for i in range(k):
        ret += _get_one_example(lines, i, True) + "\n\n"
    return ret


def _get_answer_value(answer_str: str) -> int:
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def _maybe_download_gsm8k(data_path: str) -> str:
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    if os.path.isfile(data_path):
        return data_path
    return download_and_cache_file(url)


def _flush_cache(base_url: str) -> None:
    resp = requests.get(base_url + "/flush_cache", timeout=60)
    resp.raise_for_status()


def _send_generate(
    base_url: str,
    prompt: str,
    *,
    max_new_tokens: int,
    stop: list[str],
    timeout_s: int,
) -> dict:
    sampling_params: dict = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_new_tokens": int(max_new_tokens),
    }
    if stop:
        sampling_params["stop"] = stop
    resp = requests.post(
        base_url + "/generate",
        json={"text": prompt, "sampling_params": sampling_params},
        timeout=int(timeout_s),
    )
    resp.raise_for_status()
    return resp.json()


@dataclass(frozen=True)
class BenchMetrics:
    latency_s: float
    output_tokens: int
    output_toks_per_s: float
    accuracy: Optional[float]
    invalid_rate: Optional[float]
    spec_accept_length: Optional[float]
    spec_verify_ct_sum: int


def _run_gsm8k_requests(
    base_url: str,
    *,
    prompts: list[str],
    labels: Optional[list[int]],
    max_new_tokens: int,
    concurrency: int,
    stop: list[str],
    timeout_s: int,
    expect_dflash: bool,
) -> BenchMetrics:
    if labels is not None and len(labels) != len(prompts):
        raise ValueError("labels length must match prompts length")

    start = time.perf_counter()
    total_tokens = 0
    spec_verify_ct_sum = 0
    spec_accept_lengths: list[float] = []
    correct = 0
    invalid = 0

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        futures = {
            pool.submit(
                _send_generate,
                base_url,
                prompt,
                max_new_tokens=max_new_tokens,
                stop=stop,
                timeout_s=timeout_s,
            ): i
            for i, prompt in enumerate(prompts)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            spec_verify_ct_sum += int(meta.get("spec_verify_ct", 0))
            if "spec_accept_length" in meta:
                try:
                    spec_accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass
            if labels is not None:
                pred = _get_answer_value(out.get("text", ""))
                if pred == INVALID:
                    invalid += 1
                if pred == labels[i]:
                    correct += 1

    latency = time.perf_counter() - start
    toks_per_s = total_tokens / max(latency, 1e-6)

    if expect_dflash and spec_verify_ct_sum <= 0:
        raise RuntimeError(
            "DFLASH sanity check failed: did not observe any `spec_verify_ct` in responses."
        )

    spec_accept_length = (
        float(statistics.mean(spec_accept_lengths)) if spec_accept_lengths else None
    )

    if labels is None:
        acc = None
        invalid_rate = None
    else:
        acc = correct / max(len(prompts), 1)
        invalid_rate = invalid / max(len(prompts), 1)

    return BenchMetrics(
        latency_s=float(latency),
        output_tokens=int(total_tokens),
        output_toks_per_s=float(toks_per_s),
        accuracy=acc,
        invalid_rate=invalid_rate,
        spec_accept_length=spec_accept_length,
        spec_verify_ct_sum=int(spec_verify_ct_sum),
    )


def _format_table(
    *,
    tp_sizes: list[int],
    concurrencies: list[int],
    values: dict[tuple[int, int], Optional[float]],
    float_fmt: str,
) -> str:
    header = ["tp\\conc"] + [str(c) for c in concurrencies]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for tp in tp_sizes:
        row = [str(tp)]
        for c in concurrencies:
            v = values.get((tp, c), None)
            row.append("N/A" if v is None else format(v, float_fmt))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=str, default="dflash_gsm8k_sweep.md")
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["fewshot_qa", "chat"],
        default="chat",
    )
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.75)
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-running-requests", type=int, default=64)
    parser.add_argument(
        "--tp-sizes",
        type=str,
        default="1,2,4,8",
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="1,2,4,8,16,32",
    )
    parser.add_argument(
        "--questions-per-concurrency-base",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max-questions-per-config",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--attention-backends",
        type=str,
        default="flashinfer,fa3",
    )
    parser.add_argument(
        "--compare-fused-kv",
        action="store_true",
        help="Compare baseline, DFLASH unfused, DFLASH fused.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline (only compare DFLASH modes when used with --compare-fused-kv).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    visible_gpus = int(torch.cuda.device_count())
    tp_sizes = [int(x) for x in args.tp_sizes.split(",") if x.strip()]
    tp_sizes = [tp for tp in tp_sizes if 1 <= tp <= visible_gpus]
    if not tp_sizes:
        raise RuntimeError(f"No tp sizes runnable with visible_gpus={visible_gpus}.")

    concurrencies = [int(x) for x in args.concurrencies.split(",") if x.strip()]
    concurrencies = [c for c in concurrencies if c >= 1]
    if not concurrencies:
        raise RuntimeError("No concurrencies specified.")

    num_questions_by_conc = {
        c: min(args.questions_per_concurrency_base * c, args.max_questions_per_config)
        for c in concurrencies
    }
    max_questions = max(num_questions_by_conc.values())

    attention_backends = [
        s.strip() for s in args.attention_backends.split(",") if s.strip()
    ]
    is_blackwell = _is_blackwell()
    device_sm = get_device_sm()
    if is_blackwell:
        attention_backends = [b for b in attention_backends if b == "flashinfer"]
    if device_sm < 90:
        attention_backends = [b for b in attention_backends if b != "fa3"]
    attention_backends = attention_backends or ["flashinfer"]

    data_path = _maybe_download_gsm8k(args.data_path)
    lines = list(read_jsonl(data_path))
    if len(lines) < max_questions:
        raise RuntimeError(
            f"GSM8K file only has {len(lines)} lines, need {max_questions}."
        )

    tokenizer = None
    if args.prompt_style == "chat":
        tokenizer = AutoTokenizer.from_pretrained(args.target_model)

    few_shot = (
        _get_few_shot_examples(lines, args.num_shots)
        if args.prompt_style == "fewshot_qa"
        else ""
    )

    prompts: list[str] = []
    labels: list[int] = []
    for i in range(max_questions):
        if args.prompt_style == "fewshot_qa":
            prompts.append(few_shot + _get_one_example(lines, i, False))
        else:
            assert tokenizer is not None
            user_content = (
                lines[i]["question"]
                + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        labels.append(_get_answer_value(lines[i]["answer"]))
    if not all(lab != INVALID for lab in labels):
        raise RuntimeError("Invalid labels in GSM8K data.")

    default_stop = (
        ["Question", "Assistant:", "<|separator|>"]
        if args.prompt_style == "fewshot_qa"
        else []
    )

    # Results indexed by (backend, tp, concurrency)
    baseline_toks: dict[tuple[str, int, int], Optional[float]] = {}
    baseline_acc: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_toks: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_acc: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_accept_len: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_unfused_toks: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_unfused_acc: dict[tuple[str, int, int], Optional[float]] = {}
    dflash_unfused_accept_len: dict[tuple[str, int, int], Optional[float]] = {}

    for backend in attention_backends:
        for tp in tp_sizes:
            common_server_args: list[str] = [
                "--trust-remote-code",
                "--attention-backend",
                backend,
                "--tp-size",
                str(tp),
                "--dtype",
                args.dtype,
                "--mem-fraction-static",
                str(args.mem_fraction_static),
                "--max-running-requests",
                str(args.max_running_requests),
                "--cuda-graph-bs",
                *[str(i) for i in range(1, 33)],
                "--cuda-graph-max-bs",
                "32",
            ]
            if args.disable_radix_cache:
                common_server_args.append("--disable-radix-cache")

            port_base = 20000
            last_port = port_base

            # Run baseline (skip if --skip-baseline)
            if not args.skip_baseline:
                print(f"\n=== backend={backend} tp={tp} (baseline) ===")
                baseline_port = find_available_port(port_base)
                baseline_url = f"http://127.0.0.1:{baseline_port}"
                baseline_proc = popen_launch_server(
                    args.target_model,
                    baseline_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=common_server_args,
                )
                try:
                    _send_generate(
                        baseline_url, "Hello", max_new_tokens=8, stop=[], timeout_s=300
                    )
                    for conc in concurrencies:
                        n = num_questions_by_conc[conc]
                        _flush_cache(baseline_url)
                        metrics = _run_gsm8k_requests(
                            baseline_url,
                            prompts=prompts[:n],
                            labels=labels[:n],
                            max_new_tokens=args.max_new_tokens,
                            concurrency=conc,
                            stop=default_stop,
                            timeout_s=args.timeout_s,
                            expect_dflash=False,
                        )
                        baseline_toks[(backend, tp, conc)] = metrics.output_toks_per_s
                        baseline_acc[(backend, tp, conc)] = metrics.accuracy
                        print(
                            f"[baseline] conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"acc={metrics.accuracy:.3f}"
                        )
                finally:
                    kill_process_tree(baseline_proc.pid)
                    try:
                        baseline_proc.wait(timeout=30)
                    except Exception:
                        pass
                last_port = baseline_port

            dflash_common = [
                *common_server_args,
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                args.draft_model,
            ]

            if args.compare_fused_kv:
                # Run DFLASH unfused
                print(f"\n=== backend={backend} tp={tp} (DFLASH unfused) ===")
                unfused_port = find_available_port(last_port + 1)
                unfused_url = f"http://127.0.0.1:{unfused_port}"
                unfused_proc = popen_launch_server(
                    args.target_model,
                    unfused_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        *dflash_common,
                        "--disable-speculative-dflash-fused-kv",
                    ],
                )
                try:
                    _send_generate(
                        unfused_url, "Hello", max_new_tokens=8, stop=[], timeout_s=300
                    )
                    for conc in concurrencies:
                        n = num_questions_by_conc[conc]
                        _flush_cache(unfused_url)
                        metrics = _run_gsm8k_requests(
                            unfused_url,
                            prompts=prompts[:n],
                            labels=labels[:n],
                            max_new_tokens=args.max_new_tokens,
                            concurrency=conc,
                            stop=default_stop,
                            timeout_s=args.timeout_s,
                            expect_dflash=True,
                        )
                        dflash_unfused_toks[(backend, tp, conc)] = (
                            metrics.output_toks_per_s
                        )
                        dflash_unfused_acc[(backend, tp, conc)] = metrics.accuracy
                        dflash_unfused_accept_len[(backend, tp, conc)] = (
                            metrics.spec_accept_length
                        )
                        print(
                            f"[unfused]  conc={conc:>2} n={n:<4} "
                            f"toks/s={metrics.output_toks_per_s:,.2f} "
                            f"acc={metrics.accuracy:.3f} "
                            f"accept_len={metrics.spec_accept_length:.3f}"
                        )
                finally:
                    kill_process_tree(unfused_proc.pid)
                    try:
                        unfused_proc.wait(timeout=30)
                    except Exception:
                        pass

                # Run DFLASH fused (default)
                print(f"\n=== backend={backend} tp={tp} (DFLASH fused) ===")
                fused_port = find_available_port(unfused_port + 1)
            else:
                fused_port = find_available_port(last_port + 1)

            # Run DFLASH (fused by default)
            mode_label = "DFLASH fused" if args.compare_fused_kv else "DFLASH"
            if not args.compare_fused_kv:
                print(f"\n=== backend={backend} tp={tp} ({mode_label}) ===")
            fused_url = f"http://127.0.0.1:{fused_port}"
            fused_proc = popen_launch_server(
                args.target_model,
                fused_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=dflash_common,
            )
            try:
                _send_generate(
                    fused_url, "Hello", max_new_tokens=8, stop=[], timeout_s=300
                )
                for conc in concurrencies:
                    n = num_questions_by_conc[conc]
                    _flush_cache(fused_url)
                    metrics = _run_gsm8k_requests(
                        fused_url,
                        prompts=prompts[:n],
                        labels=labels[:n],
                        max_new_tokens=args.max_new_tokens,
                        concurrency=conc,
                        stop=default_stop,
                        timeout_s=args.timeout_s,
                        expect_dflash=True,
                    )
                    dflash_toks[(backend, tp, conc)] = metrics.output_toks_per_s
                    dflash_acc[(backend, tp, conc)] = metrics.accuracy
                    dflash_accept_len[(backend, tp, conc)] = metrics.spec_accept_length
                    label = "[fused]   " if args.compare_fused_kv else "[DFLASH]  "
                    print(
                        f"{label} conc={conc:>2} n={n:<4} "
                        f"toks/s={metrics.output_toks_per_s:,.2f} "
                        f"acc={metrics.accuracy:.3f} "
                        f"accept_len={metrics.spec_accept_length:.3f}"
                    )
            finally:
                kill_process_tree(fused_proc.pid)
                try:
                    fused_proc.wait(timeout=30)
                except Exception:
                    pass

    # Render markdown
    md_lines: list[str] = []
    if args.compare_fused_kv:
        if args.skip_baseline:
            md_lines.append("# DFLASH Comparison (Unfused vs Fused)")
        else:
            md_lines.append("# DFLASH Comparison (Baseline vs Unfused vs Fused)")
    else:
        md_lines.append("# DFLASH GSM8K Sweep")
    md_lines.append("")
    md_lines.append("## Settings")
    md_lines.append(f"- target_model: `{args.target_model}`")
    md_lines.append(f"- draft_model: `{args.draft_model}`")
    md_lines.append(f"- prompt_style: `{args.prompt_style}`")
    md_lines.append(f"- max_new_tokens: `{args.max_new_tokens}`")
    md_lines.append(f"- attention_backends: `{', '.join(attention_backends)}`")
    md_lines.append(f"- tp_sizes: `{', '.join(str(x) for x in tp_sizes)}`")
    md_lines.append(f"- concurrencies: `{', '.join(str(x) for x in concurrencies)}`")
    md_lines.append(f"- device_sm: `{device_sm}`")
    md_lines.append("")

    for backend in attention_backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")

        baseline_values = {
            (tp, conc): baseline_toks.get((backend, tp, conc))
            for tp in tp_sizes
            for conc in concurrencies
        }
        dflash_values = {
            (tp, conc): dflash_toks.get((backend, tp, conc))
            for tp in tp_sizes
            for conc in concurrencies
        }

        if args.compare_fused_kv:
            unfused_values = {
                (tp, conc): dflash_unfused_toks.get((backend, tp, conc))
                for tp in tp_sizes
                for conc in concurrencies
            }

            # Speedup calculations
            speedup_fused_vs_unfused: dict[tuple[int, int], Optional[float]] = {}
            for tp in tp_sizes:
                for conc in concurrencies:
                    u = unfused_values.get((tp, conc))
                    f = dflash_values.get((tp, conc))
                    speedup_fused_vs_unfused[(tp, conc)] = (
                        f / u if u and f and u > 0 else None
                    )

            if not args.skip_baseline:
                md_lines.append("### Baseline (tok/s)")
                md_lines.append(
                    _format_table(
                        tp_sizes=tp_sizes,
                        concurrencies=concurrencies,
                        values=baseline_values,
                        float_fmt=",.2f",
                    )
                )
                md_lines.append("")
            md_lines.append("### DFLASH Unfused (tok/s)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=unfused_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            md_lines.append("### DFLASH Fused (tok/s)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=dflash_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            if not args.skip_baseline:
                speedup_unfused: dict[tuple[int, int], Optional[float]] = {}
                speedup_fused: dict[tuple[int, int], Optional[float]] = {}
                for tp in tp_sizes:
                    for conc in concurrencies:
                        b = baseline_values.get((tp, conc))
                        u = unfused_values.get((tp, conc))
                        f = dflash_values.get((tp, conc))
                        speedup_unfused[(tp, conc)] = (
                            u / b if b and u and b > 0 else None
                        )
                        speedup_fused[(tp, conc)] = f / b if b and f and b > 0 else None
                md_lines.append("### Speedup: Unfused vs Baseline")
                md_lines.append(
                    _format_table(
                        tp_sizes=tp_sizes,
                        concurrencies=concurrencies,
                        values=speedup_unfused,
                        float_fmt=".3f",
                    )
                )
                md_lines.append("")
                md_lines.append("### Speedup: Fused vs Baseline")
                md_lines.append(
                    _format_table(
                        tp_sizes=tp_sizes,
                        concurrencies=concurrencies,
                        values=speedup_fused,
                        float_fmt=".3f",
                    )
                )
                md_lines.append("")
            md_lines.append("### Speedup: Fused vs Unfused")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=speedup_fused_vs_unfused,
                    float_fmt=".3f",
                )
            )
            md_lines.append("")
            md_lines.append("### Accuracy (Unfused / Fused)")
            for tp in tp_sizes:
                for conc in concurrencies:
                    u_acc = dflash_unfused_acc.get((backend, tp, conc))
                    f_acc = dflash_acc.get((backend, tp, conc))
                    u_str = f"{u_acc:.3f}" if u_acc is not None else "N/A"
                    f_str = f"{f_acc:.3f}" if f_acc is not None else "N/A"
                    md_lines.append(f"- tp={tp}, conc={conc}: {u_str} / {f_str}")
            md_lines.append("")
        else:
            speedup_values: dict[tuple[int, int], Optional[float]] = {}
            for tp in tp_sizes:
                for conc in concurrencies:
                    b = baseline_values.get((tp, conc))
                    d = dflash_values.get((tp, conc))
                    speedup_values[(tp, conc)] = d / b if b and d and b > 0 else None

            md_lines.append("### Baseline (tok/s)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=baseline_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            md_lines.append("### DFLASH (tok/s)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=dflash_values,
                    float_fmt=",.2f",
                )
            )
            md_lines.append("")
            md_lines.append("### Speedup (DFLASH / Baseline)")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values=speedup_values,
                    float_fmt=".3f",
                )
            )
            md_lines.append("")
            md_lines.append("### DFLASH Acceptance Length")
            md_lines.append(
                _format_table(
                    tp_sizes=tp_sizes,
                    concurrencies=concurrencies,
                    values={
                        (tp, conc): dflash_accept_len.get((backend, tp, conc))
                        for tp in tp_sizes
                        for conc in concurrencies
                    },
                    float_fmt=".3f",
                )
            )
            md_lines.append("")

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nWrote markdown report to: {args.output_md}")


if __name__ == "__main__":
    main()
