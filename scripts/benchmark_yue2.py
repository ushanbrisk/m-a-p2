#!/usr/bin/env python3
"""YuE2 性能基准: 加载耗时 / 显存 / 各阶段吞吐.

用法:

    python scripts/benchmark_yue2.py --mode load
    python scripts/benchmark_yue2.py --mode plan     --gen-tokens 512
    python scripts/benchmark_yue2.py --mode e2e      --gen-tokens 512
    python scripts/benchmark_yue2.py --mode all

各模式含义:
    load  只把 YuE2-3B + VAE 读进内存, 报加载耗时与显存峰值
    plan  只跑乐谱规划 (AR 阶段), 报 token 吞吐
    e2e   完整跑一遍 (规划 -> 语义 -> NAR 合成 -> VAE 解码), 报分阶段耗时
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yue2_local  # noqa: E402

STYLE = "mandarin pop, female vocal, warm acoustic guitar, soft drums, clear diction"
LYRICS = """[Verse]
晚风轻轻吹过窗前
你留下的笑还在昨天
街灯把影子拉得很长
我数着星星回到起点

[Chorus]
让这首歌陪你走远
把所有想念唱成明天
就算世界暂时安静
我们还有这一段旋律
"""


def gpu_mem():
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        return 0.0


def reset_mem():
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    except Exception:
        pass


def build_pipe(args):
    from yue2 import YuE2Pipeline
    offline = args.offline or yue2_local.apply_offline([args.model, args.vae])
    reset_mem()
    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        args.model,
        vae=args.vae,
        local_files_only=offline,
        device=args.device,
        memory_budget_gib=args.budget,
        backend=args.backend,
        quantization=args.quantization,
        offload_ar=args.offload_ar,
        verify_hashes=not args.no_verify_hashes,
        progress=not args.quiet,
    )
    return pipe, time.perf_counter() - t0


def mode_load(args):
    pipe, load_seconds = build_pipe(args)
    try:
        # 真正把 7GB 权重搬上 GPU 才算加载完
        import torch
        t0 = time.perf_counter()
        pipe._load_model()
        to_gpu = time.perf_counter() - t0
        print(json.dumps({
            "mode": "load",
            "resolve_and_integrity_seconds": round(pipe.load_timing.get("resolve_and_integrity_seconds", 0), 2),
            "mot_load_seconds": round(pipe.load_timing.get("mot_load_seconds", 0), 2),
            "total_load_seconds": round(load_seconds, 2),
            "cpu_to_gpu_seconds": round(to_gpu, 2),
            "peak_gpu_gib": round(gpu_mem(), 2),
            "weights": pipe.weights,
        }, ensure_ascii=False, indent=2))
    finally:
        pipe.close()


def mode_plan(args):
    pipe, load_seconds = build_pipe(args)
    try:
        reset_mem()
        t0 = time.perf_counter()
        plan = pipe.plan(style=STYLE, lyrics=LYRICS, cot=args.cot, seed=args.seed)
        plan_seconds = time.perf_counter() - t0
        tokens = plan.timing.get("output_tokens", 0)
        print(json.dumps({
            "mode": "plan",
            "load_seconds": round(load_seconds, 2),
            "plan_seconds": round(plan_seconds, 2),
            "output_tokens": tokens,
            "tokens_per_second": round(tokens / plan_seconds, 2) if plan_seconds else None,
            "truncated": plan.truncated,
            "peak_gpu_gib": round(gpu_mem(), 2),
            "abc_preview": (plan.abc or "")[:400],
        }, ensure_ascii=False, indent=2))
    finally:
        pipe.close()


def mode_e2e(args):
    pipe, load_seconds = build_pipe(args)
    try:
        reset_mem()
        t0 = time.perf_counter()
        song = pipe(style=STYLE, lyrics=LYRICS, cot=args.cot, seed=args.seed)
        e2e = time.perf_counter() - t0
        audio_seconds = len(song.audio) / song.sample_rate
        print(json.dumps({
            "mode": "e2e",
            "load_seconds": round(load_seconds, 2),
            "e2e_seconds": round(e2e, 2),
            "audio_seconds": round(audio_seconds, 2),
            "realtime_factor": round(e2e / audio_seconds, 2) if audio_seconds else None,
            "timing": song.timing,
            "truncated": song.truncated,
            "peak_gpu_gib": round(gpu_mem(), 2),
        }, ensure_ascii=False, indent=2))
    finally:
        pipe.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="YuE2 性能基准",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("load", "plan", "e2e", "all"), default="all")
    p.add_argument("--model", default="m-a-p/YuE2-3B")
    p.add_argument("--vae", default="m-a-p/YuE2-Vae")
    p.add_argument("--cot", choices=("full", "melody", "off"), default="full")
    p.add_argument("--seed", type=int, default=831001)
    p.add_argument("--device", default="auto")
    p.add_argument("--budget", type=float, default=24.0)
    p.add_argument("--backend", choices=("torch", "torch-eager"), default="torch")
    p.add_argument("--quantization", choices=("none", "fp8"), default="none")
    p.add_argument("--offload_ar", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no_verify_hashes", action="store_true")
    p.add_argument("--quiet", "--no_progress", dest="quiet", action="store_true")
    args = p.parse_args(argv)

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


    runs = {"load": mode_load, "plan": mode_plan, "e2e": mode_e2e}
    order = ["load", "plan", "e2e"] if args.mode == "all" else [args.mode]
    for name in order:
        print(f"\n===== {name} =====", file=sys.stderr)
        runs[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
