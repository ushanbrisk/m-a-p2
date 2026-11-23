#!/usr/bin/env python3
"""YuE2 批量推理: 从 JSONL 请求文件依次生成多首歌.

JSONL 每行一个请求, 字段:
    id      必填, 输出子目录名 (只能用字母数字 . _ -)
    style   必填, 风格描述
    lyrics  必填, 歌词
    cot     可选, full | melody | off, 默认 full
    seed    可选, 默认 831001
    cfg_scale 可选, 0-20
    abc     可选, 外部 ABC 谱 (cover 用)

用法:

    python scripts/batch_yue2.py --input examples/requests.jsonl --output_dir output/batch
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="YuE2 批量推理",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="JSONL 请求文件")
    p.add_argument("--output_dir", default="output/batch")
    p.add_argument("--model", default="m-a-p/YuE2-3B")
    p.add_argument("--vae", default="m-a-p/YuE2-Vae")
    p.add_argument("--device", default="auto")
    p.add_argument("--budget", type=float, default=24.0)
    p.add_argument("--backend", choices=("torch", "torch-eager"), default="torch")
    p.add_argument("--quantization", choices=("none", "fp8"), default="none")
    p.add_argument("--offload_ar", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no_verify_hashes", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")
    p.add_argument("--quiet", "--no_progress", dest="quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    offline = args.offline or yue2_local.apply_offline([args.model, args.vae])
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from yue2 import YuE2Pipeline

    src = Path(args.input)
    if not src.is_absolute():
        src = PROJECT_DIR / src
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r.get("id") for r in rows]
    if any(i is None for i in ids) or len(ids) != len(set(ids)):
        raise SystemExit("每条请求都必须有唯一的 id")

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = PROJECT_DIR / out_root

    pipe = YuE2Pipeline.from_pretrained(
        args.model, vae=args.vae, local_files_only=offline,
        device=args.device, memory_budget_gib=args.budget, backend=args.backend,
        quantization=args.quantization, offload_ar=args.offload_ar,
        verify_hashes=not args.no_verify_hashes, progress=not args.quiet,
    )

    summary, failures = [], 0
    try:
        for index, row in enumerate(rows, 1):
            song_id = row["id"]
            directory = out_root / song_id
            if directory.exists() and any(directory.iterdir()):
                if not args.overwrite:
                    print(f"[{index}/{len(rows)}] {song_id}: 输出已存在, 跳过", file=sys.stderr)
                    continue
                for child in directory.iterdir():
                    child.unlink() if child.is_file() else None
            print(f"[{index}/{len(rows)}] {song_id} 生成中...", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            try:
                song = pipe(
                    style=row["style"], lyrics=row["lyrics"],
                    cot=row.get("cot", "full"), seed=row.get("seed", 831001),
                    abc=row.get("abc"), cfg_scale=row.get("cfg_scale"), id=song_id,
                )
                result = song.save_artifacts(directory)
                summary.append({"id": song_id, "status": "complete",
                                "seconds": round(time.perf_counter() - t0, 1),
                                "audio_seconds": result["audio_seconds"],
                                "output": str(directory)})
            except Exception as exc:
                failures += 1
                summary.append({"id": song_id, "status": "failed",
                                "reason": f"{type(exc).__name__}: {exc}"})
                print(f"    失败: {exc}", file=sys.stderr)
    finally:
        pipe.close()

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "batch.json").write_text(
        json.dumps({"complete": failures == 0, "expected": len(rows),
                    "failed": failures, "results": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
