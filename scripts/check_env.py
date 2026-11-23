#!/usr/bin/env python3
"""YuE2 环境自检.

检查项:
    1. Python 版本 (>= 3.10, 官方推荐 3.12)
    2. yue2 wheel 与全部依赖的精确版本
    3. CUDA / GPU / BF16 支持
    4. HF 缓存里的模型权重是否齐全
    5. 依赖能否真正 import

退出码 0 = 一切就绪, 1 = 有阻塞问题。
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path

# wheel 里写死的版本要求 (yue2_infer-0.1.5.dist-info/METADATA)
REQUIRED = {
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.7.0",
    "tiktoken": "0.12.0",
    "numpy": "2.2.6",
    "soundfile": "0.13.1",
    "accelerate": "1.13.0",
    "yue2-infer": "0.1.5",
}

MODELS = {
    "m-a-p/YuE2-3B": ["config.json", "model.safetensors", "qwen.tiktoken",
                      "yue2_generation_config.json", "weights_manifest.json"],
    "m-a-p/YuE2-Vae": ["config.json", "model.safetensors", "weights_manifest.json"],
    "m-a-p/YuE2-Vae-legacy": ["config.json", "model.safetensors", "weights_manifest.json"],
}


def hf_cache_dir():
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return root / "hub"


def snapshot_dir(repo):
    return hf_cache_dir() / ("models--" + repo.replace("/", "--")) / "snapshots"


def check_python():
    v = sys.version_info
    ok = v >= (3, 10)
    print(f"[{'OK ' if ok else 'FAIL'}] Python {v.major}.{v.minor}.{v.micro} "
          f"(要求 >=3.10, 官方推荐 3.12)")
    return ok


def check_packages():
    ok = True
    print("\n--- 依赖版本 (要求与 yue2_infer-0.1.5 wheel 一致) ---")
    for name, want in REQUIRED.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        if got is None:
            print(f"[FAIL] {name:18s} 未安装           (要求 {want})")
            ok = False
            continue
        # PEP 440: "2.10.0+cu128" 满足 "==2.10.0" (local version 不参与比较)
        base = got.split("+")[0]
        if base == want:
            extra = f"  (本地构建标识 +{got.split('+')[1]})" if "+" in got else ""
            print(f"[OK  ] {name:18s} {got}{extra}")
        else:
            print(f"[WARN] {name:18s} {got}  (要求 {want})")
            ok = False
    return ok


def check_imports():
    ok = True
    print("\n--- import 冒烟测试 ---")
    for mod in ("torch", "transformers", "safetensors", "tiktoken", "numpy",
                "soundfile", "accelerate", "yue2"):
        try:
            importlib.import_module(mod)
            print(f"[OK  ] import {mod}")
        except Exception as exc:
            print(f"[FAIL] import {mod}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


def check_cuda():
    ok = True
    print("\n--- CUDA / GPU ---")
    try:
        import torch
    except Exception:
        return False
    print(f"    torch.version.cuda = {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() == False")
        return False
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        bf16 = torch.cuda.is_bf16_supported()
        print(f"    GPU {i}: {p.name}  显存 {p.total_memory / 2**30:.1f} GiB  "
              f"sm_{p.major}{p.minor}  BF16={bf16}")
        if not bf16:
            print(f"[FAIL] GPU {i} 不支持 BF16, YuE2 未量化模式要求 BF16")
            ok = False
    return ok


def check_models():
    ok = True
    print(f"\n--- HF 缓存 ({hf_cache_dir()}) ---")
    for repo, files in MODELS.items():
        snaps = snapshot_dir(repo)
        if not snaps.is_dir():
            print(f"[FAIL] {repo:26s} 未下载")
            ok = False
            continue
        snap = next((s for s in sorted(snaps.iterdir()) if s.is_dir()), None)
        if snap is None:
            print(f"[FAIL] {repo:26s} 快照为空")
            ok = False
            continue
        missing = [f for f in files if not (snap / f).is_file()]
        size = sum(f.stat().st_size for f in snap.rglob("*") if f.is_file())
        if missing:
            print(f"[FAIL] {repo:26s} 缺文件 {missing}")
            ok = False
        else:
            print(f"[OK  ] {repo:26s} {size / 1e9:.2f} GB")
    return ok


def check_soundfile():
    print("\n--- libsndfile ---")
    try:
        import soundfile as sf
        print(f"    libsndfile {sf.__libsndfile_version__}")
        formats = sf.available_formats()
        for fmt in ("FLAC", "WAV", "MP3", "OGG"):
            print(f"    {fmt:5s} {'可用' if fmt in formats else '不可用'}")
    except Exception as exc:
        print(f"[FAIL] soundfile 不可用: {exc}")
        return False
    return True


def main():
    print("=" * 68)
    print("YuE2 环境自检")
    print("=" * 68)
    print(f"    HF_HOME   = {os.environ.get('HF_HOME', '(默认 ~/.cache/huggingface)')}")
    print(f"    HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE', '(未设置)')}")
    print(f"    python    = {sys.executable}")

    checks = [
        ("Python 版本", check_python()),
        ("依赖版本", check_packages()),
        ("import 冒烟", check_imports()),
        ("CUDA/GPU", check_cuda()),
        ("模型缓存", check_models()),
        ("libsndfile", check_soundfile()),
    ]
    failed = [name for name, ok in checks if not ok]
    print("\n" + "=" * 68)
    if failed:
        print(f"结果: 不通过 -> {', '.join(failed)}")
        return 1
    print("结果: 全部通过, 可以跑 scripts/infer_yue2.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
