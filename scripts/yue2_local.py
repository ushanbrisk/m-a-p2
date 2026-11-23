"""YuE2 项目的公共小工具: 本地 HF 缓存探测."""
from __future__ import annotations

import os
from pathlib import Path


def hf_cache_root():
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"


def repo_cached(repo):
    """判断一个 HF repo 是否已经在本地缓存里.

    已经在缓存里就走纯本地 (local_files_only), 避免 transformers 去 huggingface.co
    做 HEAD 校验时卡在指数退避重试上 —— 权重既然下好了就没必要再联网。
    """
    if not repo or Path(repo).expanduser().is_dir():
        return True  # 本地目录路径, 不需要联网
    snaps = hf_cache_root() / ("models--" + str(repo).replace("/", "--")) / "snapshots"
    return snaps.is_dir() and any(s.is_dir() for s in snaps.iterdir())


def all_cached(repos):
    return all(repo_cached(r) for r in repos if r)


def apply_offline(repos):
    """把 HF / transformers 的离线开关打开 (返回是否真的开了)。"""
    if not all_cached(repos):
        return False
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return True
