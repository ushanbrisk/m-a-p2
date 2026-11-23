#!/usr/bin/env python3
"""YuE2 本地推理脚本.

用法示例:

    # 文本生成歌曲
    python scripts/infer_yue2.py \
        --style_txt examples/style.txt \
        --lyrics_txt examples/lyrics.txt \
        --output_dir output --id my_song

    # 直接内联 style / lyrics
    python scripts/infer_yue2.py --style "mandarin pop, female vocal" \
        --lyrics "[Verse]\\n..." --output_dir output

    # 只出乐谱 (不渲染音频)
    python scripts/infer_yue2.py --style_txt examples/style.txt \
        --lyrics_txt examples/lyrics.txt --plan_only --output_dir output

    # 用外部 ABC 谱做 cover (melody 模式, 谱里不要带和弦)
    python scripts/infer_yue2.py --style_txt examples/style.txt \
        --lyrics_txt examples/lyrics.txt --abc_txt examples/melody.abc \
        --cot melody --output_dir output

上游实现见 https://github.com/multimodal-art-projection/YuE ,
本脚本只是对 `yue2` pip 包 (yue2_infer-0.1.5) 做了一层命令行封装。
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
    p = argparse.ArgumentParser(
        description="YuE2 推理: style + lyrics -> 乐谱 -> 语义 token -> 音频",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("输入")
    g.add_argument("--style", help="风格描述, 例如 'mandarin pop, female vocal, warm piano'")
    g.add_argument("--style_txt", default="examples/style.txt", help="从文件读风格描述 (每行一条)")
    g.add_argument("--lyrics", help="歌词, 用 [Verse]/[Chorus] 等方括号标签分段")
    g.add_argument("--lyrics_txt", default="examples/lyrics.txt", help="从文件读歌词")
    g.add_argument("--abc_txt", help="外部 ABC 乐谱 (cover 用), 要求不含和弦符号")
    g.add_argument("--cot", choices=("full", "melody", "off"), default="full",
                   help="full=旋律+和弦规划(默认) melody=仅旋律(cover 推荐) off=不规划")

    g = p.add_argument_group("采样")
    g.add_argument("--seed", type=int, default=831001, help="随机种子")
    g.add_argument("--cfg_scale", type=float, default=None,
                   help="文本引导强度, 0-20; 官方默认 full/melody=1.0, off=1.01")
    g.add_argument("--id", default=None, help="本次生成的标识, 决定输出子目录名")

    g = p.add_argument_group("模型")
    g.add_argument("--model", default="m-a-p/YuE2-3B", help="主模型 (HF repo id 或本地目录)")
    g.add_argument("--vae", default="m-a-p/YuE2-Vae",
                   help="解码器: m-a-p/YuE2-Vae (默认) | m-a-p/YuE2-Vae-legacy (官方 benchmark 用) | 本地路径")
    g.add_argument("--revision", default=None)
    g.add_argument("--vae_revision", default=None)
    g.add_argument("--backend", choices=("torch", "torch-eager"), default="torch",
                   help="torch=带 CUDA Graph 加速; torch-eager=更稳但慢")
    g.add_argument("--quantization", choices=("none", "fp8"), default="none")
    g.add_argument("--offload_ar", action="store_true",
                   help="AR 阶段结束后把主干卸到 CPU, 显存不够时开")
    g.add_argument("--no_verify_hashes", action="store_true",
                   help="跳过 7GB 权重的 sha256 校验 (省约 30 秒)")

    g = p.add_argument_group("运行")
    g.add_argument("--device", default="auto", help="cuda / cuda:0 / cpu")
    g.add_argument("--budget", type=float, default=24.0,
                   help="显存预算 GiB, 必须给 GPU 留 2GiB 余量; <=12 时 VAE 分块更小")
    g.add_argument("--output_dir", default="output", help="输出根目录")
    g.add_argument("--plan_only", action="store_true", help="只生成乐谱 ABC, 不渲染音频")
    g.add_argument("--offline", action="store_true", help="只用本地 HF 缓存, 不联网")
    g.add_argument("--quiet", "--no_progress", dest="quiet", action="store_true",
                   help="隐藏进度条 (进度走 stderr, 结果走 stdout)")
    return p.parse_args(argv)


def read_text(path, base):
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.read_text(encoding="utf-8")


def resolve_input(args):
    """把 --style/--lyrics/--style_txt/--lyrics_txt 归一成 style/lyrics 字符串。"""
    style = args.style
    if args.style_txt:
        extra = read_text(args.style_txt, PROJECT_DIR)
        style = f"{style}, {extra}" if style else extra
    lyrics = args.lyrics
    if args.lyrics_txt:
        extra = read_text(args.lyrics_txt, PROJECT_DIR)
        lyrics = f"{lyrics}\n{extra}" if lyrics else extra
    if not style or not lyrics:
        raise SystemExit("必须提供 style 和 lyrics (--style/--style_txt 与 --lyrics/--lyrics_txt)")
    abc = read_text(args.abc_txt, PROJECT_DIR) if args.abc_txt else None
    return style.strip(), lyrics.strip(), abc


def main(argv=None):
    args = parse_args(argv)
    style, lyrics, abc = resolve_input(args)

    # 权重已经下载好, 默认就走本地缓存, 避免 transformers 去 huggingface.co 做 HEAD 校验卡住。
    offline = args.offline or yue2_local.apply_offline([args.model, args.vae])
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from yue2 import YuE2Pipeline
    except ImportError as exc:
        raise SystemExit(
            "找不到 yue2 包。先激活环境并安装 wheel:\n"
            "  conda activate yue2\n"
            "  pip install ~/distributed_machine_learning/m-a-p2/yue2_infer-0.1.5-py3-none-any.whl\n"
            f"原始错误: {exc}"
        )

    song_id = args.id or "song"
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_DIR / out_dir
    out_dir = out_dir / song_id
    if out_dir.exists() and any(out_dir.iterdir()) and not args.plan_only:
        raise SystemExit(f"输出目录非空: {out_dir}\n先删掉它, 或换一个 --id / --output_dir")

    print(f"[yue2] model = {args.model}", file=sys.stderr)
    print(f"[yue2] vae   = {args.vae}", file=sys.stderr)
    print(f"[yue2] cot   = {args.cot}  seed = {args.seed}", file=sys.stderr)

    t0 = time.perf_counter()
    pipe = YuE2Pipeline.from_pretrained(
        args.model,
        vae=args.vae,
        revision=args.revision,
        vae_revision=args.vae_revision,
        local_files_only=offline,
        device=args.device,
        memory_budget_gib=args.budget,
        backend=args.backend,
        quantization=args.quantization,
        offload_ar=args.offload_ar,
        verify_hashes=not args.no_verify_hashes,
        progress=not args.quiet,
    )
    load_seconds = time.perf_counter() - t0
    print(f"[yue2] 加载完成, 耗时 {load_seconds:.1f}s", file=sys.stderr)

    try:
        if args.plan_only:
            plan = pipe.plan(style=style, lyrics=lyrics, cot=args.cot, seed=args.seed,
                              abc=abc, cfg_scale=args.cfg_scale, id=song_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            plan.save(out_dir)
            print(json.dumps({"stage": "plan", "output": str(out_dir),
                              "truncated": plan.truncated}, ensure_ascii=False))
            return 0

        song = pipe(style=style, lyrics=lyrics, cot=args.cot, seed=args.seed,
                    abc=abc, cfg_scale=args.cfg_scale, id=song_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = song.save_artifacts(out_dir)
        # 额外落一份 mp3 方便直接听。pip 的 soundfile wheel 自带 libsndfile 1.2.x,
        # 能直接写 mp3; 万一没有就退回 ffmpeg。flac 才是正式产物, mp3 只是便利副本。
        try:
            import soundfile as sf
            if "MP3" in sf.available_formats():
                sf.write(out_dir / "audio.mp3", song.audio, song.sample_rate)
            else:
                raise RuntimeError("soundfile 不支持 mp3")
        except Exception:
            try:
                import shutil as _shutil
                import subprocess as _subprocess
                ffmpeg = _shutil.which("ffmpeg")
                if not ffmpeg:
                    raise RuntimeError("没有 ffmpeg")
                _subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", str(out_dir / "audio.flac"),
                     "-c:a", "libmp3lame", "-q:a", "2", str(out_dir / "audio.mp3")],
                    check=True)
            except Exception as exc:  # mp3 只是便利产物, 失败不影响主结果
                print(f"[yue2] mp3 导出跳过: {exc}", file=sys.stderr)

        print(json.dumps({
            "status": "complete",
            "output": str(out_dir),
            "audio_seconds": result["audio_seconds"],
            "sample_rate": result["sample_rate"],
            "truncated": result["truncated"],
            "timing": result["timing"],
        }, ensure_ascii=False, indent=2))
        return 0
    finally:
        pipe.close()


if __name__ == "__main__":
    raise SystemExit(main())
