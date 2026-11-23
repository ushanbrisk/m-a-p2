# YuE2 本地推理运行手册

从零跑通一首完整歌曲。全流程约 30 分钟 (大部分时间在装环境), 之后每首歌几分钟。

```
0. 建环境 + 自检
1. 准备 style / lyrics
2. 跑推理
3. 看产物
4. 换参数重跑
5. 常见错误
```

---

## 0. 建环境 + 自检 (只做一次)

```bash
cd ~/distributed_machine_learning/m-a-p2
bash setup_env.sh
```

`setup_env.sh` 做四件事:

1. `conda create -n yue2 python=3.12` (不存在才建)
2. `pip install -r requirements.txt` (torch 2.10.0+cu128 + 7 个精确版本依赖)
3. `pip install --no-deps ./yue2_infer-0.1.5-py3-none-any.whl`
4. 跑 `scripts/check_env.py` 自检

脚本是幂等的: 自检通过就**跳过 pip install**, 重跑 10 秒完事 (每次 pip install
都要联网查一遍索引, 本机到 pypi 的链路很慢, 重跑要十几分钟)。

```bash
bash setup_env.sh              # 幂等, 已装好就只跑自检
bash setup_env.sh --force      # 强制重装依赖 (不重建 env)
bash setup_env.sh --recreate   # 先删掉 yue2 env 再从头建
```

自检期望输出:

```
[OK  ] Python 3.12.16 (要求 >=3.10, 官方推荐 3.12)
--- 依赖版本 ---
[OK  ] torch              2.10.0+cu128  (本地构建标识 +cu128)
[OK  ] transformers       4.57.6
[OK  ] huggingface-hub    0.36.2
[OK  ] safetensors        0.7.0
[OK  ] tiktoken           0.12.0
[OK  ] numpy              2.2.6
[OK  ] soundfile          0.13.1
[OK  ] accelerate         1.13.0
[OK  ] yue2-infer         0.1.5
--- CUDA / GPU ---
    GPU 0: NVIDIA GeForce RTX 3090  显存 23.6 GiB  sm_86  BF16=True
    ...
--- HF 缓存 ---
[OK  ] m-a-p/YuE2-3B              7.30 GB
[OK  ] m-a-p/YuE2-Vae             0.53 GB
--- libsndfile ---
    FLAC 可用 / WAV 可用 / MP3 可用
结果: 全部通过, 可以跑 scripts/infer_yue2.py
```

任何一项 FAIL 都不要往下走, 先看第 5 节。

手动激活环境:

```bash
conda activate yue2
```

---

## 1. 准备 style / lyrics

两个纯文本文件, 直接改 `examples/` 下的就行。

### `examples/style.txt`

风格描述, 逗号分隔的标签。中文歌建议带 `mandarin`, 并指定人声性别:

```
mandarin pop, female vocal, warm acoustic guitar, soft drums, clear diction, mid tempo
```

英文:

```
english indie rock, male vocal, distorted electric guitar, driving drums, raw and energetic
```

### `examples/lyrics.txt`

用方括号标签分段, 标签用英文大写 (`[Verse]` / `[Chorus]` / `[Bridge]` / `[Intro]` /
`[Outro]`), 不区分大小写但建议统一:

```
[Verse]
晚风轻轻吹过窗前
你留下的笑还在昨天

[Chorus]
让这首歌陪你走远
把所有想念唱成明天
```

段落结构建议: Verse → Chorus → Verse → Chorus → Bridge → Chorus → Outro。
段落太多会让乐谱规划阶段变长, 歌词太短 (< 4 行) 容易生成很短的歌。

---

## 2. 跑推理

```bash
conda activate yue2
cd ~/distributed_machine_learning/m-a-p2

python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output --id my_song
```

也可以直接把 style / lyrics 写在命令行:

```bash
python scripts/infer_yue2.py \
    --style "mandarin ballad, warm piano, slow tempo, female vocal" \
    --lyrics "[Verse]
风吹过窗前
[Chorus]
让这首歌陪你走远" \
    --output_dir output --id inline_song
```

### 想先看看模型写的乐谱 (不渲染音频, 快很多)

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --plan_only --output_dir output --id my_plan
```

产出 `output/my_plan/score.abc`, 直接用任何 ABC 查看器打开。

### 用外部谱做 cover

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --abc_txt examples/melody.abc \
    --cot melody \
    --output_dir output --id cover_song
```

cover 必须用 `--cot melody`, 而且**外部 ABC 里不能带和弦符号** (只留旋律),
否则伴奏会锁死在原来的和声上, 换不动风格。

### 复现官方 demo (City Pop / 今晚不眠)

```bash
python scripts/infer_yue2.py \
    --style_txt examples/official_style.txt \
    --lyrics_txt examples/official_lyrics.txt \
    --seed 12300 --budget 22 \
    --output_dir output --id tonight_awake
```

`examples/official_style.txt` / `official_lyrics.txt` 是从官方仓库的 demo 请求里
拆出来的, `examples/official_tonight_awake.json` 是完整的原始请求。

### 批量

```bash
python scripts/batch_yue2.py --input examples/requests.jsonl --output_dir output/batch
```

模型只加载一次, 后面逐首生成; 摘要写在 `output/batch/batch.json`。

---

## 3. 看产物

```
output/my_song/
├── audio.flac        # 48 kHz 立体声, 24bit —— 正式成品
├── audio.mp3         # 同一份音频的 mp3, 直接双击听
├── score.abc         # 模型写的乐谱 (旋律 + 和弦)
├── plan.json         # 规划阶段: request / 耗时 / output_tokens / truncated
├── abc_tokens.npy    # ABC 的 token id
├── prefix.npy        # 送进语义阶段的前缀 token
├── semantic.npy      # 语义 token (int32)
├── latent.npy        # 声学 latent (float32)
├── request.json      # 本次请求 (style / lyrics / cot / seed / id)
├── config.json       # 完整采样配置 + 权重指纹 + 运行时指纹
├── plan_manifest.json
└── result.json       # 耗时 / 音频时长 / truncated / 每个产物的 sha256
```

**要听的就是 `output/my_song/audio.flac`** (或 `audio.mp3`)。
`result.json` 里的 `truncated` 为 `true` 说明模型在 max_tokens 处被截断了, 歌可能不完整。

看时长和耗时:

```bash
python -c "import json;d=json.load(open('output/my_song/result.json'));print(d['audio_seconds'],'秒');print(json.dumps(d['timing'],indent=2))"
```

---

## 4. 换参数重跑

改完 `examples/style.txt` 或 `examples/lyrics.txt` 后重跑, 但**必须换一个 `--id`**,
否则会撞上 "输出目录非空" 的保护:

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output --id my_song_v2
```

或者先删掉旧的:

```bash
rm -rf output/my_song
```

### 常用调参

| 想要 | 怎么做 |
|---|---|
| 换个风格 | 改 `--style` 里的标签 |
| 歌更长 | 歌词加段落 (Verse/Chorus 重复) |
| 歌更短 | 歌词砍到 2-4 段 |
| 旋律更贴歌词 | `--cfg_scale 1.2` (默认 full/melody 是 1.0, 调大会更强地贴文本) |
| 不想要规划 | `--cot off` (直接出歌, 没有 score.abc) |
| 只要旋律没和弦 | `--cot melody` |
| 换首曲子 | `--seed 1234` |
| 显存不够 | `--budget 12` (VAE 分块更小) 或 `--quantization fp8` 或 `--offload_ar` |
| 想更快 | 保持默认 `--backend torch` (走 CUDA Graph); 换成 `torch-eager` 反而更慢 |
| 少等 20 秒 | `--no_verify_hashes` (跳过 7 GB 权重的 sha256 校验) |

### 性能基准

```bash
python scripts/benchmark_yue2.py --mode all
```

---

## 5. 常见错误

### `ModuleNotFoundError: No module named 'yue2'`

用错 Python 了。必须用 `yue2` env 的:

```bash
conda activate yue2
which python    # 应该指向 /home/luke/miniconda3/envs/yue2/bin/python
```

或者直接用绝对路径 `/home/luke/miniconda3/envs/yue2/bin/python`。

### `pip install yue2_infer-0.1.5-...whl` 报版本冲突

wheel 写死了 `Requires-Dist`, 版本必须完全对上。**不要**让 pip 自动解依赖:

```bash
pip install --no-deps ./yue2_infer-0.1.5-py3-none-any.whl
```

依赖按 `requirements.txt` 单独装。

### 装 torch 时报 `No matching distribution found for torch==2.10.0`

没加 PyTorch 官方源。PyPI 默认源的 torch 可能是别的 CUDA 轮子:

```bash
pip install torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

### `RuntimeError: The unquantized preset requires CUDA BF16 support`

GPU 不支持 BF16 (需要 Ampere 及以上, sm_80+)。本机 RTX 3080/3090 是 sm_86, 没问题。
真遇到的话只能换卡, 或者试 `--quantization fp8`。

### `ValueError: Memory budget must leave room for a 2GiB reserve`

`--budget` 太小了 (要 <= 2 GiB 才会触发)。实际可用显存是
`min(--budget - 2, 显存 - 2)` GiB, 所以**超过卡容量没关系**, 会被自动夹到上限。

想显式压低显存占用 (比如共享一张卡) 再手动调:

```bash
--budget 18
```

### `CUDA out of memory`

按顺序试:

1. `--budget 12` (VAE 改用小分块)
2. `--quantization fp8`
3. `--offload_ar` (AR 阶段后主干卸到 CPU)

### 一直卡在 "Resolving model files" / 反复 retry

`transformers` 加载时会去 huggingface.co 做 HEAD 校验, 网络不通就会指数退避重试。
加 `--offline` 走纯本地缓存:

```bash
python scripts/infer_yue2.py ... --offline
```

或者:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/infer_yue2.py ...
```

### 生成的歌很短 / 音频文件很小 (< 1 MB)

`result.json` 里 `truncated` 为 `true`, 说明模型在 token 上限处被截断, 歌没生成完。
处理办法: 歌词写长一点 (多几段), 或者确认 `--cot full` 而不是 `off`。

### 生成的歌没有声音 / 全是静音

看 `output/<id>/latent.npy` 有没有非零值。如果有, 说明是 VAE 解码环节的问题,
换一个解码器试试:

```bash
--vae m-a-p/YuE2-Vae-legacy
```

### `output/` 目录非空报错

脚本的保护机制, 防止覆盖已有结果。换 `--id` 或先 `rm -rf output/<id>`。

### 想用官方 benchmark 的解码器

```bash
--vae m-a-p/YuE2-Vae-legacy
```

`YuE2-Vae-legacy` 是官方复现 WildSongBench 分数时用的解码器,
`YuE2-Vae` 是默认的、日常更好听的版本。

---

## 6. 和 `../m-a-p` (YuE v1) 的关系

两套东西完全独立, 不要混用环境:

| | m-a-p (v1) | m-a-p2 (本项目, v2) |
|---|---|---|
| conda env | `audiocraft` (py 3.9) | `yue2` (py 3.12) |
| torch | 2.1.0+cu121 | 2.10.0+cu128 |
| 模型 | 4 个 repo | 2 个 repo (3B + VAE) |
| 输出 | mp3, 16k->44.1k 上采样 | flac, 直接 48k 立体声 |
| 脚本 | `scripts/infer_yue.py` | `scripts/infer_yue2.py` |

v1 的模型权重跑 v2 用不上; 反之亦然。
