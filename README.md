# m-a-p / YuE2 本地推理

YuE2 是 m-a-p (HKUST) 在 YuE v1 之后发布的新一代音乐生成模型: **单个 3B 的 AR-NAR
Mixture-of-Transformers 主干**先写出可编辑的 ABC 乐谱, 再生成语义 token, 最后用
flow matching 合成声学 latent, 由 VAE 解码成 48 kHz 立体声成品。

这个项目把官方 `yue2_infer` wheel 包装成一套可以直接跑的脚本 + 一个独立的 conda 环境,
和同目录下的 `../m-a-p` (YuE v1) 项目保持同样的组织方式。

> **状态: 环境已建好并跑通。** conda env `yue2` (Python 3.12.16 + torch 2.10.0+cu128)
> 装在 `/home/luke/miniconda3/envs/yue2`, 模型权重已在 `/ssd/.cache/huggingface`。
> `output/my_song/` 和 `output/tonight_awake/` (官方 demo 复现) 里已经有跑好的成品可以直接听。

---

## 目录结构

```
m-a-p2/
├── README.md                  # 本文件
├── RUN.md                     # 运行手册 (逐步操作 + 常见错误)
├── setup_env.sh               # 一键建环境 (conda env + 依赖 + wheel + 自检)
├── environment.yml            # conda env 定义 (带精确版本)
├── requirements.txt           # pip 依赖 (版本由 wheel METADATA 强制)
├── requirements.lock.txt      # 装完后 pip freeze 的完整快照 (51 行, 含 nvidia-* 轮子)
├── yue2_infer-0.1.5-py3-none-any.whl   # 官方推理 wheel
├── scripts/
│   ├── infer_yue2.py          # 主推理脚本 (文本 -> 歌)
│   ├── batch_yue2.py          # 批量推理 (JSONL -> 多首歌)
│   ├── benchmark_yue2.py      # 加载耗时 / 显存 / 吞吐基准
│   ├── check_env.py           # 环境自检 (版本 / CUDA / 模型缓存 / libsndfile)
│   └── yue2_local.py          # 公共工具: 探测模型是否已在本地 HF 缓存
├── examples/
│   ├── style.txt              # 风格描述样例
│   ├── lyrics.txt             # 歌词样例 (方括号分段)
│   ├── melody.abc             # 外部 ABC 谱样例 (cover 用, 不含和弦)
│   ├── official_style.txt     # 官方 demo 的 style (City Pop)
│   ├── official_lyrics.txt    # 官方 demo 的歌词 (今晚不眠)
│   ├── official_tonight_awake.json  # 官方 demo 完整请求
│   └── requests.jsonl         # 批量请求样例
└── output/                    # 生成结果 (每个任务一个子目录)
    ├── my_song/              # 完整一首 (155.5 s)
    ├── tonight_awake/        # 官方 demo 复现 (212.1 s)
    └── batch/                # 批量生成的 3 首 + batch.json 摘要
```

---

## 环境

**为什么要单独开 env**: `audiocraft` env 是 Python 3.9 + torch 2.1.0, 而 YuE2 要求
Python >= 3.10 (官方推荐 3.12) + torch 2.10.0 + transformers 4.57.6, 两者装不上。

### 版本清单

版本不是随便挑的 —— `yue2_infer-0.1.5` 的 METADATA 里写死了 `Requires-Dist`, 版本对不上
pip 会直接拒绝安装 wheel。

| 包 | 版本 | 来源 / 说明 |
|---|---|---|
| python | 3.12.16 | 官方推荐; wheel 要求 >= 3.10 |
| torch | 2.10.0+cu128 | 走 PyTorch 官方 cu128 源; 本机驱动 570.124.04 / CUDA 12.8 |
| transformers | 4.57.6 | 与模型 `config.json` 里的 `transformers_version` 一致 |
| huggingface-hub | 0.36.2 | 本机 base env 是 1.32.0, 本 env 里**必须降级** |
| safetensors | 0.7.0 | |
| tiktoken | 0.12.0 | 文本 tokenizer (`qwen.tiktoken`) |
| numpy | 2.2.6 | |
| soundfile | 0.13.1 | 读写 flac/wav/mp3。pip 的 wheel 自带 libsndfile 1.2.2 (支持写 mp3), 系统那个 1.0.28 不支持 |
| accelerate | 1.13.0 | |
| yue2-infer | 0.1.5 | 本地 wheel, `--no-deps` 安装 |

torch 2.10.0+cu128 会连带装一整套 `nvidia-*-cu12` 轮子, 是本 env 的大头:

```
nvidia-cublas-cu12==12.8.4.1     nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83     nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90  nvidia-cusparse-cu12==12.5.8.93
nvidia-nccl-cu12==2.27.5         nvidia-nvjitlink-cu12==12.8.93
triton==3.6.0                    cuda-bindings / cuda-pathfinder / ...
```

完整 51 行快照 (含 certifi / tokenizers / sympy 等间接依赖) 见 `requirements.lock.txt`。
整个 env 装完约 7.3 GB, 其中 ~6 GB 是 CUDA 轮子。

### 建环境

```bash
cd ~/distributed_machine_learning/m-a-p2
bash setup_env.sh            # 建 env + 装依赖 + 装 wheel + 自检
```

`setup_env.sh` 是幂等的: 已经装好且版本对得上时会**跳过 pip install**, 10 秒跑完
(每次 pip install 都要联网查一遍索引, 本机到 pypi 的链路很慢, 重跑要十几分钟)。
想强制重装加 `--force`, 想从头重建加 `--recreate`。

或者手动来:

```bash
conda create -n yue2 python=3.12 -y
conda activate yue2
pip install -r requirements.txt                                    # torch 走 cu128 源
pip install --no-deps ./yue2_infer-0.1.5-py3-none-any.whl
python scripts/check_env.py
```

系统层依赖 (本机已满足):

```bash
sudo apt-get install -y libsndfile1     # soundfile 写音频要
```

> `pip install` 那个 wheel 时**必须加 `--no-deps`**: wheel 写死了 `Requires-Dist`,
> 让 pip 自动解依赖会触发大范围版本重排。依赖按 `requirements.txt` 单独装。
>
> torch 必须走 cu128 源 (`--extra-index-url https://download.pytorch.org/whl/cu128`),
> PyPI 默认源的 torch 可能是别的 CUDA 轮子。

---

## 模型清单 (已下载到 `/ssd/.cache/huggingface/hub`)

| 用途 | 模型 | 大小 | 必需 | 本机已下载 |
|---|---|---|---|---|
| 主生成 | `m-a-p/YuE2-3B` | 7.3 GB | 是 | 是 |
| 解码器 (默认) | `m-a-p/YuE2-Vae` | 0.53 GB | 是 | 是 |
| 解码器 (官方 benchmark 用) | `m-a-p/YuE2-Vae-legacy` | 0.53 GB | 可选 | 是 |
| 音频转谱 (cover 用) | `m-a-p/SheetSage2` | 0.23 GB | 可选 | 是 |
| SheetSage2 的编码器 | `m-a-p/MERT-v2-FullSong` | 2.53 GB | 可选 | 否 |

纯文本生成歌曲只需要前两个。`YuE2-Vae-legacy` 是官方复现 benchmark 分数时用的解码器
(`m-a-p/YuE2-Vae` 是默认的、日常更好听的版本)。

没下的话用 `hf` 命令拉 (和 YuE v1 一个套路):

```bash
hf download m-a-p/YuE2-3B
hf download m-a-p/YuE2-Vae
hf download m-a-p/YuE2-Vae-legacy      # 复现官方 benchmark 才需要
hf download m-a-p/SheetSage2            # cover 流程需要
hf download m-a-p/MERT-v2-FullSong      # cover 流程需要
```

> YuE v1 那四个仓库 (`YuE-s1-7B-anneal-*` / `YuE-s2-1B-general` / `YuE-upsampler` /
> `xcodec_mini_infer`) 跑 YuE2 一个都用不上, 别下。

> `HF_HOME=/ssd/.cache/huggingface` 已经在 shell 里设好了, 脚本直接用 repo id
> (`m-a-p/YuE2-3B`) 就能命中缓存。**不用手动加 `--offline`**: 脚本会自己探测模型是否
> 已在缓存里, 在的话就自动切离线 (`HF_HUB_OFFLINE=1`), 免得 transformers 去
> huggingface.co 做 HEAD 校验时卡在指数退避重试上。

---

## 快速开始

```bash
conda activate yue2
cd ~/distributed_machine_learning/m-a-p2

python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output --id my_song
```

产物在 `output/my_song/`:

```
output/my_song/
├── audio.flac        # 48 kHz 立体声成品 (24bit)
├── audio.mp3         # 同一份音频的 mp3 (方便直接听)
├── score.abc         # 模型写的乐谱 (旋律 + 和弦)
├── plan.json         # 规划阶段的 request / 耗时 / token
├── abc_tokens.npy    # 规划阶段的 token id
├── plan_manifest.json# 规划阶段产物指纹
├── semantic.npy      # 语义 token (int32)
├── latent.npy        # 声学 latent (float32)
├── prefix.npy        # 前缀 token
├── request.json      # 本次请求 (style / lyrics / cot / seed)
├── config.json       # 完整生成配置 + 权重指纹
└── result.json       # 耗时、显存、truncated 标记、各文件 sha256
```

批量生成:

```bash
python scripts/batch_yue2.py --input examples/requests.jsonl --output_dir output/batch
```

复现官方 demo (City Pop / 今晚不眠):

```bash
python scripts/infer_yue2.py \
    --style_txt examples/official_style.txt \
    --lyrics_txt examples/official_lyrics.txt \
    --seed 12300 --budget 22 \
    --output_dir output --id tonight_awake
```

性能基准:

```bash
python scripts/benchmark_yue2.py --mode all
```

---

## 实测结果 (2026-11-09, RTX 3090 24GB)

### 跑完整一首 (`examples/style.txt` + `examples/lyrics.txt`)

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 解析模型路径 (本地 HF 缓存) | 0.1 s | `--offline` 时完全不走网络 |
| sha256 权重校验 | 21.6 s | 7.3 GB 主模型 + 0.5 GB VAE; `--no_verify_hashes` 可跳过 |
| 权重加载 (MOT 上 GPU) | 0.4 s | 显存峰值 6.76 GiB (bf16) |
| 乐谱规划 (AR) | 12.4 s | 1333 tokens, **107 tok/s**, CUDA Graph + flash attention |
| 语义 token (AR) | 35.8 s | 3889 tokens, **109 tok/s** |
| NAR 声学合成 (flow matching, 32 步) | 24.4 s | |
| VAE 解码 | 8.5 s | |
| **端到端** | **85.7 s** | 产出 **155.5 s** 音频, RTF ≈ 0.55 |

产物在 `output/my_song/`。音频: 48 kHz / 立体声 / FLAC 24bit (29 MB), peak 0.84,
RMS 0.128, 无削波 (0 个满量程采样), 99.8% 非静音, `truncated` 全为 false
(没被 token 上限截断)。

模型写出的 ABC 是真谱: D 大调, 和弦进行 D - F#m7 - G - A7sus4 - Bm7 - Em7 - A/C#,
段落标签 (verse / chorus / bridge / interlude) 与歌词结构一一对应。

### 复现官方 demo (`examples/official_style.txt` + `official_lyrics.txt`, `--seed 12300`)

官方 City Pop demo 的歌词更长, 生成的曲子也更长:

| 阶段 | 耗时 | 备注 |
|---|---|---|
| sha256 权重校验 | 22.3 s | |
| 乐谱规划 (AR) | 22.1 s | 2377 tokens, **108 tok/s** |
| 语义 token (AR) | 50.4 s | 5304 tokens, **105 tok/s** |
| NAR 声学合成 | 37.2 s | |
| VAE 解码 | 9.0 s | |
| **端到端** | **123.8 s** | 产出 **212.1 s** 音频, RTF ≈ 0.58 |

产物在 `output/tonight_awake/`: peak 0.99999 (贴着 24-bit 上限, 但 0 个采样溢出,
不算削波), RMS 0.155, 99.7% 非静音, `truncated` 全为 false。

### 批量 (`examples/requests.jsonl`, 3 首)

模型只加载一次, 之后逐首生成:

| id | 风格 | 生成耗时 | 音频时长 | RTF |
|---|---|---|---|---|
| `song_zh_pop` | mandarin pop | 41.7 s | 65.0 s | 0.64 |
| `song_en_rock` | english indie rock | 37.7 s | 62.7 s | 0.60 |
| `song_zh_ballad` | mandarin ballad | 42.2 s | 78.9 s | 0.53 |
| **合计** | | **121.6 s** | **206.6 s** | **0.59** |

摘要写在 `output/batch/batch.json`。

> 显存: 官方推荐 24 GB。`--budget` 默认 24, 超出卡容量会被自动夹到
> `显存 - 2` GiB, 所以 3090 / 3080 都能直接跑。真 OOM 了再依次试
> `--budget 12` → `--quantization fp8` → `--offload_ar`。
>
> 本机 6 张卡: 3 张 3090 (24 GB) + 3 张 3080 (20 GB)。`--device auto` 用的是
> `cuda:0`, torch 按 PCI bus 顺序枚举, 对应物理 `0000:09:00.0` 那张 3090。

---

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--cot` | `full` | `full` = 旋律+和弦规划; `melody` = 仅旋律 (cover 推荐); `off` = 不规划直接出歌 |
| `--seed` | 831001 | 随机种子 |
| `--cfg_scale` | 跟随官方 (full/melody=1.0, off=1.01) | 文本引导强度, 0-20, 越大越贴 prompt |
| `--budget` | 24 | 显存预算 GiB。实际可用 = `min(budget-2, 显存-2)`, 超出卡容量会自动夹到上限; 调到 <= 12 时 VAE 分块更小、更省显存 |
| `--vae` | `m-a-p/YuE2-Vae` | 换成 `m-a-p/YuE2-Vae-legacy` 复现官方 benchmark |
| `--backend` | `torch` | `torch` 走 CUDA Graph 加速; `torch-eager` 更稳但慢 |
| `--quantization` | `none` | `fp8` 省显存, 质量有损 |
| `--offload_ar` | 关 | AR 阶段后把主干卸到 CPU, 显存不够时开 |
| `--plan_only` | 关 | 只出乐谱不渲染音频, 快速看模型写了什么 |
| `--offline` | 自动 | 纯本地缓存, 不联网。**不用手动加**: 脚本会自己探测模型是否已在 HF 缓存里, 在的话就自动切离线, 免得 transformers 去 huggingface.co 做 HEAD 校验时卡住 |
| `--no_verify_hashes` | 关 | 跳过 7 GB 权重的 sha256 校验 (实测省 ~22 秒) |

---

## 和 YuE v1 的区别 (`../m-a-p`)

| | YuE v1 | YuE2 |
|---|---|---|
| 架构 | 两段式: Stage1 (7B) + Stage2 (1B) | 单个 3B AR-NAR Mixture-of-Transformers |
| 规划 | 无显式乐谱 | 先出 ABC 乐谱 (旋律 + 和弦), 可编辑 |
| 解码 | xcodec_mini_infer + YuE-upsampler (Vocos) | YuE2-Vae / YuE2-Vae-legacy |
| 采样率 | 16 kHz -> 44.1 kHz 上采样 | 直接 48 kHz 立体声 |
| 输出 | mp3 | flac (+ 附带 mp3) |
| 环境 | audiocraft env (py 3.9 / torch 2.1) | 独立 yue2 env (py 3.12 / torch 2.10) |

YuE v1 的四个仓库 (`YuE-s1-7B-anneal-*`、`YuE-s2-1B-general`、`YuE-upsampler`、
`xcodec_mini_infer`) 跑 YuE2 一个都用不上。

---

## 授权

YuE2-3B / YuE2-Vae 权重为 **CC BY-NC 4.0** (权重非商用)。官方声明: 个人用户、创作者、
音乐人可以免费使用并对生成物进行商业化, 无需支付版税。商用前请自行确认条款。
