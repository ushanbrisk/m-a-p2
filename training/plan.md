# YuE2 训练方案（调研 + 规划）

> 文档版本：v1（2026-11-24）
> 适用仓库：`/home/luke/distributed_machine_learning/m-a-p2`
> 目标读者：第一次接触这个项目的同学。请**从头读到尾**，不需要先懂大模型。
>
> 这份文档回答四个问题：
> 1. 我们手上的模型、数据、机器，到底是什么状况？（第 1–4 章）
> 2. 要让 YuE2 "学会我们的音乐"，技术上真正的拦路虎是什么？（第 5 章，最重要）
> 3. 训练具体怎么做？数学、代码结构、超参是什么？（第 6–7 章）
> 4. 分几步走、每步怎么验收、风险在哪？（第 8–11 章）

---

## 目录

- [0. 这份文档怎么读](#0-这份文档怎么读)
- [1. 结论先行（一页纸速览）](#1-结论先行一页纸速览)
- [2. 第一课：YuE2 生成一首歌时到底发生了什么](#2-第一课yue2-生成一首歌时到底发生了什么)
- [3. 现状盘点：仓库、硬件、软件、磁盘](#3-现状盘点仓库硬件软件磁盘)
- [4. 数据：我们有什么，以及怎么把它变成训练样本](#4-数据我们有什么以及怎么把它变成训练样本)
- [5. 核心缺口：语义 token（Semantic Token）](#5-核心缺口语义-tokensemantic-token)
- [6. 训练目标设计：NAR 声学 LoRA 完整方案](#6-训练目标设计nar-声学-lora-完整方案)
- [7. 代码架构设计](#7-代码架构设计)
- [8. 分阶段实施计划](#8-分阶段实施计划)
- [9. 评测与验收：怎么判断 adapter 有没有用](#9-评测与验收怎么判断-adapter-有没有用)
- [10. 风险登记册](#10-风险登记册)
- [11. Day-1 具体动作清单](#11-day-1-具体动作清单)
- [附录 A：术语表](#附录-a术语表)
- [附录 B：关键代码位置索引](#附录-b关键代码位置索引)
- [附录 C：参考资料](#附录-c参考资料)

---

## 0. 这份文档怎么读

如果你完全不了解大模型，**第 2 章是必读的**，它会用大白话把"训练 YuE2"这件事讲清楚，不需要任何前置知识。

如果你已经知道什么是 LoRA、什么是 flow matching，可以直接跳到 **第 5 章**（核心缺口）和 **第 6 章**（训练方案），那是这份文档真正需要你做决策的部分。

后面每一章都遵循同样的结构，方便你跳读：

| 结构 | 含义 |
| --- | --- |
| **结论** | 一句话回答"这章想说什么" |
| **背景** | 为什么需要知道这件事 |
| **细节** | 具体数字、具体文件、具体行为 |
| **对训练的影响** | 这一章的结论会怎样改变我们的训练代码 |

**关于"我验证过"和"我推测的"**：文档里凡是我在本地机器上实际跑过、看过文件、读过源码确认的，我会写明来源（例如"`yue2/modeling_yue2.py:609`"或"`ffprobe` 实测"）；凡是来自官方论文或社区仓库、我没有本地复现的，我会标明出处（arXiv 编号 / HuggingFace 仓库名）。**这两类不要混为一谈**，因为我们的训练正确性依赖于前者。

---

## 1. 结论先行（一页纸速览）

### 1.1 三句话版本

1. **能训，而且该训的是"声学那一半"（NAR）**。用我们本地 19,112 首真实歌曲的声学 latent 做 flow matching 微调，可以产出第一个 adapter，这是最快、风险最低、收益最直接的一条路。
2. **真正的拦路虎不是算力，是"语义 token"**。YuE2 官方**没有公开**"真实音频 → 语义 token"的编码器（量化头），而我们的 mp3 音频里没有任何语义 token。如果不解决它，"音乐那一半"（AR）就没法用真实音频训练。
3. **歌词是第二大缺口**。我们的数据有歌名/艺人/专辑/歌单/分类，**没有歌词**，机器上也没有 Whisper。这会影响 AR 半（它靠歌词对齐旋律），但**不影响 NAR 半**。

### 1.2 现状速查表

| 项目 | 状况 | 是否够用 |
| --- | --- | --- |
| GPU | 3× RTX 3080 (20 GB) + 3× RTX 3090 (24 GB)，共 132 GB 显存，当前全空闲 | ✅ 够（LoRA 训练 + DDP） |
| CPU / 内存 | 56 核 Xeon E5-2680 v4 @2.4 GHz，157 GB RAM（可用 143 GB） | ✅ 够（音频解码是 CPU 密集型） |
| 磁盘 | `/ssd` 剩 147 GB ✅；`/ssd11` 只剩 **1.3 GB** ⚠️；`/home` 剩 **9.8 GB** ⚠️ | ⚠️ 所有产物必须落 `/ssd` |
| 模型权重 | `YuE2-3B`（7.26 GB）+ `YuE2-Vae` 已在 HF 缓存 | ✅ |
| 音频 | 19,112 首，433 GB，≈1,130 小时 | ✅ 远超官方 346,000 小时的 1/300 |
| 元数据 | 19,112/19,112 全部 join 成功（歌名/艺人/专辑/歌单/分类） | ✅ |
| 歌词 | **完全没有** ❌ | ❌ 需要外部补 |
| 语义 token 编码器 | **官方未公开** ❌ | ❌ 需要自训或绕过 |
| SheetSage2（转谱） | 权重已缓存，但需要**独立 python 环境** | ⚠️ 需装环境 |
| Python 依赖 | `yue2` 环境缺 scipy / torchaudio / librosa / peft | ⚠️ 有替代方案，见 §3.3 |

### 1.3 推荐的推进顺序（详细版在第 8 章）

```
Phase 0  造清单（manifest）：扫描 + 元数据 join + 风格标签 + 黑名单过滤     [CPU, 约 10 分钟]
   ↓
Phase 1  抽 VAE latent：真实音频 → 25 Hz × 64 维潜变量                   [6 卡, 19k 首约 1 小时]
   ↓
Phase 2  ★ NAR 声学 LoRA 训练 → 第一个 adapter                            [6 卡, 数小时]
   ↓
Phase 3  采样评测：A/B 对比基座 vs adapter 的真实音频                      [单卡, 约 1 小时]
   ↓
Phase 4  （可选）语义 tokenizer 自训 → 解锁 AR 音乐半 LoRA                 [数天]
```

**Phase 2 是本方案的主体**，第 6 章整章都在讲它。

---

## 2. 第一课：YuE2 生成一首歌时到底发生了什么

在讨论怎么训练之前，我们必须先搞清楚"模型在做哪几件事"。这一章是全文的基础，请耐心读完。

### 2.1 结论

YuE2 生成一首歌要经过**三个阶段**，其中前两个阶段由同一个 3.58B 参数的模型完成，第三个阶段由一个独立的 VAE（音频编解码器）完成：

```
[Tags 风格标签] + [Lyrics 歌词]
        │
        │  ① 阶段一：写谱（AR 自回归）
        ▼
   ABC 乐谱文本（"C" "F" "G" 这种记谱符号）
        │
        │  ② 阶段二：唱歌（AR 自回归，继续往下写）
        ▼
   语义 token 序列（每 40 毫秒一个，32768 种取值之一）
        │
        │  ③ 阶段三：渲染成声音（NAR 扩散/流匹配 + VAE 解码）
        ▼
   48 kHz 立体声波形
```

### 2.2 三个阶段的通俗解释

**阶段一：写谱（ABC 乐谱）**
YuE2 先像作曲家一样，把歌曲写成一份**人类可读的乐谱**，用的是 ABC 记谱法——一种纯文本格式，`X:1` 是标题、`K:G` 是调号、`M:4/4` 是拍号、`Q:1/4=88` 是速度，然后是 `V: Vocal`（主唱声部）和 `V: Ins`（器乐声部）两条旋律线，和弦用 `"Am7"` 这种引号包起来的符号写在主唱声部里。
> 为什么这很重要？因为乐谱是**可编辑、可审查、可复用**的。你可以在生成完之后改掉某个小节再重新渲染。YuE2 官方把这叫"symbolic planning"（符号规划），也是它和 Suno 这类纯端到端模型最大的区别。

**阶段二：唱歌（语义 token）**
乐谱写完之后，模型还要"把它唱出来"。但它不是直接输出波形（波形太长、连续、不好学），而是先输出一串**离散的语义 token**：每 40 毫秒吐一个，每个 token 是 0~32767 之间的一个整数。
> 打个比方：语义 token 就像"乐谱的另一套记法"。它记录了"这一刻大概是什么声音、什么情绪、什么音高轮廓"，但不记录精确的波形。它是**音乐内容**的紧凑表示。

**阶段三：渲染成声音（VAE）**
最后一步把语义 token 变成真正的波形。这一步不是自回归的，而是用**流匹配（flow matching）**——一种扩散模型的变体——一次性算出 64 维的连续"声学 latent"，再由 VAE 解码成 48 kHz 立体声。
> 打个比方：阶段一、二是"写 + 唱"，阶段三是"录音棚混音"。前两半决定"唱什么"，后半决定"录出来什么音色"。

### 2.3 关键：模型内部其实分成"两半"

上面三个阶段里，**阶段一和阶段二由同一套网络参数完成，阶段三由另一套参数完成**。官方论文把前者叫 AR 分支（AutoRegressive，自回归），后者叫 NAR 分支（Non-Autoregressive，非自回归）。

在代码层面，这"两半"共享同一个 Transformer 骨架，**每一层都长成两副模样**：

```python
# yue2/modeling_yue2.py:227  DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, config):
        # AR 半（"音乐半"）
        self.input_layernorm          = RMSNorm(...)
        self.self_attn                = Attention(...)     # 注意力
        self.post_attention_layernorm = RMSNorm(...)
        self.mlp                     = MLP(...)            # 前馈
        # NAR 半（"声学半"）
        self.nar_input_layernorm      = RMSNorm(...)
        self.nar_self_attn            = Attention(...)      # 注意力
        self.nar_pre_mlp_layernorm    = RMSNorm(...)
        self.nar_mlp                 = MLP(...)             # 前馈
```

官方管这个叫 **Mixture-of-Transformers（MoT，混合专家）**。两半之间用一种"混合注意力掩码"隔开，规则是（`yue2/modeling_yue2.py:668-685`）：

| 查询来自 \ 键来自 | AR 位置（文字/乐谱/语义 token） | NAR 位置（声学 latent） |
| --- | --- | --- |
| **AR 位置** | 只能看**前面的** AR 位置（因果） | ❌ 完全看不见 |
| **NAR 位置** | 能看**全部** AR 位置 | 能看**全部** NAR 位置（双向） |

> 用大白话说：**"写歌的人"看不到"混音台"的信号，但"混音台"能看到"写歌的人"写的东西**。这保证了训练时声学 latent 的梯度不会泄漏回离散 token 的预测（不然模型就能作弊，直接"看答案"）。

除了 28 层双份 Transformer，NAR 半还有几个专属模块（`yue2/modeling_yue2.py:465-474`）：

| 模块 | 形状 | 作用 |
| --- | --- | --- |
| `vae2llm` | Linear(64 → 2048) | 把 64 维声学 latent 抬升到模型隐藏维度 |
| `llm2vae` | Linear(2048 → 64) | 把模型输出压回 64 维声学 latent |
| `time_embedder` | MLP(256 → 2048 → 2048) | 把时间步 t 编码成向量，告诉模型"现在演到哪一步了" |
| `latent_pos_embed` | 固定正弦表 (24576, 2048) | 声学帧的位置编码（**不是**学出来的） |

### 2.4 为什么这对我们是好消息

因为两半的参数是**完全分开**的，我们可以：

- **只训 NAR 半**（声学 LoRA）：不动"写歌"的脑回路，只让它更懂我们的音色。风险小。
- **只训 AR 半**（音乐 LoRA）：不动渲染器，只改它写歌的偏好。风险也小。
- **两半都训**（社区 Ruach Studio 的做法，"music half" rank 160 + "sound half" rank 320）。

而 **LoRA（Low-Rank Adaptation）** 是达成"只训一小部分"的手段：它在原有的线性层旁边加一个低秩的旁路，训练时只更新这个旁路（几千万参数），推理时把它合并回原权重或者单独挂载。优点是显存小、训练快、base 权重不动（随时可以撤销）。

### 2.5 顺带记住的硬数字

| 东西 | 数值 | 出处 |
| --- | --- | --- |
| 层数 / 隐藏维度 | 28 层 / 2048 | `YuE2Config` |
| 注意力头 | 16 个 Q 头，8 个 KV 头（GQA），head_dim 128 | `YuE2Config` |
| MLP 中间层 | 6144 | `YuE2Config` |
| 词表大小 | 184,704 | `YuE2Config` |
| 上下文长度 | 24,576 token | `YuE2Config` |
| latent 维度 / 帧率 | 64 维 / 25 Hz | `YuE2Config` + VAE |
| 语义 token | 25 Hz，32,768 种 | 官方论文 |
| VAE 压缩比 | 1920（48 kHz ÷ 1920 = 25 Hz） | `YuE2-Vae/config.json` |
| `timestep_shift` | 1.0 | `YuE2Config`（**重要，见 §6.2**） |

---

## 3. 现状盘点：仓库、硬件、软件、磁盘

### 3.1 仓库现状

`m-a-p2` 目前是一个**纯推理**仓库：

```
m-a-p2/
├── README.md / RUN.md          使用说明
├── environment.yml / requirements.txt / requirements.lock.txt
├── setup_env.sh                环境重建脚本
├── yue2_infer-0.1.5-py3-none-any.whl    官方推理 wheel（本地留档）
├── scripts/
│   ├── infer_yue2.py           主推理入口（批量/单条）
│   ├── yue2_local.py           本地模型加载辅助
│   └── ...
├── examples/                   style.txt / lyrics.txt / official_style.txt / song.json
├── output/                     已生成的成品音频
└── tutorial/                   教程
```

**关键事实**：`scripts/infer_yue2.py` 通过 `import yue2` 使用 `yue2_infer-0.1.5` 这个 wheel；wheel 装在 conda 环境 `yue2` 里（`site-packages/yue2/`）。本仓库**没有任何训练代码**，上游 `YuE` 仓库（`/home/luke/distributed_machine_learning/YuE`）的 `src/yue2/` 也**只有推理**，官方从未发布训练脚本。

所以：**训练部分要从零写**。好消息是我们把 `training/` 放在独立目录，训练和推理互不干扰：

```
m-a-p2/
├── scripts/     ← 现有推理，不动
└── training/    ← 新增：训练全套（本文档描述的就是它）
```

### 3.2 硬件

```
$ nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
index, name, memory.total [MiB], memory.used [MiB]
0, NVIDIA GeForce RTX 3080, 20480 MiB, 15 MiB
1, NVIDIA GeForce RTX 3080, 20480 MiB, 15 MiB
2, NVIDIA GeForce RTX 3080, 20480 MiB, 15 MiB
3, NVIDIA GeForce RTX 3090, 24576 MiB, 15 MiB
4, NVIDIA GeForce RTX 3090, 24576 MiB, 15 MiB
5, NVIDIA GeForce RTX 3090, 24576 MiB, 15 MiB
```

- 当前**全部空闲**（每张卡只占 15 MiB，是驱动开销）。
- 合计 132 GB 显存。
- CPU：56 核 Xeon E5-2680 v4 @ 2.4 GHz；内存 157 GB（可用 143 GB）。

**这套配置能做什么**（估算依据见 §6.5）：

| 用途 | 结论 |
| --- | --- |
| 单卡 LoRA 训练（NAR 半，rank 32） | ✅ 20 GB 卡绰绰有余（权重 7.3 GB + 激活 2–4 GB） |
| 6 卡 DDP 数据并行 | ✅ 推荐，吞吐提升接近 5–6 倍 |
| 单卡 VAE 编码 19k 首 | ✅ 约 1 小时（6 卡并行） |
| 全模型微调（3.58B 参数 + AdamW） | ❌ 不可能（要 ~100 GB 优化器状态） |
| 训练时的音频解码 | ✅ 56 核足够，可以和 GPU 训练同时跑 |

### 3.3 软件环境与依赖缺口

当前 conda 环境 `yue2`（Python 3.12）实测：

| 包 | 状态 | 影响 |
| --- | --- | --- |
| `torch 2.10.0+cu128` | ✅ | 够新，支持 `torch.compile`、GQA SDPA |
| `transformers 4.57.6` | ✅ | |
| `accelerate 1.13.0` | ✅ | 可用于 DDP / 混合精度 |
| `safetensors 0.7.0` | ✅ | 存 adapter 用它 |
| `numpy 2.2.6` / `tiktoken 0.12.0` / `yaml` / `tqdm` | ✅ | |
| `soundfile 0.13.1` | ✅ | **能读 FLAC 和 MP3**（libsndfile 1.2.2） |
| `scipy` | ❌ | 少一个常用库（重采样/信号处理） |
| `torchaudio` | ❌ | 音频 I/O 与重采样 |
| `librosa` | ❌ | 节拍/BPM/音高分析 |
| `peft` | ❌ | LoRA 库 |
| `datasets` / `pandas` | ❌ | 数据集工具 |

**应对方案（都不需要装新东西）**：

1. **音频解码与重采样 → 用系统 `ffmpeg`**（已装，`/usr/bin/ffmpeg`）。
   `ffmpeg -i in.flac -f f32le -ar 48000 -ac 2 -` 直接输出 48 kHz 立体声 float32 到管道。
   为什么用 ffmpeg 而不是 soundfile：我们要做的是 **44.1 kHz → 48 kHz 重采样 + 解 FLAC + 解 MP3**，ffmpeg 一条命令搞定且是 C 实现，比纯 numpy 快一个量级。
   （另外提醒：89% 的 `.mp3` 文件其实是 **FLAC**，用 `librosa.load` 之类只认 mp3 的 API 会翻车。）

2. **LoRA → 自己实现**（不依赖 peft）。
   理由有三：(a) 我们要同时支持"挂载推理"和"合并回 base"两种用法，自己写不到 200 行；(b) peft 对这种非标准双分支结构（`nar_self_attn` / `nar_mlp`）的 target_modules 匹配要绕弯；(c) 少一个依赖 = 少一个环境事故点。
   社区（Mothersuperior）的实现也是纯手写 LoRA，格式为 `lora_A [r, in]` / `lora_B [out, r]`，合并规则 `W += B @ A`。

3. **scipy / librosa 的用途（BPM、节拍、音高）→ 先不做**。
   这些只影响"风格标签里的 BPM 字段"，属于锦上添花。第一版可以用固定 BPM 或干脆不带 BPM。

4. **SheetSage2（转谱）→ 独立环境**。
   官方明确要求 SheetSage2 用 Python 3.10/3.11 + torch 2.8 + transformers 4.45.2 + `mir_eval`/`pretty_midi`/`mido` 等，与 YuE2 的依赖不兼容（上游 `YuE/docs/covers.md` 原话："Keep separate environments"）。权重已在本地缓存，需要新建一个 venv。

### 3.4 磁盘预算（这一条最容易被忽略，然后卡住整个流程）

```
$ df -h /ssd /ssd11 /home/luke
/dev/sdb    1.9T  1.6T  147G  92%  /ssd        ← ✅ 唯一有空间的地方
/dev/sdg1  458G  433G  1.3G 100%  /ssd11       ← ⚠️ 音乐本体已经塞满
/dev/sda5  916G  860G  9.8G  99%  /home/luke   ← ⚠️ 也快满了
```

**规则（写进代码里，不许违反）**：

- **所有训练产物（manifest、latent、feature、checkpoint、adapter）一律落 `/ssd/yue2_training/`**。
- `/ssd11/music/song_download` **只读**，那是 433 GB 的音乐本体，写进去会爆盘。
- `/home/luke` 只放代码（`m-a-p2` 本身）和最终的 adapter 导出（几十 MB）。

**空间估算**（这是选参数的关键）：

| 产物 | 单价 | 19,112 首全量 | 2,000 首子集 |
| --- | --- | --- | --- |
| VAE latent（fp16，25 Hz × 64 维） | 64×25×2 B/秒 ≈ 3.2 KB/秒 | **≈ 13.0 GB** | ≈ 1.4 GB |
| VAE latent（fp32） | 6.4 KB/秒 | ≈ 26 GB | ≈ 2.8 GB |
| MERT 特征（fp16，25 Hz × 1024 维） | 51.2 KB/秒 | **≈ 208 GB** ❌ | ≈ 21.5 GB ⚠️ |
| 语义 token（uint16，25 Hz） | 50 B/秒 | ≈ 0.2 GB | ≈ 0.02 GB |
| ABC 文本 | ~1 KB/首 | ≈ 20 MB | ≈ 2 MB |

> **注意 MERT 特征那一行**：全量 208 GB，`/ssd` 也放不下。所以 MERT 特征**不能全量预存**，只能在需要时按批现算，或者只对训练语义 tokenizer 头的那一小部分歌曲预存。这是第 5 章的一个关键约束。

### 3.5 模型与权重缓存

`HF_HOME` 指向 `/ssd/.cache/huggingface/hub`，已有：

| 仓库 | 大小 | 用途 |
| --- | --- | --- |
| `m-a-p/YuE2-3B` | 7.26 GB（BF16，`model.safetensors` 单文件） | 主模型 |
| `m-a-p/YuE2-Vae` | — | 声学 VAE（**含编码器**，见 §4.3） |
| `m-a-p/SheetSage2` | — | 乐谱转录（需独立环境） |
| `m-a-p/MERT-v2-FullSong` | — | MERT2 编码器（**只有编码器，没有量化头**，见 §5） |

全部**离线可用**，不需要联网下载模型。

---

## 4. 数据：我们有什么，以及怎么把它变成训练样本

### 4.1 结论

我们有 19,112 首真实歌曲、433 GB、约 1,130 小时音频，**并且 100% 能关联到歌名/艺人/专辑/歌单/分类元数据**。这批数据足够训练一个高质量的风格/音色 LoRA。缺的是**歌词**和**语义 token**。

### 4.2 音频现状（实测）

抽样 120 个文件（`ffprobe`）+ 全部文件的魔数嗅探结果：

| 项目 | 实测值 |
| --- | --- |
| 文件数 | 19,112（`.mp3` 扩展名）+ 1 个 `truncated.txt` |
| 实际编码 | **89% 是 FLAC**（文件头 `fLaC`），11% 是真 MP3 ⚠️ |
| 采样率 | 44.1 kHz（89%）/ 48 kHz（11%）—— 必须统一重采样到 48 kHz |
| 声道 | 100% 立体声 |
| 时长 | 中位数 208.5 秒；p25 = 171 s；p75 = 255 s；最长 655.8 s |
| 总时长 | ≈ **1,130 小时** |
| 单文件大小 | 平均 24 MB，最大 63.7 MB |
| 质量黑名单 | `truncated.txt` 列出 **85 个**被截断到 25–35 秒的文件（占 0.44%） |

**处理要点**：
1. 解码要同时支持 FLAC 和 MP3 → 用 ffmpeg（它按内容嗅探，不看扩展名）。
2. 统一重采样到 48 kHz（VAE 的原生采样率，48 kHz ÷ 1920 = 25 Hz 的 latent 帧率必须是整数，否则帧数对不齐）。
3. 剔除 `truncated.txt` 里的 85 个文件。
4. 时长 < 30 秒的丢弃（潜变量帧数太少，训不出东西）；> 5 分钟的切段处理。

### 4.3 元数据（100% 命中，很好）

元数据不在音乐目录里，而在 **`/ssd/music/song_info_database/`** 和 `/ssd/music/song_info_database2/`，是按歌单分类的 JSON 数组：

```json
{
  "song_name": "有个爱你的人不容易 (Live)",
  "artist": "那英",
  "album": "中国新歌声第二季 第14期",
  "song_id": 509781676,
  "playlist_id": 13688776647,
  "playlist_name": "伤感",
  "category": "伤感"
}
```

- 两个目录合计 **1,216,466** 首不重复歌曲。
- 我们本地的 19,112 个文件（文件名就是 `song_id.mp3`）**全部命中**（我实测 join：19112/19112）。
- 额外可用：`/ssd/music/male_female/artist_songs_relation.csv`（艺人性别：male/female）、`/ssd/music/top_1000_artists.json`（热度榜）。

> ⚠️ **注意**：本机**没有** `meta_download/` 和 `comments_download/` 目录（下载器 `botCrawl/download_music.py` 里定义了这两个常量，但实际没写出来）。所以**拿不到逐曲的评论、时长、收藏数等字段**，只能靠上面的 JSON 库。

### 4.4 风格标签（style）怎么造

YuE2 的提示词格式固定（`yue2/protocol.py:108-109`）：

```
{指令}
[Tags]
{style}
[Lyrics]
{lyrics}
```

`style` 是一行逗号分隔的英文短标签，官方推荐的顺序是 **language（语言）→ genre（流派）→ instruments（乐器）→ mood（情绪）→ key（调性）→ BPM（速度）**。官方例子（`examples/official_style.txt`）：

```
City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night
```

我们本地的歌单分类是中文的场景标签，一共 33 类。本地子集分布：

| 分类 | 数量 | 分类 | 数量 | 分类 | 数量 |
| --- | --- | --- | --- | --- | --- |
| 下午茶 | 3,961 | 古风 | 698 | 快乐 | 225 |
| 伤感 | 2,399 | 浪漫 | 666 | 安静 | 187 |
| 清新 | 1,614 | 地铁 | 649 | 思念 | 144 |
| 午休 | 1,569 | 夜晚 | 621 | 兴奋 | 143 |
| 放松 | 1,294 | 乡村 | 614 | 旅行 | 142 |
| 怀旧 | 1,089 | 学习 | 608 | 散步 | 124 |
| 后摇 | 712 | 治愈 | 456 | 民谣 | 40 |

**造标签的规则（第一版）**：

| 字段 | 来源 | 示例 |
| --- | --- | --- |
| `language` | 由 `song_name` / `artist` 的字符集判定（中文/日文/韩文/拉丁），再结合 `category=粤语` 修正 | `mandarin`、`japanese`、`korean`、`english` |
| `genre` | `category` 映射 + `playlist_name` 关键词匹配 | `pop`、`classical`、`folk`、`chinese traditional`、`post-rock`、`ambient` |
| `instrument` | `playlist_name` 关键词（如"钢琴曲""纯音乐""Chillhop""Bass"） | `piano`、`guitar`、`synth`、`drums` |
| `mood` | `category` 直接映射（伤感→melancholic、治愈→healing…） | `warm`、`sad`、`energetic` |
| `vocal` | `male_female` CSV 判定艺人性别 + 是否纯音乐歌单 | `female vocal`、`male vocal`、`instrumental` |
| `key` / `BPM` | 第一版留空（需要额外分析，见 §3.3 第 3 条） | — |

**必须承认的局限**：`playlist_name` 是网友手写的歌单名，噪声很大（例如"你毁了几首我爱听的歌"这种根本不是风格词）。所以规则要保守：**只在高置信度关键词命中时才加标签，宁可少加也不要加错**。加错标签比不加标签更糟——模型会真的去学那个错误风格。

### 4.5 歌词（本章最需要讨论的一块）

#### 4.5.1 为什么歌词重要

看 AR 半的输入结构就明白了：

```
[EOD] + tokenize("[Tags]\n{style}\n[Lyrics]\n{lyrics}\n") + [ABC_START] + ABC + [ABC_END, MUSIC_START] + 语义token + [MUSIC_END]
```

**歌词和乐谱是并排的条件**。模型在写 ABC 乐谱时，同时看着歌词——它需要知道"这一句唱什么字"，才能让旋律的音节数和歌词对齐（中文一字一音节、英文一音节对应一个元音）。

所以：
- **AR 半（音乐半）强依赖歌词**。没有歌词，模型只能写"没有词的歌"，和真实数据分布差很远。
- **NAR 半（声学半）不依赖歌词**。NAR 的条件是"已完成的语义 token 序列"，歌词只是作为 AR 部分被间接带进去。所以**只训 NAR 半，缺歌词不是阻塞项**。

#### 4.5.2 我们的现状

- 元数据里**没有歌词字段**（只有 song_name / artist / album / playlist_name / category）。
- 本机**没有 Whisper**（我检查过全部 conda 环境：`whisper`、`faster-whisper`、`funasr` 都不存在）。
- `/ssd/music/` 下**没有** `meta_download/`、`comments_download/`（本来可能有逐曲元数据，没落盘）。

#### 4.5.3 三种补法（按推荐度排序）

**方案 A：接外部歌词库（推荐，但需要联网）**

- **LRCLIB**（`lrclib.net`）：免费的逐曲歌词库，带时间戳，按 `artist + title` 检索，中英文都有。我们的元数据正好有 `song_name` + `artist`，匹配率预计不低（热门歌覆盖率很好，冷门歌会缺）。
- 也可以用 NetEase 的歌词接口（`botCrawl` 那个项目本来就在爬网易云，接口还在）。
- **风险**：匹配错误（同名歌、翻唱版）会引入脏数据；建议只采用**匹配置信度高**（歌名完全一致 + 艺人一致）的条目，其余标记为"无歌词"。
- **成本**：一次性的网络请求（19k 首 × 1 次），几十分钟。

**方案 B：本机装 Whisper 转写（可行，但要装环境 + 算力）**

- 用 `faster-whisper`（CTranslate2 版，比原版快）在 GPU 上转写，medium/large 模型。
- **成本**：19k 首 × 3.5 分钟音频，large 模型在 6 卡上大约 6–10 小时（大约 2–4× 实时）。可接受。
- **风险**：转写会错字（中文歌尤其），且**没有时间戳对齐**的话价值有限——我们其实需要的是"歌词文本 + 段落结构"，时间戳对齐是加分项。
- 缺点：需要装 `faster-whisper` + 下载模型（pip 和 HF 都需要网络，本机 pypi 镜像慢）。

**方案 C：放弃歌词，只训 NAR 半（零成本，强烈建议作为第一步）**

- NAR 半不需要歌词，风格标签就够。
- AR 半留到后面，等确定了歌词方案再上。

**方案 D：用歌名拼伪歌词（不推荐）**

- 把 `song_name` 套进 `[Verse]` 模板。风险是模型学到"歌名 → 旋律"这种无意义映射，属于噪声而非信号。

#### 4.5.4 建议

> **第一版（Phase 2）采用方案 C**：只训 NAR 半，不碰歌词。
> 同时在 Phase 0 阶段就把 **lyrics 字段在 manifest 里预留好**（值可以是空字符串），这样以后接上 LRCLIB 或 Whisper，不需要改数据管线。
> Phase 4 之后如果要做 AR 半，再补方案 A。

### 4.6 数据管线各阶段的产物

```
/ssd/yue2_training/
├── manifest.jsonl            Phase 0：一行一首歌的元数据 + 风格标签 + 质量标记
├── splits/
│   ├── train.jsonl           训练集（默认 95%）
│   └── val.jsonl             验证集（默认 5%，用于观测过拟合）
├── latents/
│   └── {song_id}.npy         Phase 1：[T, 64] float16 latent
├── semantic/                 （Phase 4，可选）
│   └── {song_id}.npy         [T] uint16 语义 token
├── abc/                      （Phase 4，可选）
│   └── {song_id}.abc         SheetSage2 转录的乐谱
├── features/                 （Phase 4，可选，只存子集）
│   └── {song_id}.npy         MERT 特征
├── runs/
│   └── {run_name}/           训练输出：checkpoint、日志、配置快照
└── adapters/
    └── {name}/               最终导出的 adapter
```

---

## 5. 核心缺口：语义 token（Semantic Token）

**这是整份文档里最需要你做决策的一章。** 前面的数据、GPU 都是可以绕路解决的工程问题；这一章是真正决定"我们能训什么"的问题。

### 5.1 结论

> **YuE2 官方没有公开"真实音频 → 语义 token"的编码器（量化头）。**
> 因此，我们手上这 19,112 首真实歌曲，**目前无法直接变成 YuE2 的语义 token**。
> 这直接限制了 AR 半（音乐半）的训练，但**不限制 NAR 半（声学半）**。

### 5.2 官方是怎么做的

从官方技术报告（arXiv:2609.33757）：

> "SheetSage2" produces the symbolic score. The "MERT2 tokenizer" produces semantic tokens. A VAE produces continuous acoustic latents.

MERT2 的结构：
- 前端：log-mel + ConvNeXt 下采样
- 主干：24 层 Conformer，**因果注意力**（layers 0–13）
- 瓶颈：**四层残差向量量化器（RVQ）**，**32,768** 个码字
- 输出：**每 40 ms 一个 ID**（即 25 Hz / 375 bit/s）
- 蒸馏目标：冻结的 MuQ + Qwen2-Audio-Instruct 表征融合后量化

也就是说：**语义 token 是 MERT2 编码器 + 一个学出来的量化头共同产生的**。

### 5.3 我们本地到底有什么（关键核实）

我逐个检查了缓存里的文件：

| 仓库 | 里面有什么 | 能否产生 YuE2 语义 token |
| --- | --- | --- |
| `m-a-p/MERT-v2-FullSong` | 876 个张量，**全部是编码器**（Conformer 层、ConvNeXt 下采样、layer norm） | ❌ **没有任何量化头** |
| `m-a-p/YuE2-Vae` | 声学 VAE，编码器输出 128 通道 → 切成 mean/scale → 64 维 latent | ❌ 这不是语义 tokenizer |

`YuE2-Vae/modeling_vae.py:491-496` 的 `encode()` 文档字符串里，官方自己写明了：

> *"This audio VAE is not the unreleased semantic audio tokenizer semantic tokenizer."*

另外还有一个**版本错配**：YuE2 用的语义 tokenizer 建立在 **MERT-v2-30s（因果分支）** 之上，而我们的 README 明确写着：

> *"The causal branch is used for YuE2 tokenization."*（`MERT-v2-FullSong/README.md`）

也就是说，即使我们想自己训量化头，**正确的因果版 MERT-v2-30s 我们本地根本没有缓存**（只有双向的 FullSong）。

> **给初学者的类比**：YuE2 是一个"读谱唱歌"的歌手，语义 token 就是"唱出来的中间结果"。官方只给了乐谱和"唱法说明"，唯独没给"怎么从录音里听出唱法"的听写器（量化头）。我们要用真实录音训练，就得自己造这个听写器。

### 5.4 三条补齐路线

#### 路线 1：自训语义 tokenizer（最彻底，最贵）

**思路**：蒸馏。用 YuE2 自己生成的歌当"标准答案"——因为 YuE2 生成一首歌时，**中间的语义 token 是已知的**，同时解码出的音频也是已知的。于是得到 (MERT特征, 语义token) 的配对数据，训一个"MERT特征 → 32768 码"的头。

**步骤**：
1. 用我们自己的 6 张卡批量生成 2,000–5,000 首 YuE2 歌曲（用不同风格标签，控制种子）。每首约 60–90 秒（AR 约 40 秒 + NAR 求解 32 步约 20 秒）。
   - 成本估算：3,000 首 × 75 秒 ÷ 6 卡 ≈ **10.4 小时**。
2. 保存每首的：语义 token（现成的）+ 解码音频（现成的）+ MERT 特征（现算）。
3. 训一个小型 Transformer 头（8 层，d=512，512 帧窗口，参考社区实现）：输入 MERT-v2-FullSong 第 20 层特征（逐轨实例归一化），输出 32,768 类的分类。
4. 用这个头给我们的 19,112 首真实歌曲打语义 token。
5. 有了真实语义 token，**AR 半 LoRA 和 NAR 半 LoRA 都能训**。

**成本**：约 3–5 天（含调试）。
**风险**：自蒸馏的 token 分布只覆盖"YuE2 能生成的音乐"，真实录音（尤其是真唱、复杂编曲）会有分布偏移；社区实测 top-1 准确率只有 16–19%（意思是有 80% 的 token 猜错）。但社区同时发现 **"近似码"渲染出来几乎一样**，所以准确率低不等于不能用——**前提是必须同时训一个 NAR LoRA 来吸收误差**。

#### 路线 2：下载社区已训好的语义 tokenizer 头（最快，有外部依赖）

社区 `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4`（以及 v5）公开了：
- `tokenizer_head_joint_v5.safetensors`：8 层、d512 的语义 token 编码头，**正好以 MERT-v2-FullSong 第 20 层特征为输入**（和我们缓存的模型对得上）。
- `nar_lora_joint_v5.safetensors`：配套的 NAR LoRA（rank 32）。

**成本**：几十分钟下载 + 半天适配。
**风险**：(a) 第三方权重，许可证 CC-BY-NC-SA，需要在项目里注明出处；(b) 他们的训练数据与我们不同，效果需要自己验证；(c) 版本耦合——他们的脚本基于某个 commit 的 `yue2` 包。

#### 路线 3：绕开语义 token（**第一版强烈推荐**）

**核心洞察**：NAR 半的**声学目标**来自 VAE latent，**不需要语义 token**。语义 token 只是 NAR 的**条件输入**。所以有两条路：

- **路线 3a（伪语义条件，推荐）**：对每首真实歌曲，先用 SheetSage2 转出 ABC 乐谱（**这个模型是公开的、本地已有缓存**），再让**冻结的 YuE2 AR** 拿着这首真实歌的乐谱生成语义 token，最后用这个语义 token 当条件、真实音频的 latent 当目标来训 NAR LoRA。
  - **优点**：条件分布和推理时完全一致（NAR 训练时和推理时都看到"一段语义 token"）；不需要第三方权重；不需要自训量化头。
  - **代价**：每首歌多两次预处理（SheetSage2 转谱 + AR 生成），**估算**约 1 分钟 GPU/首。2,000 首 ≈ 33 GPU 小时 ÷ 6 卡 ≈ 5.5 小时。
  - **残余误差**：AR 生成的语义 token 与真实录音不是逐帧对齐（AR 会加自己的编曲）。但"乐谱是真实的"，所以旋律/节奏大方向是对的。

- **路线 3b（纯文本条件，最省事）**：只给 NAR 文本条件（风格标签），把语义 token 整段屏蔽掉。官方代码里这叫 **codec dropout**，通过 `nar_cond_end` 参数实现（`yue2/modeling_yue2.py:676-682`）：
  ```python
  if nar_cond_end > 0:
      # Codec dropout: NAR only sees positions < nar_cond_end (text) + NAR
      text_k[0, 0, :nar_cond_end] = 1.0
      mask = (ar_q * ar_k * causal) + (nar_q * text_k) + (nar_q * nar_k)
  ```
  - **优点**：零额外预处理，立刻能训。
  - **代价（很严重）**：**推理时也必须用同样的屏蔽模式**，否则 NAR 会看到它训练时从没见过的语义 token。而一旦屏蔽语义 token，生成出来的音频就**不再跟随乐谱**——整首歌会变成"纯靠文字描述的器乐"。这对"翻唱/编辑乐谱"的能力是致命的。
  - **定位**：适合做"器乐音色微调"，不适合做"完整的歌 LoRA"。

### 5.5 建议

> **Phase 2（第一个 adapter）走路线 3a**：SheetSage2 转谱 + 冻结 AR 生成伪语义 token + 真实 latent 做目标。
> **Phase 4（可选）走路线 1**：自训语义 tokenizer，解锁 AR 半 LoRA。
> **路线 3b 作为训练时的"数据增广"混入**（比如 10–20% 的样本用纯文本条件），提升 NAR 对条件的鲁棒性，但**推理默认走标准模式**。

**风险提示**：路线 3a 依赖 SheetSage2，需要新建一个独立 python 环境并从 pypi 装 `mir_eval`/`pretty_midi`/`mido` 等包。**如果这一步装不动，就退回路线 3b**，并接受"只能做器乐风格 LoRA"这个降级。

---

## 6. 训练目标设计：NAR 声学 LoRA 完整方案

### 6.1 结论

我们要做的是：**用真实歌曲的声学 latent 当答案，训练 NAR 分支的 LoRA**，让 YuE2 渲染出来的声音更接近我们这批歌的音色、质感、混音风格。

这是一个 **flow matching（流匹配）** 训练任务，和 Stable Diffusion 的训练思路类似，但更简单：不需要 UNet 架构的复杂度，只需要"给一个带噪的 latent，让网络预测它应该往哪个方向动"。

### 6.2 数学（务必看懂这一节，后面所有代码都依赖它）

#### 6.2.1 训练时发生什么

设：
- `z0` = 真实音频经 VAE 编码得到的 latent，形状 `[T, 64]`（T = 帧数，25 Hz）
- `ε` = 标准高斯噪声，形状 `[T, 64]`
- `t ∈ [0, 1]` = 时间步，0 表示"干净的音频"，1 表示"纯噪声"

**第一步：造一个中间状态**

```
z_t = (1 - t) · z0 + t · ε
```

t=0 时 `z_t = z0`（干净），t=1 时 `z_t = ε`（纯噪声）。中间是线性插值。

**第二步：告诉模型"现在到哪一步了"，让模型预测速度**

网络输入 `(z_t, t, 条件)`，输出一个 `[T, 64]` 的张量 `v_θ`，含义是**速度**（velocity）——即"如果把 t 稍微减小一点，z 应该朝哪个方向移动"。

正确的答案是：

```
v* = ε - z0
```

（因为 `d/dt [(1-t)z0 + tε] = ε - z0`。）

**第三步：算损失**

```
L = mean_over_T_and_64( || v_θ(z_t, t, cond) - (ε - z0) ||² )
```

论文里的写法是 `1/(64|Z|) Σ ||·||²`，即对 64 个维度求和再对帧求平均。我们实现时用 `F.mse_loss(..., reduction="none").sum(dim=-1).mean()` 更贴近论文（对 64 维求和、对帧取平均）。这个归一化差异只影响学习率标度，不影响正确性，但**要保持一致**。

#### 6.2.2 推理时发生什么（必须和训练严格对齐）

推理用的是 ODE 求解（`yue2/nar.py:170-198`）：

```python
state = noise                       # 从纯噪声出发
dt = 1.0 / steps                    # steps = 32
for step in range(steps):
    t = 1.0 - step * dt             # t: 1 → 0
    first = self.velocity(state, raw_t=t_logit)
    mid   = state - first * (dt / 2)   # midpoint 法
    state = state - self.velocity(mid, raw_t=t_logit_mid) * dt
```

几个**必须和训练端一模一样**的细节：

| 细节 | 训练端必须怎么做 | 为什么 |
| --- | --- | --- |
| **时间参数的传递方式** | 传 `raw_t = logit(t)`，模型内部再做 `sigmoid` | `nar.py:186` 用 `torch.logit(torch.tensor(t))`。`modeling_yue2.py:603-606` 的 `_shift_t_value` 会做 `sigmoid(raw_t)`，而 `timestep_shift=1.0` 时 `sigmoid(logit(t)) = t`，**净效果等于直接把 t 喂进去**。但为了和推理逐位一致，训练端也走 `logit` 路径。 |
| **`logit` 的 clamp** | `clamp(-20, 20)` | `nar.py:186` 明确 clamp。t→0 或 1 时不加 clamp 会得到 ±inf。 |
| **flow 的方向** | `z_t = (1-t)z0 + t·ε`，目标 `v = ε - z0` | 与 `state = state - v·dt`（t 递减）自洽。 |
| **`timestep_shift`** | 用配置里的 1.0 | 如果改成别的值，训练/推理都会变，但必须一致。 |
| **latent 的尺度** | 用 VAE 编码出的 posterior **均值**（`sample=False`） | 推理时是从纯噪声 ODE 演化到数据分布，用采样会增加噪声。训练端用均值即可（`modeling_vae.py:512-516`）。 |

#### 6.2.3 时间步采样策略（一个容易被忽略但很重要的细节）

推理时，模型**只在 32 个特定的 t 值**被求值：
```
t ∈ {1 - k/32} ∪ {1 - k/32 - 1/64}   （步点 + 中点）
```

如果训练时在 t 上均匀随机采样，模型在那些"推理真正会用到"的 t 上可能训得不够精细。一个常见的改进是**按推理网格采样**：从上述集合里随机取一个，再加一点小抖动（比如 ±1/64 的均匀噪声），保证覆盖率的同时聚焦推理路径。

**建议**：50% 的样本从推理网格取（含抖动），50% 完全均匀采样（保证整个 [0,1] 区间都被覆盖，防止某些 t 区域完全没训过）。这两个比例是个起点，实际要靠验证集的 loss 曲线微调。

> 对比：扩散模型文献常用 logit-normal 分布采样 t。这里我们**不**照搬，因为我们的推理是确定性的固定网格，贴合网格更划算。

### 6.3 训练样本的完整构造

一条训练样本 = 一个 30 秒片段（默认），由这些部分组成：

```
[AR 部分]
  [EOD]                                  ← 文本分隔符（token id 151643）
  tokenize("Generate music with codec tokens from the given conditions.\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n")
  [ABC_START]                            ← 151847
  tokenize(abc_text)                     ← SheetSage2 转出的 ABC 乐谱（可选）
  [ABC_END, MUSIC_START]                 ← 151848, 151851
  semantic_tokens[i] + 151853             ← 语义 token（伪语义或真实语义），每个 = 值 + CODEC_OFFSET
  [MUSIC_END]                            ← 151852

[NAR 部分]
  [LATENT_START]                         ← 184621
  x_t[0], x_t[1], ..., x_t[T-1]          ← 这一段不是真的 token，隐状态被"注入"替换
  [LATENT_END]                           ← 184622
```

**关键理解**：NAR 部分的"位置"是占位用的。代码里（NAR 部分的参考实现 `modeling_yue2.py:636-666`）的做法是：

1. 先把整条序列（含 NAR 占位 token）过 `embed_tokens`；
2. 然后**把 NAR 位置的 embedding 整个替换掉**：
   ```python
   latent_hidden_nar = self.vae2llm(x_nar.unsqueeze(0))     # [1, N_nar, 2048]
   latent_hidden_nar = latent_hidden_nar + time_emb          # 时间步 embedding
   latent_hidden_nar = latent_hidden_nar + pos_emb           # latent 位置编码
   token_emb[0, nar_indices] = latent_hidden_nar[0]         # ← 替换
   ```
   其中 `x_nar` 在 LATENT_START / LATENT_END 两个位置上填 0，在内容帧上填 `z_t`。
3. 走完 28 层 Transformer（MoT 路由由 `ar_mask` 决定）；
4. 取出 NAR 内容位置的 hidden，過 `llm2vae` 得到 `v_pred`。

**位置编号有两个，容易搞混**：
- **RoPE 位置**：整条序列统一 `arange(S)`。AR 部分占 0..A-1，NAR 部分占 A..A+N_nar-1。**和推理端一致**（`nar.py:122`：`positions = arange(ar_length, ar_length + nar_length)`）。
- **latent 位置编码**：`arange(N_nar)`，从 0 重新开始，并 `clamp(max=max_latent_frames-1)`。**也和推理端一致**（`nar.py:124`）。

这两个必须一致，否则模型会学到错位的相对距离。

### 6.4 三种条件模式（训练时如何组样本）

| 模式 | NAR 能看到什么 | 训练目标 | 推理时对应 | 评价 |
| --- | --- | --- | --- | --- |
| **A. 完整条件** | 文本 + ABC + 语义 token + NAR 位置 | 真实 latent | 标准推理（`nar_cond_end=0`） | ✅ **主模式**。分布一致 |
| **B. 伪语义条件** | 同上，但语义 token 是"AR 为这首真实歌生成的版本" | 真实 latent | 标准推理 | ✅ **第一版就用这个**。有轻微内容错位 |
| **C. 纯文本（codec dropout）** | 只有文本（`nar_cond_end` 截断到文本末尾） | 真实 latent | 需要我们自己的 dropout 推理脚本 | ⚠️ 作为 10–20% 的增广，别当主力 |

> **模式 B 的误差在哪、为什么可以接受**：SheetSage2 转出的 ABC 是真实歌的旋律，AR 拿着它生成语义 token 时会尽量贴合，但 AR 终究是"重新表演一遍"，编曲细节会偏离真实录音。于是 NAR 被训练成"把一段**大致对应**的语义 token 渲染成**真实录音的音色**"。这个目标本身是良定义的：它教 NAR"真实录音的音色是什么样的"，同时保留了对语义 token 的响应能力。社区验证过"近似 token 渲染结果几乎一致"，说明这条路的误差容忍度是够的。

### 6.5 显存与超参估算

#### 6.5.1 显存账（单卡）

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| Base 权重（BF16） | 7.26 GB | `model.safetensors` 实际大小 |
| LoRA 参数（rank 32，196 个模块） | ≈ 35 M | 见下方计算 |
| AdamW 状态（fp32 参数 + 2 个动量） | ≈ 35M × 12 B ≈ 0.42 GB | |
| 梯度（fp32） | ≈ 0.14 GB | |
| 激活（开梯度检查点，30 s 片段） | ≈ 1–2 GB | 不开检查点约 4–6 GB |
| 注意力 mask（S≈1650，float32 additive） | 1650² × 4 B ≈ 11 MB | 每层复用 |
| **合计** | **≈ 9–10 GB** | 20 GB 卡余量充足 |

LoRA 参数计算（NAR 半，28 层，每层 7 个模块）：

| 模块 | 形状 | rank 32 的参数量 |
| --- | --- | --- |
| `nar_self_attn.q_proj` | 2048 → 2048 | 32×2048 + 2048×32 = 131 K |
| `nar_self_attn.k_proj` | 2048 → 1024 | 98 K |
| `nar_self_attn.v_proj` | 2048 → 1024 | 98 K |
| `nar_self_attn.o_proj` | 2048 → 2048 | 131 K |
| `nar_mlp.gate_proj` | 2048 → 6144 | 262 K |
| `nar_mlp.up_proj` | 2048 → 6144 | 262 K |
| `nar_mlp.down_proj` | 6144 → 2048 | 262 K |
| **每层小计** | | **≈ 1.25 M** |
| **28 层合计** | | **≈ 35 M** |

（AR 半的目标模块是完全对称的一组：`self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj`，参数量相同。）

#### 6.5.2 建议超参（第一版）

| 参数 | 建议值 | 依据 |
| --- | --- | --- |
| 片段长度 | 30 秒（T = 750 帧） | 显存宽裕；30 s 足够学习音色特征 |
| LoRA rank | 32 | 社区 NAR LoRA 用 32；够用且省显存 |
| LoRA alpha | 32（缩放系数 = alpha/r = 1.0） | 保守起点；要更强再升 alpha |
| 目标模块 | NAR 半 7 个模块 × 28 层 | 见 §2.3 |
| 学习率 | 1e-4，cosine 衰减，warmup 3% | 社区 NAR/AR LoRA 用 1e-4 ~ 5e-5 |
| 优化器 | AdamW（β=0.9/0.99，wd=0.01） | |
| 精度 | BF16 autocast + LoRA 参数保持 FP32 | 3.58B 全参数用 fp32 放不下 |
| 每卡 batch | 2（30 s 片段） | 20 GB 卡可到 4；先用 2 保守 |
| 全局 batch | 12（6 卡 × 2） | |
| 梯度累积 | 1（DDP 下不需要） | |
| 梯度裁剪 | 1.0 | flow matching 早期容易出尖峰 |
| 步数 | 2,000–4,000（2,000 首数据） | 约 1–2 个 epoch |
| checkpoint | 每 200 步存一次 + 每 500 步存"最佳" | |
| EMA | 可选，对 LoRA 参数做 EMA | 社区用 EMA 平滑权重 |
| 验证 | 每 200 步在 val 集上算 flow loss | |

#### 6.5.3 序列长度与显存的可调关系

| 片段 | T (latent 帧) | AR 部分 | 总长 S | 备注 |
| --- | --- | --- | --- | --- |
| 15 s | 375 | ~600 | ~980 | 显存最省 |
| **30 s** | **750** | **~900** | **~1650** | **默认** |
| 60 s | 1500 | ~1650 | ~3150 | mask 变成 40 MB，注意力开销上升 |

### 6.6 LoRA 的目标模块选择

| 方案 | 目标 | 学到的东西 | 参数量 |
| --- | --- | --- | --- |
| **NAR LoRA（推荐）** | `nar_self_attn.{q,k,v,o}_proj`、`nar_mlp.{gate,up,down}_proj` | 音色、混音、质感 | 35 M @ r=32 |
| NAR + `vae2llm`/`llm2vae` 全量微调 | 上面 + 两个 Linear（各 2048×64） | 额外 26 万参数，直接改 latent 接口 | +0.26 M |
| AR LoRA | `self_attn.{q,k,v,o}_proj`、`mlp.{gate,up,down}_proj` | 写谱/唱法的偏好 | 35 M @ r=32 |
| 两者都上 | — | 完整微调 | 70 M @ r=32 |

> **一个 MoT 特有的细节**：AR 半的 LoRA 会通过"AR 位置的 value"影响 NAR 的输出（因为 NAR 位置能看见 AR 位置）。反之 NAR 的 LoRA 不影响 AR。所以两半基本解耦，可以分别训练、分别评估。

---

## 7. 代码架构设计

### 7.1 目录结构

```
m-a-p2/training/
├── plan.md                     ← 本文档
├── README.md                   ← 怎么跑（速查版）
├── requirements.txt            ← 训练额外依赖（目前为空，见 §3.3）
├── configs/
│   ├── manifest.yaml           Phase 0 配置
│   ├── latents.yaml            Phase 1 配置
│   ├── nar_lora.yaml           Phase 2 训练配置
│   └── ar_lora.yaml            （后续）AR 训练配置
├── yue2train/                  ← Python 包（核心代码）
│   ├── __init__.py
│   ├── config.py               配置加载与校验
│   ├── utils.py                日志、随机种子、分布式初始化、计时
│   ├── paths.py                所有路径常量（唯一写死磁盘位置的地方）
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manifest.py         扫描 + 元数据 join + 风格标签合成
│   │   ├── style.py            category/playlist → style 标签的映射规则
│   │   ├── audio.py            ffmpeg 解码/重采样/切段
│   │   ├── latents.py          VAE 编码 → .npy
│   │   ├── transcription.py     SheetSage2 转谱（独立环境调用）
│   │   ├── semantic.py         AR 生成伪语义 token
│   │   └── dataset.py          torch Dataset（拼装 AR 部分 + latent）
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── lora.py             LoRA 层、注入、保存、加载、合并
│   │   ├── targets.py          目标模块选择规则
│   │   └── nar_forward.py      可微的、批量的 NAR 前向（训练专用）
│   ├── train/
│   │   ├── __init__.py
│   │   ├── flow_matching.py    噪声/t 采样、损失
│   │   ├── trainer.py          通用训练循环（优化器、调度器、EMA、日志、ckpt）
│   │   ├── train_nar.py        NAR LoRA 训练入口
│   │   └── train_ar.py         AR LoRA 训练入口（后续）
│   └── eval/
│       ├── __init__.py
│       ├── sample.py           用 adapter 生成音频做 A/B 对比
│       └── metrics.py          客观指标（latent MSE、频谱、响度）
├── scripts/                    ← 可直接执行的脚本（薄壳）
│   ├── 00_check_env.py         环境自检
│   ├── 01_build_manifest.py
│   ├── 02_extract_latents.py
│   ├── 03_transcribe_abc.py
│   ├── 04_build_semantic.py
│   ├── 05_train_nar_lora.py
│   └── 06_sample_with_adapter.py
└── adapters/                   ← 导出的 adapter（软链到 /ssd，仓库里只放小的）
```

### 7.2 为什么这样分

- **`scripts/` 是薄壳**：每个脚本只做"读配置 → 调 `yue2train` 里的函数 → 打印结果"。真正的逻辑在包里，方便测试和复用。
- **`modeling/nar_forward.py` 单独一个文件**：这是整个项目最关键的 200 行，必须能被人单独读懂和验证。
- **`paths.py` 集中管理路径**：因为 `/ssd` 空间紧张，我们需要一处地方能一眼看到"什么东西放在哪、占多少空间"。

### 7.3 数据流全景

```
/ssd11/music/song_download/*.mp3          （只读）
        │
        │ ① ffprobe 探测时长/采样率
        │ ② join /ssd/music/song_info_database/*.json
        │ ③ 合成 style 标签
        │ ④ 过滤 truncated.txt / 时长 < 30s
        ▼
   manifest.jsonl  ─────────────────────────────────────┐
        │                                               │
        │ ⑤ ffmpeg → 48k stereo float32 → 切 30s 片段   │
        │ ⑥ YuE2-Vae.encode() → [T,64] fp16             │
        ▼                                               │
   latents/{song_id}.npy                                │
                                                        │
        │ ⑦ SheetSage2 转谱（独立环境）                  │
        ▼                                               │
   abc/{song_id}.abc                                    │
        │                                               │
        │ ⑧ 冻结的 AR：text+abc → 语义 token             │
        ▼                                               │
   semantic/{song_id}.npy                               │
                                                        │
        ▼                                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Dataset: 拼装 AR 部分 token + 取 latent + 采 t/噪声      │
   └──────────────────────────────────────────────────────────┘
                                │
                                ▼
              NAR 前向（可微、批量、MoT 路由）
                                │
                                ▼
                   flow matching loss → LoRA 梯度
                                │
                                ▼
                  adapters/{name}/（safetensors）
                                │
                                ▼
              合并回 base 或挂载 → 采样对比
```

### 7.4 NAR 训练前向：逐步张量形状

以 batch=2、30 秒片段为例。设 AR 部分长度 A ≈ 900，latent 帧 T = 750：

| 步骤 | 代码位置 | 张量形状 | 说明 |
| --- | --- | --- | --- |
| 1. 拼 token 序列 | 我们自己的代码 | `[B, A+T+2]` | AR 部分 + LATENT_START + T 个内容位 + LATENT_END |
| 2. token embedding | `model.model.embed_tokens` | `[B, A+T+2, 2048]` | |
| 3. 造 `x_nar` | 我们自己的代码 | `[B, T+2, 64]` | START/END 位置填 0，内容位填 `z_t` |
| 4. `vae2llm` | `model.vae2llm` | `[B, T+2, 2048]` | 64 → 2048 |
| 5. 加时间 embedding | `model.time_embedder` | `[B, T+2, 2048]` | 所有 NAR 位置共享同一个 t |
| 6. 加 latent 位置 embedding | `model.latent_pos_embed` | `[B, T+2, 2048]` | 索引 `arange(T+2)` |
| 7. 替换 NAR 位置的 embedding | 我们自己的代码 | `[B, A+T+2, 2048]` | `inputs_embeds` 的 NAR 段被覆盖 |
| 8. 构造混合掩码 | 我们自己的代码 | `[B, 1, A+T+2, A+T+2]` | 见 §2.3 的四象限规则 |
| 9. RoPE | `model.model.rotary_emb` | cos/sin `[1, S, 128]` | `position_ids = arange(S)` |
| 10. 28 层 MoT | `model.model(...)` + `ar_mask` | `[B, S, 2048]` | 见下 |
| 11. 最终 RMSNorm | `model.model.norm` | `[B, S, 2048]` | |
| 12. `llm2vae` | `model.llm2vae` | `[B, S, 64]` | |
| 13. 取内容位 | 我们自己的代码 | `[B, T, 64]` | `v_pred` |

**第 10 步的 MoT 路由**（`modeling_yue2.py:256-295`）：每一层对**所有**位置同时算 AR 和 NAR 两套 Q/K/V，再用 `torch.where(ar_mask, ar_value, nar_value)` 逐位置选择。

> **一个值得做的优化**：官方实现为了代码简洁，两套投影都算全序列，其中一半是浪费的（NAR 位置的 AR 投影算完就被丢弃）。我们可以只对 AR 位置算 AR 投影、只对 NAR 位置算 NAR 投影再拼起来——数学上完全等价，但能省掉接近 40% 的计算量。
> **但必须做数值验证**：实现之后要用同一个输入，对比官方 `DecoderLayer.forward` 与我们的快路径，断言输出在 1e-4 量级内一致。这个验证脚本要留在 `training/tests/` 里。

### 7.5 LoRA 实现要点

#### 7.5.1 结构

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base                       # 冻结
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))   # 初始化为小随机
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))  # 初始化为 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * delta
```

- `lora_B` 初始化为 0 → 训练开始时 adapter 的效果为 0，**不会破坏 base 模型的初始行为**。这是 LoRA 的标准做法，很重要。
- `lora_A` 用 Kaiming 均匀初始化（`nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))`）。

#### 7.5.2 保存格式

用 safetensors，键名规则（对齐社区格式，方便互操作）：

```
layers.{i}.nar_self_attn.q_proj.lora_A   [r, in]
layers.{i}.nar_self_attn.q_proj.lora_B   [out, r]
```

同时在一个 JSON 里记录元信息：

```json
{
  "base_model": "m-a-p/YuE2-3B",
  "base_revision": "8f24312f187763b9854adeec87e437e2b03bbff1",
  "target": "nar",
  "rank": 32, "alpha": 32, "scaling": 1.0,
  "modules": ["nar_self_attn.q_proj", "..."],
  "training": {"steps": 2400, "lr": 1e-4, "data": "manifest-v1", "segments_s": 30},
  "code_version": "training/plan.md@v1"
}
```

> **必须记录 base 模型版本**。社区经验很明确：**在哪个权重上训的 adapter，就只能在那个权重上用**。换权重后 LoRA 的语义就变了。我们的 `scripts/06_sample_with_adapter.py` 会在加载时校验 base revision，不匹配就报错并提示。

#### 7.5.3 三种使用方式

1. **合并（推荐给最终交付）**：把 `B@A*scale` 加到 base 的 `weight` 上，得到一份新的完整权重。推理端零额外开销，代码完全不用改。
2. **挂载（推荐给 A/B 对比和调试）**：加载 base 后替换对应模块为 `LoRALinear`，可以随时调节 `scaling`（0 = 关，1 = 全强度，0.75 = 社区常用的"轻一点"）。
3. **导出给外部工具**（ComfyUI / yue2.cpp 等）：社区格式需要把分开的 `q/k/v` 合并成 `qkv_proj`、把 `gate/up` 合并成 `gate_up_proj`。**这是 Phase 4 之后的可选工作**，第一版不做，但格式设计上留好接口。

### 7.6 分布式训练与数据加载

- **框架**：`torchrun` + `torch.distributed`（NCCL）。不引入 accelerate 的复杂抽象，保持可控。
- **启动方式**：
  ```bash
  torchrun --nproc_per_node=6 -m yue2train.train.train_nar --config configs/nar_lora.yaml
  ```
- **每卡**：一个进程，各自加载完整模型（7.3 GB）+ 自己的 LoRA 副本。
- **数据**：`DistributedSampler`，`shuffle=True`，`seed` 固定，`drop_last=True`。
- **梯度同步**：`model` 里只有 LoRA 参数 `requires_grad=True`，DDP 会自动只同步这些参数（其余冻结参数不参与 all-reduce）。
- **数据加载**：用 `DataLoader(num_workers=6)`，worker 读 `.npy`（mmap 模式），非常轻，不需要额外的进程池。

> **注意**：NCCL 在这种"模型大、参数少"的场景下通信量很小（LoRA 梯度只有 35M 参数 × 4 B = 140 MB/步），IB 带宽足够。

### 7.7 梯度检查点（Gradient Checkpointing）

`YuE2ForCausalLM` 虽然声明了 `supports_gradient_checkpointing = True`，但 `Backbone.forward`（`modeling_yue2.py:418-443`）**并没有真正调用 `torch.utils.checkpoint`**——也就是说 `model.gradient_checkpointing_enable()` 在这个模型上**是空操作**。

因此我们要在自己的前向里手动实现：

```python
for i, layer in enumerate(model.model.layers):
    if self.gradient_checkpointing and self.training:
        x = torch.utils.checkpoint.checkpoint(
            layer, x, cos, sin, None, i, attention_mask, ar_mask, None,
            use_reentrant=False,
        )
    else:
        x = layer(x, cos, sin, None, i, attention_mask, ar_mask, None)
```

30 秒片段不开检查点也能跑（显存约 4–6 GB 激活），但 60 秒片段建议开启。**建议默认开启**，换取更大的 batch。

### 7.8 训练循环骨架

```
for step in range(total_steps):
    batch = next(dataloader)                       # [B] 个样本
    t, noise = sample_t_and_noise(batch)            # 见 §6.2
    z_t = (1 - t) * z0 + t * noise                  # [B, T, 64]
    v_target = noise - z0                          # [B, T, 64]
    v_pred = nar_forward(model, batch, z_t, t)      # [B, T, 64]
    loss = ((v_pred - v_target) ** 2).sum(-1).mean()
    loss.backward()
    clip_grad_norm_(lora_params, 1.0)
    optimizer.step(); scheduler.step(); optimizer.zero_grad()
    if step % 200 == 0: log(loss, lr, grad_norm)
    if step % 200 == 0: save_checkpoint()
    if step % 500 == 0: validate()                  # val 集 flow loss
```

### 7.9 断点续训

每次 checkpoint 存：
- LoRA 参数（safetensors）
- 优化器状态（AdamW 的 exp_avg / exp_avg_sq）
- scheduler 状态
- 当前 step / 总步数 / RNG 状态
- 配置快照（yaml 原文）
- 训练日志（jsonl）

恢复时：加载 → `optimizer.load_state_dict` → `scheduler.load_state_dict` → 设置 `step` → 继续。

---

## 8. 分阶段实施计划

每个阶段都给出：**目标 / 输入 / 输出 / 验收标准 / 预计耗时 / 主要风险**。

### Phase 0 — 造清单（Manifest）

| 项目 | 内容 |
| --- | --- |
| **目标** | 把 19,112 个音频文件变成一份带元数据、风格标签、质量标记的结构化清单 |
| **输入** | `/ssd11/music/song_download/*.mp3`、`/ssd/music/song_info_database*/*.json`、`truncated.txt`、`male_female/artist_songs_relation.csv` |
| **输出** | `/ssd/yue2_training/manifest.jsonl`（每行一首）+ `splits/{train,val}.jsonl` |
| **验收** | ① 行数 = 19,112 − 85（黑名单）− 时长不合格；② 每行有 `song_id / path / title / artist / category / style / duration / quality`；③ `style` 字段非空且不含中文残留（除 `chinese traditional` 这类固定译名）；④ train/val 划分按 `song_id` 哈希，**同一首歌不能跨集**（避免泄漏） |
| **耗时** | 约 10–15 分钟（主要是 ffprobe 19k 次 + 读 1.2M 行 JSON） |
| **风险** | ① JSON 库很大（单文件 91 MB），要流式处理，内存峰值控制在 2 GB 内；② ffprobe 并发不能太高，否则磁盘 IO 被打满（建议 32 并发） |

**关键设计决定**：
- **train/val 必须按 song_id 划分，且同一首歌的多个片段不能跨集**。因为我们的训练样本是"30 秒片段"，同一首歌会产生多个片段，如果随机划分片段，会有 99% 的验证 loss 是虚假的（模型见过同一首歌的另一段）。**必须按歌划分。**
- 清单里预留 `lyrics` 字段（默认空字符串）和 `semantic_path` / `abc_path` 字段（默认 null），后续阶段填。

### Phase 1 — 抽取 VAE latent

| 项目 | 内容 |
| --- | --- |
| **目标** | 把真实音频变成模型能吃的 `[T, 64]` latent |
| **输入** | manifest + `/ssd11/music/song_download/*.mp3` |
| **输出** | `/ssd/yue2_training/latents/{song_id}.npy`（float16） |
| **验收** | ① 随机抽 5 首，用 `YuE2-Vae.decode()` 解回音频，和原音频对比（听感上应基本一致，可算波形相关系数/谱距离做粗检）；② latent 帧数 = `floor(采样点数 / 1920)`；③ 全量跑完后统计总帧数，和 25 Hz × 总时长对得上（误差 < 1%） |
| **耗时** | 6 卡并行，约 **1 小时**（19,112 首）；单卡约 6 小时 |
| **风险** | ① VAE 编码器在大片段上显存吃紧 → **按 30 秒切片编码**；② 边界效应：切段边缘因感受野会有轻微瑕疵 → 训练时随机加 0.1–0.5 秒 padding 裁掉；③ 44.1 kHz 文件重采样质量 → 用 ffmpeg 的 `soxr` 高质量重采样 |

**实现要点**：
- VAE 编码器必须 `decoder_only=False` 加载（`modeling_vae.py:389`：`if not self.decoder_only: self.encoder = ...`；`:443-445` 也会按 `decoder_only` 过滤要加载的张量）。
- VAE 权重**必须是 FP32**（`modeling_vae.py:459-461` 会校验，非 FP32 直接报错）。
- 编码用 `torch.inference_mode()`。
- 编码时的 batch：一次喂 1 段 30 秒（`[1, 2, 1440000]`），显存约 2–3 GB。

### Phase 2 ★ — NAR 声学 LoRA 训练（**主体**）

| 项目 | 内容 |
| --- | --- |
| **目标** | 产出第一个可用的 adapter |
| **前置** | Phase 0、1 完成；若走伪语义路线，还需 3a、3b |
| **输入** | manifest + latents + （可选）abc + （可选）semantic |
| **输出** | `/ssd/yue2_training/adapters/nar_lora_v1/` |
| **验收** | ① 训练 loss 平滑下降，val loss 也在下降（不只看 train）；② val flow loss 相比 base 模型有明确改善（base 的 loss 用同一个函数在未加 LoRA 时测）；③ 用 `06_sample_with_adapter.py` 生成 10–20 首音频，人工 A/B 对比 base；④ adapter 文件 < 200 MB |
| **耗时** | 2,000 首数据、2,400 步、6 卡 → **约 2–3 小时** |
| **风险** | 见表 §10 |

**子阶段 2a：转谱（SheetSage2）** — 仅伪语义路线需要
- 新建独立环境（Python 3.11 + torch 2.8 + transformers 4.45.2 + mir_eval/pretty_midi/mido）。
- 每首转谱约 2–5 秒（GPU）→ 2,000 首约 1.5 小时（6 卡并行）。
- 产出 `abc/{song_id}.abc`（melody-only，与上游 cover 流程一致）。

**子阶段 2b：伪语义 token**
- 冻结 AR，输入 = `[Tags]`+style + （可选）ABC，`cot="melody"`，生成到 `MUSIC_END`。
- 每首约 40–60 秒（**估算**：AR 自回归约 100 token/s，一首歌约 5,000 个语义 token；上游没有公布吞吐基准，`sampling.py:147` 会在运行时输出 `output_tps`，**以我们自己的实测为准**）→ 2,000 首约 30 GPU 小时 ÷ 6 卡 ≈ **5 小时**。
- 产出 `semantic/{song_id}.npy`（uint16，存的是 0–32767 的**原始码值**，用的时候再加 `CODEC_OFFSET`）。

**子阶段 2c：训练**
- 见 §6、§7。

### Phase 3 — 采样与评测

| 项目 | 内容 |
| --- | --- |
| **目标** | 证明 adapter 有用（或者证明没用，然后改） |
| **输入** | adapter + base 模型 + 一批固定提示词 |
| **输出** | `output/ab_test/{base,adapter}/*.flac` + 一份对比表 |
| **验收** | ① 固定 20 个提示词 × 3 个种子，base 与 adapter 各生成一遍；② 客观指标：latent 空间的 MSE、谱包络距离、响度差；③ 主观：至少听 5 组，判断"音色是否更接近目标风格"；④ 检查**没有副作用**：歌词/旋律是否还正常（LoRA 太强会毁掉内容） |
| **耗时** | 约 1 小时（单卡） |
| **风险** | LoRA 强度过大导致内容崩坏 → 提供 `scaling` 滑杆，导出 0.5 / 0.75 / 1.0 三档分别听 |

### Phase 4（可选，长期）— 语义 tokenizer 与 AR 半

- 4a：批量生成 2,000–5,000 首 YuE2 歌曲 → (音频, 语义token) 配对数据（10.4 小时 / 6 卡）
- 4b：训语义 tokenizer 头（8 层 d512）
- 4c：用头给真实歌曲打语义 token
- 4d：AR LoRA 训练（需要歌词，走 §4.5.3 方案 A 接 LRCLIB）

---

## 9. 评测与验收：怎么判断 adapter 有没有用

这一节很重要，因为**"loss 降了"不等于"生成的歌变好了"**。

### 9.1 三层评测

**第 1 层：数值（自动，秒级）**
- 验证集上的 flow matching loss（base vs adapter）。这是最直接的信号。
- 训练/验证 loss 的差值（过拟合指标）。

**第 2 层：音频客观指标（自动，分钟级）**
| 指标 | 测什么 | 期望 |
| --- | --- | --- |
| 谱包络距离 | adapter 生成 vs 真实录音的 mel 谱差异 | 变小 = 音色更接近 |
| 响度 / 动态范围（LUFS, LRA） | 混音风格 | 接近目标分布 |
| 高频噪声底（6–16 kHz 的"沙沙"声） | 社区报告过的常见退化 | 不升高 |
| 低频占比（sub 20–60 Hz） | 社区报告过的"轰鸣"退化 | 不升高 |

**第 3 层：主观 A/B（不可替代）**
- 固定提示词、固定种子，base 和 adapter 各生成一版。
- 听三件事：**音色是否更接近目标风格**、**内容有没有被破坏**（旋律、歌词、结构）、**有没有新引入的瑕疵**。

### 9.2 社区已知的典型退化（来自公开的微调经验）

我们在设计评测指标时参考了社区公开记录（Ruach Studio、Mothersuperior 的公开报告），他们踩过的坑我们要提前防：

| 退化现象 | 原因 | 我们的对策 |
| --- | --- | --- |
| 高频人声"起沙" | LoRA 学到了压缩/母带处理的伪影 | 训练集做**响度归一化**，或者显式加"不做母带"的数据；监控高频噪声底指标 |
| 训练过久后开始"背歌" | 小数据集 + 步数过多 | 保留 val 集，loss 回升就停；步数上限设在 2–4 个 epoch |
| 语义 token 重复率上升 | 条件分布偏移 | 统计生成结果的 token 重复率（社区实测从 9.3% 降到 3.0% 是好结果） |
| 内容崩坏（旋律/歌词错乱） | LoRA 强度过大 | 提供 scaling 滑梯，优先选 0.5–0.75 强度 |
| base 权重被污染 | 训练时误更新了 base | 冻结参数 + 训练后校验 base 权重哈希 |

---

## 10. 风险登记册

| # | 风险 | 概率 | 影响 | 对策 | 触发信号 |
| --- | --- | --- | --- | --- | --- |
| R1 | 语义 token 无解，AR 半训不了 | **高** | 中 | Phase 2 先做 NAR 半；Phase 4 自训 tokenizer | — |
| R2 | SheetSage2 独立环境装不上（pypi 慢） | 中 | 中 | 退回纯文本条件（codec dropout）路线，接受"器乐 LoRA"定位 | pip 超时/编译失败 |
| R3 | `/ssd` 空间被 latent 吃满 | 中 | 高 | 严格按 §3.4 的体积表规划；latent 用 fp16；MERT 特征不预存 | `df -h /ssd` 低于 50 GB |
| R4 | 训练/推理分布不一致（条件模式对不上） | **中高** | **高** | 严格按 §6.2 的一致性检查表逐条核对；写一个"数值对齐测试" | 生成的音频明显跑调/无内容 |
| R5 | LoRA 学到录音瑕疵（底噪、母带、编码伪影） | 中 | 中 | 训练前做**质量过滤**（至少剔除截断文件）；监控 §9.1 的客观指标 | 生成的音频比 base 更"脏" |
| R6 | 显存 OOM（24 GB 卡 + 60 s 片段 + 评估） | 中 | 中 | 片段默认 30 s；开梯度检查点；先在单卡跑通再上 6 卡 | CUDA OOM |
| R7 | 过拟合（2,000 首训 4 个 epoch） | 中 | 中 | 按歌划分 val；监控 val loss；步数上限 | val loss 回升 |
| R8 | DDP 卡住/超时（NCCL + 56 核 CPU 抢资源） | 低 | 中 | 设置 `OMP_NUM_THREADS=2`；数据加载 `num_workers=4`；用 `torchrun` 而非 `mp.spawn` | 进程 hang |
| R9 | 依赖环境损坏（往 `yue2` 环境装包） | 低 | **高** | **训练依赖一律装在 `yue2` 环境之外的独立 venv，或只用现有包** | `import yue2` 失败 |
| R10 | 底座模型版本变了，adapter 失效 | 低 | 中 | adapter JSON 里记 base revision；加载时校验 | revision 不匹配报错 |
| R11 | 磁盘上的 `truncated.txt` 之外还有别的坏文件 | 中 | 低 | 解码失败的文件在 manifest 里标记并跳过（不中断整批） | ffmpeg 报错 |

---

## 11. Day-1 具体动作清单

如果今天就要动手，按这个顺序做（**每一步都能独立停下来**）：

### 上午

1. **[10 分钟]** 建目录骨架 + 写 `paths.py`：
   ```bash
   mkdir -p /ssd/yue2_training/{latents,abc,semantic,runs,adapters,splits}
   mkdir -p /home/luke/distributed_machine_learning/m-a-p2/training/{configs,yue2train,scripts}
   ```
2. **[30 分钟]** 写 `scripts/00_check_env.py`：检查 GPU、模型缓存、依赖、磁盘剩余量，输出一个体检报告。**先确认地基没问题再写业务代码。**
3. **[1 小时]** 写 `yue2train/data/style.py` + `manifest.py` + `scripts/01_build_manifest.py`，先只跑 100 首验证输出格式，再跑全量。
4. **[30 分钟]** 人工检查 20 条 manifest：风格标签是否合理、有没有明显脏数据。

### 下午

5. **[1 小时]** 写 `yue2train/data/audio.py`（ffmpeg 解码 + 重采样 + 切段）和 `latents.py`（VAE 编码）。
6. **[30 分钟]** 跑 20 首，**把 latent 解回音频听一下**——这是整个流程里最重要的 sanity check。如果解码出来是噪音，说明 VAE 编码理解错了，后面全白搭。
7. **[1 小时]** 全量抽 latent（后台跑，约 1 小时），同时开始写 `modeling/lora.py` 和 `modeling/nar_forward.py`。
8. **[30 分钟]** 写数值对齐测试：同一输入下，我们的 NAR 前向 vs 官方 `nar_velocity`，断言 `max|Δ| < 1e-4`（bf16 下放宽到 1e-2）。**这一步过了，才说明我们对模型的理解是对的。**

### 第二天

9. 写 `train/flow_matching.py` + `train/trainer.py` + `train/train_nar.py`。
10. 用 **200 首**的数据跑 200 步，确认 loss 在下降、不 NaN、不 OOM。
11. 搭 SheetSage2 独立环境，开始转谱（为伪语义路线做准备）。

---

## 附录 A：术语表

| 术语 | 通俗解释 |
| --- | --- |
| **AR（自回归）** | 一个字一个字地写。从左到右，每个 token 只能看前面的。 |
| **NAR（非自回归）** | 一次性同时算出所有位置。这里指声学 latent 的并行生成。 |
| **flow matching（流匹配）** | 一种扩散模型变体。训练时给模型一个"半成品"，让它预测"往哪走"，然后和正确答案的差距做损失。 |
| **latent（潜变量）** | 压缩后的中间表示。1920 个采样点 → 1 帧 64 维数字。 |
| **VAE** | 编解码器。把波形压缩成 latent，再解压回波形。 |
| **token / 词表** | 文本的"积木块"。YuE2 的词表有 184,704 块，其中 32,768 块专用于语义 token。 |
| **MoT（混合专家）** | 一层里有两套权重，一套管 AR，一套管 NAR。 |
| **RoPE（旋转位置编码）** | 给位置信息加进注意力的方法。这里 AR 和 NAR 共用一套位置编号。 |
| **KV cache** | 生成时缓存已算好的键值，避免重复计算。**训练时用不到**。 |
| **GQA（分组查询注意力）** | 16 个查询头共享 8 组键值头，省显存。 |
| **LoRA** | 在原权重旁边加一条低秩旁路，只训这条旁路。 |
| **adapter** | LoRA 训练产物的统称（一个小文件）。 |
| **DDP** | 数据并行：多张卡各算一部分数据，然后同步梯度。 |
| **梯度检查点** | 少存中间激活、回头重算，用时间换显存。 |
| **EMA** | 参数滑动平均，抑制震荡。 |
| **bf16 / fp16 / fp32** | 不同的数值精度。bf16 动态范围大，训练首选。 |
| **scaling（LoRA 强度）** | 把 LoRA 的贡献乘以一个系数。0 = 关闭，1 = 全强度。 |
| **codec dropout** | 训练时故意遮住语义 token，强制模型学会用别的条件。 |
| **MERT2** | 一个音乐理解模型（632M），YuE2 的语义 token 就是从它来的。 |
| **SheetSage2** | 一个乐谱转录模型，把录音转成 ABC 乐谱。 |
| **LRCLIB** | 一个免费的逐曲歌词库。 |

---

## 附录 B：关键代码位置索引

| 想了解什么 | 去哪里看 |
| --- | --- |
| 模型配置（层数、维度、上下文长度） | `site-packages/yue2/modeling_yue2.py:61` `YuE2Config` |
| 单层结构（两套 attention + 两套 MLP） | `site-packages/yue2/modeling_yue2.py:227` `DecoderLayer` |
| MoT 路由（`torch.where` 合并） | `site-packages/yue2/modeling_yue2.py:256-295` |
| NAR 训练前向参考实现 | `site-packages/yue2/modeling_yue2.py:609` `nar_velocity` |
| 混合注意力掩码的四种关系 | `site-packages/yue2/modeling_yue2.py:668-685` |
| codec dropout（`nar_cond_end`） | `site-packages/yue2/modeling_yue2.py:676-682` |
| 时间步 shift | `site-packages/yue2/modeling_yue2.py:603` `_shift_t_value` |
| NAR 专属模块定义 | `site-packages/yue2/modeling_yue2.py:465-474` |
| 推理端 NAR 求解器（32 步 midpoint） | `site-packages/yue2/nar.py:170-198` `CachedNAR.solve` |
| 推理端 AR 前缀构造 | `site-packages/yue2/nar.py:34` `song_chunks` |
| 采样时的词表限制 | `site-packages/yue2/sampling.py:26` `distribution` |
| Token 常量（EOD/ABC/MUSIC/CODEC/LATENT） | `site-packages/yue2/protocol.py:7-11` |
| 提示词格式 | `site-packages/yue2/protocol.py:108-109` `SongRequest.text` |
| 官方生成配置（采样参数） | `site-packages/yue2/protocol.py:42` `GenerationConfig` |
| VAE 编码器 | `site-packages/yue2/modeling_vae.py:491` `encode` |
| VAE 解码（分块，带 halo） | `site-packages/yue2/modeling_vae.py:539` `decode_tiled` |
| 流水线 API | `site-packages/yue2/pipeline.py:121` `YuE2Pipeline` |
| 现有推理入口 | `m-a-p2/scripts/infer_yue2.py` |
| 风格标签范例 | `m-a-p2/examples/official_style.txt` |

---

## 附录 C：参考资料

**官方**
- YuE2 技术报告：arXiv:2609.33757（`https://arxiv.org/abs/2609.33757`）
  —— 训练数据 346,000 小时；`L = 0.25·L_AR + L_FM`；`z_t = (1-t)z0 + tε`；目标速度 `ε - z0`；28 层 MoT；混合掩码四规则；条件丢弃机制；语义 tokenizer 来自 MERT2 + 32768 码 RVQ。
- 模型仓库：`m-a-p/YuE2-3B`、`m-a-p/YuE2-Vae`、`m-a-p/SheetSage2`、`m-a-p/MERT-v2-FullSong`
- 代码仓库：`https://github.com/multimodal-art-projection/YuE`（本地副本 `/home/luke/distributed_machine_learning/YuE`）
  —— `docs/generation.md`（风格/歌词格式）、`docs/covers.md`（转谱流程与独立环境要求）、`skills/yue2-music/references/abc-editing.md`（ABC 方言规范）

**社区（用于了解可行性与已知陷阱，未在本地复现）**
- Ruach Studio 的 YuE2-3B-RS-FW 微调报告：LoRA 双半（music rank 160 / sound rank 320）、lr 5e-5、EMA、3×3090、SheetSage2 乐谱 + Whisper/LRCLIB 歌词；已知退化现象（高频"沙"、低频轰鸣）
- `Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4/v5`：自研语义 tokenizer 头（8 层 d512，MERT-v2-FullSong 第 20 层特征 → 32,768 码）+ 配套 NAR LoRA（rank 32）；top-1 准确率 16–19%；"近似码渲染几乎一致"
- `AcademiaSD/YuE2-3B-NF4-for-LoRA-Training`：量化底座做 LoRA 的思路
- DiffSynth-Studio / ComfyUI 的 YuE2 适配（了解社区工具链生态）

**本项目内部**
- `m-a-p2/README.md`、`m-a-p2/RUN.md`：现有推理流程
- `botCrawl/raw_data.txt`：音乐数据的抓取与下载流程（本文档元数据部分的来源）
- `/ssd/music/song_info_database*/`：逐曲元数据（1,216,466 首）
- `/ssd/music/male_female/artist_songs_relation.csv`：艺人性别

---

*文档结束。实施过程中如果发现任何与本文档不符的事实，请优先相信实测结果，并回来更新这份文档。*
