# YuE2 入门讲义：从一段文字到一首歌

> 面向完全零基础的读者。不要求你会深度学习，不要求你听过"flow matching"或"VAE"。
> 目标：读完这份文档，你能看懂论文里的每一个术语，能读懂仓库里每一个函数的输入和输出，
> 并且能自己跑出一首歌，还能说出**为什么**它会跑出这个结果。

---

## 目录

| 章节 | 内容 | 适合什么时候读 |
|---|---|---|
| [第 0 章](#第-0-章这份文档怎么用) | 这份文档怎么用、符号约定 | 一开始 |
| [第 1 章](#第-1-章我们要做的事) | 任务的输入输出到底长什么样 | 一开始 |
| [第 2 章](#第-2-章来龙去脉yue2-想解决什么) | 来龙去脉：YuE2 想解决什么问题 | 想明白"为什么" |
| [第 3 章](#第-3-章整体架构一张图看懂) | 整体架构，一张图看懂 | 想看全局 |
| [第 4 章](#第-4-章逐层拆解五个阶段) | 五个阶段逐层拆解 | 想看细节 |
| [第 5 章](#第-5-章abc-记谱法小课堂) | ABC 记谱法小课堂 | 读不懂 score.abc 时 |
| [第 6 章](#第-6-章代码地图) | 代码地图：哪个文件干什么 | 要动手改代码时 |
| [第 7 章](#第-7-章精读每个模块的输入与输出) | 精读：每个模块的输入与输出 | 要动手改代码时 |
| [第 8 章](#第-8-章本项目的脚本层) | 本项目 `scripts/` 这一层 | 要跑起来时 |
| [第 9 章](#第-9-章磁盘上的文件到底是什么) | 磁盘上的文件到底是什么 | 想复现/校验时 |
| [第 10 章](#第-10-章四种玩法与三种模式) | 四种玩法与三种模式 | 想试试不同效果时 |
| [第 11 章](#第-11-章亲手跑一遍) | 亲手跑一遍 | 立刻 |
| [第 12 章](#第-12-章调参与常见问题) | 调参与常见问题 | 遇到问题时 |
| [第 13 章](#第-13-章术语表) | 术语表 | 随时查 |
| [第 14 章](#第-14-章论文数据附录) | 论文数据附录 | 想深挖时 |

---

## 第 0 章 这份文档怎么用

### 0.1 三句话摘要

如果只读三句话，请读这三句：

1. **YuE2 把"写歌"拆成了三步**：先写出一张人类可读的乐谱（ABC 记谱法），再把这张谱子展开成密集的"语义 token"，最后把语义 token 渲染成声音波形。
2. **这三步由同一个 3.58B 的模型主干（一个叫 Mixture-of-Transformers 的结构）完成**，前两步是自回归的（一个 token 一个 token 地写），第三步是并行的（整首歌一起算）。
3. **这三条路互相不冲突**：同一个 checkpoint，既能"从零创作"，也能"照着你改过的谱子重新演奏"，还能"把别人唱过的歌重新演绎一遍"。

### 0.2 一些约定

- **代码位置**：`/home/luke/distributed_machine_learning/m-a-p2`
- **官方包**：`yue2_infer-0.1.5-py3-none-any.whl`（安装后是 `site-packages/yue2/`）
- **本项目脚本**：`scripts/*.py`（是对官方包的一层命令行封装）
- **论文**：*YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality*，arXiv:2609.33757，m-a-p / 香港科技大学，56 页技术报告
- 本文引用代码时写成 `文件:行号`，例如 `nar.py:243`，你可以直接打开对照着看。

> 阅读建议：**第 1–4 章是"思想"，第 5–7 章是"实现"，第 8–12 章是"动手"**。
> 如果你只想赶紧跑一首歌，可以直接跳到第 11 章；但如果你想真正理解这个模型在干什么，请按顺序读。

---

## 第 1 章 我们要做的事

### 1.1 先看一个真实的结果

本项目已经跑好了一首歌，产物在 `output/my_song/`。它的 `score.abc` 长这样（截取开头）：

```
X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D
% intro
V: Vocal
z24z4"D"z4|"D"z32|"F#m7"z32|"G"z32|
V: Ins
Z|z4f4z2f2z4f6g2z2a6|z4e4z2e2z4e6f2z2g6|...
```

而它的输入只有两段文字：

```
style:  mandarin pop, female vocal, warm acoustic guitar, soft drums, clear diction, mid tempo
lyrics: [Verse] 晚风轻轻吹过窗前 / 你留下的笑还在昨天 / ...
```

**输入是文字，输出是一张真谱子 + 一段 155 秒的 48 kHz 立体声音频。** 这就是全部。
中间发生了什么，是这份文档要讲清楚的事。

### 1.2 输入与输出，先说死

**输入（三样东西）**

| 名字 | 类型 | 说明 | 例子 |
|---|---|---|---|
| `style` | 字符串 | 风格描述，像一串逗号分隔的标签 | `mandarin pop, female vocal, warm piano` |
| `lyrics` | 字符串 | 歌词正文，用 `[Verse]` `[Chorus]` 等方括号分段 | 见上 |
| `seed` | 整数 | 随机种子。**同样的 seed + 同样的输入 = 同样的歌** | `831001` |
| `cot` | 字符串 | "要不要先写谱"（三种模式，见第 10 章） | `full` / `melody` / `off` |
| `abc` | 字符串（可选） | 你自己写的谱子（翻唱/改编时用） | 见 `examples/melody.abc` |

**输出（一次生成的完整产物）**

| 文件 | 内容 | 你会拿它干嘛 |
|---|---|---|
| `audio.flac` | 48 kHz / 立体声 / 24 bit PCM 波形 | **这就是成品**，直接听 |
| `audio.mp3` | 同一份音频的 mp3 副本 | 双击方便试听 |
| `score.abc` | 模型写的 ABC 乐谱（旋律 + 和弦 + 曲式） | 拿去改、拿去给别人看 |
| `semantic.npy` | 语义 token 序列（int32） | 中间产物，一般不用管 |
| `latent.npy` | 声学 latent 矩阵（float32，形状 `[帧数, 64]`） | 想换解码器时用 |
| `request.json` | 这次请求的完整参数 | 复现时要用 |
| `config.json` | 实际生效的采样配置 + 权重指纹 | 复现时要用 |
| `result.json` | 耗时、时长、是否被截断、每个文件的 sha256 | 校验用 |

### 1.3 关键问题：为什么不能"一步到位"？

小白最自然的疑问是：现在的大语言模型不就能一次写出一整首歌吗？为什么要多绕一圈写谱子？

答案是**信息密度不匹配**。

- 一首歌的最终形态是波形。48 kHz 立体声，3 分半的歌 = 约 1500 万个数。
- 一张 ABC 谱子，同样的歌 = 大约 1700 个字符、1332 个 token。
- 中间的"语义 token"层 = 3888 个 token。

论文的原话是：*"一首成品歌是一个稠密的声学对象，但创造它并不需要是一个稠密的决策。"*
（*A finished song is a dense acoustic object, but creating it need not be one dense decision.*）

换句话说：**让模型直接从"文字"跳到"波形"，等于让它在 1500 万维空间里做一次押韵式的猜谜；而让它先在几千维的乐谱空间里把"写什么"这件事想清楚，再去做"怎么录"这件事，每一步都轻松得多。**

---

## 第 2 章 来龙去脉：YuE2 想解决什么

### 2.1 两条老路，各有各的遗憾

在 YuE2 之前，AI 作曲大致分两派：

**符号派（Symbolic）**

- 代表：Music Transformer、SymPAC 等。
- 做法：把乐谱（音高、时值、和弦）编码成离散 token，用语言模型的方式一个一个预测。
- 优点：**生成的东西是可读的、可编辑的、可验证的**。你拿到的是一张真谱子。
- 遗憾：**止步于"谱"**。你得到的还是一份乐谱，要变成录音还得靠人用音源库/DTM 演奏。

**音频派（Audio）**

- 代表：MusicLM、MusicGen、后来的各种"文生音乐"服务。
- 做法：把音频压成 latent，用扩散模型或 flow matching 直接生成波形。
- 优点：**一步到位**，出来就是成品录音。
- 遗憾：**作曲过程是隐式的**。你没法指着成品说"第 17 小节的副歌转调太平了"——因为模型内部根本没有"第 17 小节"这个概念。

YuE2 的立论是：**这两条路应该合并，而不是二选一。**

### 2.2 核心洞见："渐进式音乐承诺"

论文把创作过程描述成一串**逐渐加码的承诺（progressive musical commitment）**：

```
文字 + 歌词  →  承诺"要表达什么"
      ↓
     乐谱      →  承诺"旋律、和声、节奏、曲式"
      ↓
 表演/音频表征 →  决定"什么时候唱、怎么咬字、什么音色、什么制作"
```

每一层都把一件"模糊的事"变成"确定的事"，但把其余的留给下一层发挥。YuE2 把这三层**全部放进一个模型里**，用一条流水线串起来。

于是乐谱同时扮演两个角色：

1. **它是一个中间生成状态**（模型自己先写谱，再照着谱子唱）；
2. **它是一个面向用户的可编辑接口**（你改谱子，模型就照改后的谱子重新唱，但保留你没动的东西）。

第 2 点是白盒（white-box）音乐生成的真正含义：**你可以打开机器的"草稿"来改。**

### 2.3 一个模型，三种玩法

论文把三种操作形式化成三个式子，注意它们的差别只在"乐谱从哪来"：

| 操作 | 乐谱来源 | 公式 |
|---|---|---|
| **创作** Create | 模型自己生成 | $s \sim p(s \mid y)$，再 $c,z \sim p(c,z \mid y,s)$ |
| **编辑** Edit | 用户改过的 | $\tilde{s}$，再 $\tilde{c},\tilde{z} \sim p(c,z \mid y,\tilde{s})$ |
| **翻唱** Cover | 转谱系统（SheetSage2）转出来的 | $\hat{s}$，再 $c',z' \sim p(c,z \mid y',\hat{s})$ |

其中 $y$ = 文字+歌词，$s$ = 乐谱，$c$ = 语义 token，$z$ = 声学 latent。

**同一个 checkpoint，三种玩法，不需要额外训练。** 这是 YuE2 最有意思的设计之一。

### 2.4 论文的三个主张，以及支撑证据

论文不是空谈。三个核心主张，各有实验支撑（数字来自论文表格）：

**主张一：写了谱，质量更高。**

同一个 checkpoint，开/关"乐谱规划"，让专业音乐人盲听对比：

| 评价维度 | 有规划 | 无规划 | 平局 | p 值 |
|---|---:|---:|---:|---:|
| 总体质量 | **49.3%** | 34.6% | 16.1% | 0.0070 |
| 音乐性 | **45.0%** | 29.4% | 25.6% | 0.0141 |

> **这正是本项目 `--cot full` 和 `--cot off` 的区别。** 你可以在本机上跑两遍对比着听。

**主张二：合成一个模型，比"语言模型 + 扩散模型"两段式更好。**

论文做了一个公平对照：LM 1.7B + DiT 1.7B（合计 3.4B）vs YuE2 的 3.58B 单体，数据量、语义 tokenizer、VAE、规划方式、候选选择规则全部对齐：

| 维度 | 统一 MoT | LM + DiT | p 值 |
|---|---:|---:|---:|
| 总体质量 | **53.4%** | 35.6% | 0.0084 |
| 音质 | **48.5%** | 29.6% | 3.06e-4 |

**主张三：整体质量到了"前沿"水平。**

在 WildSongBench（192 个真实 prompt，15 个流派桶，中英混合）上：

| 系统 | SongBench Global Avg ↑ | SongBench Musicality ↑ |
|---|---:|---:|
| Suno v5 | 6.8721 | 5.9918 |
| Mureka 9 | 6.9377 | **6.0488** |
| **YuE2**（2 候选） | 6.7316 | 5.9075 |
| **YuE2**（best-of-8） | **6.9632** | **6.2666** |

配合专家盲听：YuE2 best-of-8 **以 57.3% : 30.5% 击败 Suno v4.5**；对 Suno v5 是 40.4% : 39.9%（基本打平）；但 **Suno v6 以 59.3% : 31.6% 赢下**。

> **请把这句话记住**：论文自己很克制地说过，*"这些结果并不确立普遍的指标优越性或人类偏好"*，*"最高均值之间的微小差距并不构成统计显著性"*。所以"frontier quality"（前沿质量）的准确含义是"**与被评估的商用系统同处第一梯队**"，不是"全面碾压"。

### 2.5 顺带一提：论文还训练了两个"辅助模型"

论文不只发布了 YuE2。为了能从**没有配套乐谱的录音**里学出这套流程，它还训练了：

- **MERT2**（音乐表征模型，632M）：在 70 万小时音频上预训练，在 15 个 MARBLE 音乐理解基准中的 14 个上超过此前最好结果。它同时充当了 YuE2 的**语义 tokenizer**——把音乐压缩成 25 Hz、32768 词表的离散 token（375 bit/s）。
- **SheetSage2**（音频转谱模型）：从录音里转出 ABC 乐谱（旋律 + 和弦 + 曲式），在 lead-sheet 转录对比中领先 15 个"基准×指标"组合中的 12 个。它就是**翻唱功能的乐谱来源**。

> 在本项目里，如果你只想"文字 → 歌"，这两个模型**完全不需要下载**。只有做 cover（翻唱）流程才需要 `m-a-p/SheetSage2`。

---

## 第 3 章 整体架构：一张图看懂

### 3.1 数据流全景

```
                    ┌─────────────────────────────────────────┐
   style (文字)  ───►│                                         │
   lyrics (歌词) ───►│   AR 阶段（自回归，一个 token 一个 token）│
                    │                                         │
                    │   ① 乐谱规划 ──► ABC 记谱法（文本）      │
                    │                     "C" E2 G2 A2 ...     │
                    │                                         │
                    │   ② 语义展开 ──► 32768 词表的稠密 token   │
                    │                     25 Hz, 3888 个       │
                    └────────────────────┬────────────────────┘
                                         │  条件（三样全给）
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   NAR 阶段（非自回归，整首歌并行）        │
                    │   flow matching，32 步中点法              │
                    │   声学 latent：25 Hz × 64 维              │
                    └────────────────────┬────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │   VAE 解码器（Oobleck，全卷积）           │
                    │   64 维 latent @ 25 Hz ──► 波形 @ 48 kHz  │
                    └────────────────────┬────────────────────┘
                                         ▼
                              audio.flac（立体声，48 kHz）
```

注意图里最重要的一点：**AR 阶段产生的东西（谱子 + 语义 token）会同时作为 NAR 阶段的输入条件**。也就是说，声学合成这一步看着"旋律 + 和弦 + 语义"，它自己并不"创作"任何音乐结构，它只负责**表演**。

### 3.2 五个阶段一句话版

| 阶段 | 名字 | 类型 | 输入 | 输出 |
|---|---|---|---|---|
| 0 | 构造 prompt | 纯字符串拼接 | style, lyrics, cot | 一段形如 `指令\n[Tags]\n...\n[Lyrics]\n...\n` 的文本 |
| 1 | 乐谱规划 | **AR** | prompt | ABC 文本 + token id 序列 |
| 2 | 语义展开 | **AR** | prompt + ABC | 32768 词表内的语义 token 序列 |
| 3 | 声学合成 | **NAR** | prompt + ABC + 语义 token | `[帧数, 64]` 的 float32 latent |
| 4 | 音频解码 | 卷积网络 | latent | 48 kHz 立体声波形 |

### 3.3 关键数字速查表

**主干模型（`m-a-p/YuE2-3B`）**

| 项目 | 数值 | 出处 |
|---|---|---|
| 参数量 | ≈ 3.58 B | 论文 Table 13 / `config.json` |
| 层数 | 28 | `modeling_yue2.py:82` |
| 隐藏维度 | 2048 | `modeling_yue2.py:81` |
| 前馈维度 | 6144 | `modeling_yue2.py:85` |
| 注意力头 | 16 个 Q 头 / 8 个 KV 头（GQA），head_dim 128 | `modeling_yue2.py:83-85` |
| 上下文长度 | 24,576 token | `protocol.py:12` |
| RoPE theta | 1,000,000 | `modeling_yue2.py:88` |
| 归一化 / 激活 | RMSNorm / SwiGLU | `modeling_yue2.py:125, 216` |
| 词表大小 | 184,704 | `protocol.py:12` |
| 训练精度 | bfloat16 | `pipeline.py:216` |

**词表分区**（`protocol.py:7-12`）—— 这个表很重要，后面到处要用：

| 区间 | 内容 | 常量名 |
|---|---|---|
| `0 – 151642` | 普通文本 BPE token（151,643 个，就是 Qwen 词表） | — |
| `151643 – 151850` | 208 个特殊 token | — |
| `151643` | 特殊区的**第 1 格**。它在 tokenizer 里其实是个空字符串，但协议拿它当"请求结束"标记用 | `EOD` |
| `151847 / 151848` | 特殊区里的 `<abc>` / `</abc>`，即 ABC 段起止标记 | `ABC_START` / `ABC_END` |
| `151851 / 151852` | 音乐（语义 token）段起止标记。**这两个不在 tokenizer 里**，是协议自己加的 | `MUSIC_START` / `MUSIC_END` |
| `151853 – 184620` | 语义 token 编码区（32,768 个） | `CODEC_OFFSET` + `CODEC_SIZE` |
| `184621 – 184623` | latent 段起止/填充标记（仅训练路径用） | `LATENT_START/END/PAD` |

> 那 208 个特殊 token 到底是怎么排的？看 `tokenization_yue2.py:16-21` 就一目了然：
>
> ```python
> specials = ["", "", "", "<R>", "<S>", "<X>", "<mask>", "<sep>"]   # 8 个
> specials += [f"<extra_{i}>" for i in range(200)]                   # 再加 200 个
> specials[204:206] = ["<abc>", "</abc>"]                           # 顶掉其中 2 个
> ```
>
> 也就是 **3 个空串 + 5 个具名符号 + 200 个 `<extra_i>` = 208 个**（`<abc>` / `</abc>` 是从 `<extra_i>` 里
> "插队"进来的，所以总数不变），编号正好从 151643 连到 151850。
> 实测印证：`151847` 解码出来是 `<abc>`，`151848` 是 `</abc>`，
> `151849` / `151850` 则是没被顶掉的 `<extra_198>` / `<extra_199>`。
>
> **初学者只要记住一条**：普通文本 ≤ 151642，语义 token 在 151853 往后，
> 中间这一小段（151643 ~ 151852）是给"结构性标记"准备的，
> 它们的 id 都写在 `protocol.py:7-12` 里，认常量名比记数字靠谱得多。

**解码器（`m-a-p/YuE2-Vae`）**

| 项目 | 数值 |
|---|---|
| 架构 | Oobleck（全卷积残差），6 个下采样块 |
| 输入 / 输出 | 48 kHz 立体声 |
| 压缩比 | 1920× |
| latent 帧率 | 25 Hz |
| latent 维度 | 64 |

**采样默认配置**（`protocol.py:46-47`）

| | ABC 阶段 | 语义阶段 |
|---|---|---|
| temperature | 0.7 | 1.0 |
| top_p | 0.9 | 0.95 |
| top_k | 30 | 100 |
| repetition_penalty | 1.005 | 1.2 |
| penalty_window | 100 | 50 |
| min_tokens | 32 | 200 |
| max_tokens | 4096 | 9000 |

> 注意两阶段的**性格差异**：写谱要**保守、规整**（温度 0.7、几乎不重复），写语义要**发散、有细节**（温度 1.0、惩罚重复 1.2）。

**flow matching**

| 项目 | 数值 | 出处 |
|---|---|---|
| 步数 | 32 | `protocol.py:48` |
| 求解器 | 二阶中点法（midpoint） | `protocol.py:49`、`nar.py:170` |
| 时间参数化 | `logit(t)` clamp 到 [-20, 20]，再 sigmoid（shift=1.0 时为恒等） | `nar.py:186`、`modeling_yue2.py:603` |
| 噪声来源 | 整首歌的噪声在 CPU 上用 FP32 **一次性抽完**，再按分块切片 | `nar.py:45-46` |

---

## 第 4 章 逐层拆解：五个阶段

### 4.1 阶段 0：把文字变成 prompt

**位置**：`protocol.py:108-109`，`SongRequest.text()`

模型看到的"问题"其实是一段拼出来的文本。完整格式：

```
{模式指令}

[Tags]
{style}

[Lyrics]
{lyrics}
```

三种 `cot` 对应的指令不同（`protocol.py:14-18`）：

| `cot` | 指令原文 |
|---|---|
| `off` | `Generate music with codec tokens from the given conditions.` |
| `melody` | `Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions.` |
| `full` | `Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.` |

注意 `full` 的指令里有一个**很关键的信息**："chord-annotated"。它明确告诉模型"这次要写带和弦符号的谱子"。这就是同一个 checkpoint 能切换三种行为的原因——**靠指令切换，不靠换权重**。

最终送进 tokenizer 的 token 序列长这样（`protocol.py:115-126`）：

```
[EOD] + tokenizer.encode(prompt) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
```

`[EOD]` 放在最前面，是给模型的一个"起跑信号"（官方叫法就是 end-of-text，兼做起点标记，有点反直觉但就是这样）。

### 4.2 阶段 1：AR 生成 ABC 乐谱

**位置**：`pipeline.py:255-269`（`plan()`）→ `sampling.py:56-154`（`generate_tokens`）

这是标准的大语言模型式生成：

1. **prefill（预填充）**：把前缀（prompt + 指令）一次性喂进模型，得到第一个 token 的 logits。
2. **循环**：每一步取 logits → 加掩码 → 采样 → 得到下一个 token → 喂回去。
3. **直到**采到 `ABC_END`（`151848`）为止。

**"加掩码"这一步是 YuE2 特有的、也是最重要的一个细节**（`sampling.py:26-53`）：

```python
allowed = torch.full_like(scores, float("-inf"))
if phase == "abc":
    allowed[..., :EOD] = 0          # 写谱阶段：只允许普通文本 token
else:
    allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0   # 语义阶段：只允许语义 token
allowed[..., end] = 0               # 结束标记永远允许
```

**这保证了模型在写谱时不可能吐出一个语义 token，在写语义时不可能吐出一个文本 token。** 两个词表互不串台。你可以把它理解成"给模型戴了两副互斥的嘴"。

另外还有一条：`scores[..., end] = -torch.inf`（`sampling.py:38`），当 `step < min_tokens` 时**禁止提前结束**。写谱最少 32 个 token，语义最少 200 个——防止模型刚起步就写个"完"字。

**输出**：`ids`（token id 列表）+ `timing`（耗时、token 数、tok/s 等）+ `truncated`（是否撞到 max_tokens 被硬截断）。

**`truncated` 很重要**：如果模型在 4096 个 token 处被砍断，谱子就是断的，歌也跟着断。检查 `result.json` 里的 `truncated.abc` 和 `truncated.semantic`。

### 4.3 阶段 2：AR 生成语义 token

**位置**：`pipeline.py:271-283`（`generate_semantic`）

这一阶段和阶段 1 唯一的不同是：

- 前缀换成了 `plan.prefix`（prompt + 完整的 ABC token）；
- 采样参数换成语义阶段那套（温度 1.0）；
- 输出 token 被减去 `CODEC_OFFSET`（`pipeline.py:283`），还原成 0–32767 的语义 id。

**语义 token 是什么？** 它是 MERT2 这个音乐理解模型对音频的"离散压缩表示"。每 40 ms 一个（也就是 25 Hz），每个 token 从 32768 个码本条目里选一个。它编码了"这一瞬间大概是什么声音/什么演奏动作"，但**不编码精确的相位和波形**。

粗略地说：**语义 token 是"演奏说明"，latent 才是"录音本身"。**

**CFG（Classifier-Free Guidance）在这一阶段才真正起作用。** 回忆一下 `cfg_scale` 的负分支（`protocol.py:129-138`）：

| `cot` 模式 | 负分支（unconditional）长什么样 |
|---|---|
| `off` | 只有指令（"Generate music with codec tokens..."），没有标签没有歌词 |
| `full` / `melody` | **同样的指令 + 同样的 ABC token**，只是没有 `[Tags]` 和 `[Lyrics]` |

然后在采样时（`sampling.py:109`）：

```python
logits = unconditional + cfg_scale * (conditional - unconditional)
```

`cfg_scale = 1.0` 时，结果就是 conditional 本身（默认行为）。调大它，等于"更用力地把注意力推向文本条件、远离无文本条件"。

> **重要**：因为符号模式的负分支**保留了 ABC**，所以调大 `cfg_scale` 不会破坏你已经定好的旋律和和弦——它只改变"文字对表演的影响强度"。这是符号规划带来的一个额外好处。

### 4.4 阶段 3：NAR + flow matching 合成声学 latent

**位置**：`nar.py` 全文 + `modeling_yue2.py:603-700`

这是整个系统里最"不像语言模型"的一段。

#### 4.4.1 为什么需要 NAR（Non-Autoregressive）？

AR 生成语义 token 时，3888 个 token 要一个一个生成，每个都要跑一遍 28 层 Transformer → 35.8 秒。

但声学 latent 有 3888 帧 × 64 维，如果也一个一个生成，会慢到无法接受。

**NAR 的思路：整首歌一起算。** 3888 帧在一次前向里同时更新 32 遍。代价是每一步都要重算，所以总时间是 24.4 秒——比 AR 的 35.8 秒还快。

#### 4.4.2 flow matching 在干什么？

用最朴素的话说：

> 我们想知道"最终的 latent 长什么样"（记作 $z_0$），但我们只会从一团随机噪声出发。
> 于是我们训练一个网络 $v_\theta(x_t, t)$，让它在"当前状态 $x_t$、当前时间 $t$"下，告诉你"该往哪个方向走、走多快"。
> 训练时我们造一条直线：$x_t = (1-t)z_0 + t\epsilon$，其中 $\epsilon$ 是纯噪声。目标速度就是 $\epsilon - z_0$。
> 推理时从 $t=1$（纯噪声）出发，用 32 步 ODE 积分走到 $t=0$（干净的 latent）。

论文的损失函数（`L_FM`，Eq. 6）：

$$\mathcal{L}_{\mathrm{FM}} = \frac{1}{64|\mathcal{Z}|}\sum_{i\in\mathcal{Z}}\left\|v_\theta(z_t,t,y,s,c)_i - (\epsilon_i - z_{0,i})\right\|_2^2$$

总损失是（Eq. 7）：

$$\mathcal{L} = 0.25\,\mathcal{L}_{\mathrm{AR}} + \mathcal{L}_{\mathrm{FM}}$$

> 注意 **0.25 这个权重**：AR 的损失被**降权**了。这说明训练时模型更看重"把声音做对"，而不是"把 token 预测准"。这个数字是论文 Table 里的一个设计选择。

#### 4.4.3 MoT 的混合注意力掩码

MoT（Mixture-of-Transformers）的核心是：**每一层有"两套专家"，用哪套由位置类型决定。**

```
                        被看的位置
                   ┌──────────┬──────────┐
                   │  AR 位   │  NAR 位  │
  ┌────────────────┼──────────┼──────────┤
  │  查询是 AR 位   │ 因果 ✓   │  屏蔽 ✗  │
  │  查询是 NAR 位  │ 全看 ✓   │ 双向 ✓   │
  └────────────────┴──────────┴──────────┘
```

翻译成人话：

- **AR 位看不到 NAR 位**——写谱/写语义的时候，绝不能偷看未来的声音（否则就是作弊，会破坏自回归的因果性）。
- **NAR 位能看所有 AR 位**——声学生成时，谱子和语义 token 是它的"谱子"，当然要全看。
- **NAR 位之间双向可见**——同一首歌的 3888 帧互相之间没有先后因果关系（不像文字那样"前一个词决定后一个词"），音频帧是同时发生的。

代码里，`modeling_yue2.py:245-304` 的 `DecoderLayer.forward` 用 `torch.where(mask, ar_out, nar_out)` 来路由：AR 位置走 `self_attn` + `mlp`，NAR 位置走 `nar_self_attn` + `nar_mlp`。**两套 Q/K/V 投影、两套 MLP、两套 LayerNorm，完全独立**（`modeling_yue2.py:232-243`）。

#### 4.4.4 推理实现里的一个坑（重要）

论文/训练路径里的混合掩码是**显式构造一个 S×S 矩阵**实现的（`modeling_yue2.py:668-685`）：

```python
mask = (ar_q * ar_k * causal) + (nar_q * ar_k) + (nar_q * nar_k)
```

但**推理时用的不是这条路**。推理走 `nar.py` 的 `CachedNAR`，它用**结构化**的方式实现同样的语义：

1. `_prefill()`（`nar.py:132-149`）：先把 AR 前缀跑一遍**因果注意力**，把每层的 K/V 缓存下来。
2. `velocity()`（`nar.py:151-167`）：NAR 帧的 Q/K/V 用 `nar_self_attn` 算出来，然后 K/V 拼成 `[AR 缓存 | NAR 自己]`，**不加因果掩码**地做一次注意力。

因为 AR 那一侧的注意力本来就是因果的、且 NAR 的 K/V 根本不参与 AR 前缀的计算，"AR 看不到 NAR" 是**结构性成立**的，不需要掩码。

> **初学者常在这里绕晕**：为什么 `modeling_yue2.py` 里有一套 `nar_velocity`，推理却不用它？
> 答：`nar_velocity` 是**训练时**用的完整实现（一次前向算完整首歌，不分块）；
> `nar.py::CachedNAR` 是**推理时**用的分块实现（为了塞进 24,576 的上下文窗口）。
> 两者语义等价，实现策略不同。

#### 4.4.5 分块（chunking）与显存

上下文窗口是 24,576 个位置，而 **AR 侧和 NAR 侧抢的是同一批格子**。

这里有个容易看漏的关键点：**每一块不只是带自己的声学帧，它还带自己那一段语义 token**。
看 `nar.py:47` 是怎么拼出一块的：

```python
Chunk(
    prefix + [value + CODEC_OFFSET for value in codec[a:b]] + [MUSIC_END],   # AR 侧
    noise[a:b],                                                              # NAR 侧
)
#  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  AR 侧 = 整条前缀 + 本块的语义 token + MUSIC_END
#                                    ^^^^^^^^^^^^^^^^^^^^^ NAR 侧 = 本块的声学帧
```

假设这一块有 `frames` 帧（帧和语义 token 是**一一对应**的，所以它对应的语义 token 也是 `frames` 个），那么这一整块占的位置是：

| 侧 | 装了什么 | 位置数 |
|---|---|---|
| AR | 整条前缀 `prefix_tokens` | `prefix_tokens` |
| AR | 本块的语义 token | `frames` |
| AR | `MUSIC_END`（段尾标记） | 1 |
| NAR | 本块的声学帧 | `frames` |
| NAR | 首尾各补一帧（`LATENT_START` / `LATENT_END`，见 `nar.py:114`、`nar.py:156`） | 2 |

合计 = `prefix_tokens + frames + 1 + frames + 2` = **`prefix_tokens + 2 × frames + 3`**。

它必须塞进 24,576 里，于是：

```python
frames ≤ (context - prefix_tokens - 3) // 2
size = min((context - prefix_tokens - 3) // 2, CONTEXT)
```

> **所以那个 `// 2` 不是拍脑袋写的**，是"**每一帧要占两个格子**"直接除出来的：
> 一半格子给这一帧的语义 token，一半格子给它自己的声学帧。
> （`protocol.py:141-145` 算块大小，`nar.py:117-118` 负责在真正用的时候再校验一次。）

- `-3` 里的三个位置：`MUSIC_END`（AR 侧结尾）+ `LATENT_START`、`LATENT_END`（NAR 侧首尾）。
- 前缀 1535 个 token 时，每块最多 (24576 − 1535 − 3) // 2 = 11519 帧。
- 一首 3888 帧的歌 → **只有 1 块**，不用分块。

**分块的一个漂亮性质**：噪声是**整首歌一次性抽完**的（`nar.py:45-46`），然后按历史切点切片。所以**怎么分块都不影响结果**——同一个 seed 永远得到同一首歌。

```python
generator = torch.Generator(device="cpu").manual_seed(int(seed))
noise = torch.randn((len(codec), 64), dtype=torch.float32, device="cpu", generator=generator)
```

#### 4.4.6 "一次 AR prefill，32 步复用"

`CachedNAR._prefill()` 只跑一次（`nar.py:127` 触发），把 AR 前缀的 K/V 缓存住。因为在 flow matching 的 ODE 积分过程中，**AR 前缀是不变的**，所以 32 步 velocity 计算全都复用同一份缓存。

这是把 NAR 成本压下来的关键：32 步 × 3888 帧，但只付一次"读谱子"的钱。

### 4.5 阶段 4：VAE 解码成波形

**位置**：`modeling_vae.py` + `pipeline.py:320-356`（`decode`）

VAE 的工作很简单：把 `[帧数, 64]` 的 latent 变成 `[帧数 × 1920 − 64, 2]` 的波形。

- 帧数 3888 × 1920 − 64 = 7,464,896 个采样点 ÷ 48000 = **155.5187 秒** ✓
- 7,464,896 × 2 声道 × 3 字节（24 bit）= 44.8 MB 原始 PCM，压成 FLAC 后 29 MB。

> **为什么是 `−64`？** 转置卷积的边界处理会少掉几个采样点，所以解码器的自然输出长度是 `1920×帧数 − 64`（见 `modeling_vae.py:521` 的注释和 `natural_output_length()`）。这不是 bug，是卷积网络的固有性质——**这也正是分块解码必须留 halo 的根本原因**。

**分块解码（tiled decode）**：为了不把整首歌的中间激活都堆在显存里，VAE 一次只解一块（`modeling_vae.py:539-579`）：

- `core_frames = 1024`（`pipeline.py:148`；当显存预算 ≤ 12 GiB 时自动降到 512）
- `halo_frames = 16`：每块前后多解 16 帧"上下文"，解完再把中间那部分裁下来。

**为什么需要 halo？** 因为卷积网络的感受野是跨块的——第 1000 帧的输出其实依赖第 990–1010 帧的输入。多解 16 帧再裁掉，就保证了每块的输出和"整首歌一次解"完全一致。

`pipeline.py:340-349` 选择用分块还是整首：

```python
tiles = 1 if full else (z.shape[-1] + self.vae_core_frames - 1) // self.vae_core_frames
```

默认走分块（`full=False`）。

最后 `pipeline.py:350-352` 做两道检查：音频里不能有 NaN/Inf；然后 clamp 到 [-1, 1]。

### 4.6 串起来看：`pipeline.py:378-398` 的 `__call__`

```python
def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
             semantic_sampling=None, cancelled=None, on_token=None, **kwargs):
    request = self._request(style, lyrics, tags=tags, **kwargs)     # 1. 组装请求
    config = self.effective_config(request, abc_sampling, semantic_sampling)  # 2. 记录配置
    request_id = identity({"request": ..., "config": ..., "weights": ...})  # 3. 算身份指纹
    plan = self.plan(request=request, ...)                 # 4. 阶段 1
    semantic = self.generate_semantic(plan, ...)           # 5. 阶段 2
    latents = self.synthesize(semantic, ...)               # 6. 阶段 3
    audio = self.decode(latents)                          # 7. 阶段 4
    return SongResult(audio, 48000, semantic, latents, config, self.weights, timing, request_id)
```

**七个步骤，一目了然。** 这就是整个系统的骨架。

---

## 第 5 章 ABC 记谱法小课堂

### 5.1 ABC 是什么？

ABC 是一种**纯文本**的西方乐谱记谱法，1987 年前后出现，用几百个字符就能完整表达一首曲子的旋律、和弦、节拍、速度、曲式。最大的优点是：**人可读、可编辑、可用文本工具处理**。

你可以在任何支持 ABC 的编辑器/播放器里打开 `score.abc` 直接听。

### 5.2 逐行读懂真实的 `output/my_song/score.abc`

```
X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:D
% intro
V: Vocal
z24z4"D"z4|"D"z32|"F#m7"z32|"G"z32|
V: Ins
Z|z4f4z2f2z4f6g2z2a6|z4e4z2e2z4e6f2z2g6|...
```

| 行 | 含义 |
|---|---|
| `X:1` | 曲调编号（可以随便写，1 是惯例） |
| `T:` | 标题（这里是空的） |
| `M:4/4` | 拍号：4/4 拍（四个四分音符一小节） |
| `L:1/32` | **单位音符长度**。后面所有不带数字的音符默认都是 1/32 拍长 |
| `Q:1/4=88` | 速度：四分音符 = 每分钟 88 拍 |
| `V: Vocal ...` | 定义一个"声部"叫 Vocal，高音谱号 |
| `V: Ins ...` | 定义第二个声部叫 Ins（伴奏） |
| `K:D` | 调号：D 大调 |
| `% intro` | 注释行。YuE2 用它来标记**段落**（曲式）：intro / verse / chorus / bridge / interlude / outro |
| `V: Vocal` | 切换到 Vocal 声部，后面的音符属于它 |
| `z24z4` | 休止符。`z` 是休止，后面的数字是"持续几个单位"。`z24` = 休止 24 个单位（24/32 拍长） |
| `"D"` | **和弦符号**。引号里的内容是和弦标记，写在音符上方（这里是 D 大三和弦） |
| `\|` | 小节线 |
| `Z` | 整小节休止（伴奏声部在这里不出声） |
| `f4` | 音符。`f` = F 音，`4` = 持续 4 个单位（4/32 拍长 = 八分音符） |
| `f6` | 同样的 F 音，但持续 6 个单位（6/32 拍长 = 附点八分音符） |
| `Z3` | 3 个单位的小节休止 |

**音高字母表**（七声音阶）：`C D E F G A B`，大写 = 低八度，小写 = 高八度。所以 `c` 是中央 C，`c'` 是再高一个八度，`C` 是低八度的 C。

**时值写法**：字母后面的数字表示"几个单位"。`d4` = d 音 4 个单位；`d` 后面没数字就是默认的 1 个单位（1/32）。`d-` 表示**连音线**，把这个音延长到下一个。

### 5.3 这份谱子里"没有"什么——这才是重点

回头看 `output/my_song/score.abc` 的实际内容：

```
% verse
V: Vocal
"D"d4d4d4d4d4A4f4c4-|"F#m7"c4z16z4B2c6|"G"d4d4d8d4B4B4A4-|"D"A4z16z4"A/C#"z8|
```

你会发现谱子里**只有**：

- 旋律（人声唱的音）
- 和弦符号（`D` `F#m7` `G` `A7sus4` `Bm7` `Em7` `A/C#`）
- 段落标记
- 调号、拍号、速度
- 一条粗略的伴奏声部

谱子里**没有**：

- ❌ 鼓的每一拍
- ❌ 电吉他的失真音色
- ❌ 混响量、混响时间
- ❌ 音量、动态、表情
- ❌ 具体的录音棚、混音、母带处理
- ❌ 咬字、换气、颤音

**这些全部由第 4.4 节的 NAR 阶段，基于 `style` 描述 + 语义 token "现场表演"出来。**

这就是第 1.3 节那个"渐进式承诺"的落点：

> **谱子只承诺"写什么"（音高、时值、和声、形式），不承诺"怎么录"。**
> 于是同一张谱子可以被"演奏"成无数种风格——这正是"翻唱"和"改一个和弦就换风格"能成立的原因。

本项目里 `--cot full` 生成的谱带和弦符号，`--cot melody` 生成的谱**不带**和弦符号。

> **为什么 cover 必须用 `--cot melody`？**
> 因为如果外部 ABC 里带着和弦符号，伴奏就会被锁死在原来的和声上（模型会忠实地按谱演奏），风格就换不动了。去掉和弦符号 = 只保留"旋律骨架"，把和声重新配给伴奏自由发挥。
> **这就是 `RUN.md` 里那句"外部 ABC 里不能带和弦符号"的真正原因。**

---

## 第 6 章 代码地图

### 6.1 目录总览

```
m-a-p2/
├── README.md                  # 项目总览
├── RUN.md                     # 运行手册（操作步骤 + 常见错误）
├── setup_env.sh               # 一键建环境
├── environment.yml            # conda env 定义
├── requirements.txt           # pip 依赖
├── requirements.lock.txt      # 装完后 pip freeze 的完整快照
├── yue2_infer-0.1.5-py3-none-any.whl   # ← 官方推理 wheel（真正的模型代码在这）
├── scripts/                   # ← 本项目自己写的封装层
│   ├── infer_yue2.py          #   主推理脚本：文本 → 歌
│   ├── batch_yue2.py          #   批量推理
│   ├── benchmark_yue2.py      #   性能基准
│   ├── check_env.py           #   环境自检
│   └── yue2_local.py          #   公共工具：探测模型是否在本地缓存
├── examples/                  # 输入样例
├── output/                    # 生成结果（每首一个子目录）
└── tutorial/intro.md          # ← 你正在读的这份
```

### 6.2 两层结构

理解这个项目的关键，是意识到**代码分两层**：

```
┌─────────────────────────────────────────────────┐
│  第二层（本项目写的）                            │
│  scripts/infer_yue2.py  等                     │
│    · 解析命令行参数                              │
│    · 读文件、拼字符串                            │
│    · 调官方 API                                  │
│    · 落盘产物、写 mp3                            │
└────────────────────┬────────────────────────────┘
                     │  from yue2 import YuE2Pipeline
                     ▼
┌─────────────────────────────────────────────────┐
│  第一层（官方 wheel 提供的）                      │
│  site-packages/yue2/                            │
│    · 真正的模型实现                              │
│    · 采样、flow matching、VAE 解码               │
│    · 权重复核、产物存储                          │
└─────────────────────────────────────────────────┘
```

**所以：如果你只想"跑歌"，动第二层就够了。如果你想定制生成过程，动第一层。**

### 6.3 官方包各模块职责

| 模块 | 行数 | 职责 | 关键类/函数 |
|---|---:|---|---|
| `pipeline.py` | 398 | **总调度**：把五个阶段串起来，管理产物的存与取 | `YuE2Pipeline`、`SymbolicPlan`、`SemanticResult`、`SongResult` |
| `protocol.py` | 145 | **语言契约**：prompt 格式、token id 常量、采样默认值、分块公式 | `SongRequest`、`Sampling`、`GenerationConfig`、`token_prefixes`、`chunk_ranges` |
| `modeling_yue2.py` | 705 | **主干模型**：MoT 架构（双注意力 + 双 MLP）、KV cache、训练路径的 NAR | `YuE2ForCausalLM`、`DecoderLayer`、`Backbone`、`StaticKVCache` |
| `nar.py` | 261 | **声学生成**：分块、AR 前缀缓存、32 步 midpoint 积分 | `synthesize`、`CachedNAR`、`song_chunks`、`Chunk` |
| `modeling_vae.py` | 589 | **编解码器**：Oobleck 全卷积 VAE、分块解码 | `YuE2VAE`、`YuE2VAEConfig` |
| `sampling.py` | 154 | **采样循环**：预填充、逐步生成、CFG 混合、词表掩码 | `generate_tokens`、`distribution`、`window_penalty` |
| `tokenization_yue2.py` | 41 | **文本 tokenizer**：冻结的 qwen.tiktoken BPE（**不是**音频 tokenizer） | `YuE2TextTokenizer` |
| `storage.py` | 148 | **存储与校验**：路径解析、sha256、原子写 JSON | `resolve_model`、`model_identity`、`write_json`、`collect_hashes` |
| `cuda_graph.py` | 240 | CUDA Graph 加速的 AR 解码 | `GraphAR` |
| `quantization.py` | 114 | AR 权重的 FP8 量化 / 反量化 | `prepare_fp8_ar`、`restore_ar` |
| `progress.py` | 316 | 终端进度条（进度走 stderr，结果走 stdout） | `Progress`、`_Stage` |
| `fast.py` | 426 | 实验性的 vLLM 后端 | `generate_vllm` |
| `cli.py` | 244 | 官方命令行：`yue2 generate/batch/doctor` | `main` |

---

## 第 7 章 精读：每个模块的输入与输出

> 本章的写法：**先说这个函数是干什么的，再说它吃什么、吐什么、吐出来的东西长什么样。**
> 想跳过理论直接上手的读者，读完 7.7（pipeline）和第 8 章就够了。

### 7.1 `protocol.py` —— 语言契约

这个文件不含任何模型逻辑，它定义了**"YuE2 怎么说话"**。改它 = 改协议。

#### `SongRequest`（`protocol.py:80-112`）

```python
@dataclass(frozen=True)
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"
    seed: int = 831001
    abc: str | None = None
    cfg_scale: float | None = None
    id: str = "song"
```

- **输入**：`style`、`lyrics` 必填；其余有默认值。
- **输出**：一个不可变（frozen）的数据类。
- **构造时的校验**（`__post_init__`，`protocol.py:90-102`）：
  - `cot` 必须是 `off`/`melody`/`full`
  - `seed` 必须是 `[0, 2**63)` 的整数
  - `id` 必须匹配 `[A-Za-z0-9][A-Za-z0-9_.-]{0,179}`（**防止路径穿越**，`..` 被单独拒绝）
  - 给了 `abc` 就不能是 `cot="off"`，且 `abc` 不能为空
  - `cfg_scale` 必须在 `[0, 20]`

- **两个派生属性**：

  `request.text()` → `str`（`protocol.py:108-109`）：
  ```
  {指令}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n
  ```

  `request.guidance` → `float`（`protocol.py:104-106`）：
  ```python
  return (1.01 if self.cot == "off" else 1.0) if self.cfg_scale is None else self.cfg_scale
  ```
  也就是：**没显式指定 `cfg_scale` 时，符号模式用 1.0（等于不引导），`off` 模式用 1.01。**

#### `Sampling`（`protocol.py:21-41`）

一个 frozen dataclass，装着 7 个采样参数。所有数值都有范围校验（温度 0–5、`top_p` ∈ (0,1]、`repetition_penalty` > 0 等）。

#### `GenerationConfig`（`protocol.py:44-67`）

```python
@dataclass(frozen=True)
class GenerationConfig:
    abc: Sampling = field(default_factory=lambda: Sampling(.7, .9, 30, 1.005, 100, 32, 4096))
    semantic: Sampling = field(default_factory=Sampling)
    ode_steps: int = 32
    ode_method: str = "midpoint"
    context: int = CONTEXT       # 24576
    version: str = PROTOCOL_VERSION
```

**注意 `__post_init__` 的严格性**（`protocol.py:53-55`）：`context` 必须正好是 24576，`ode_method` 必须正好是 `"midpoint"`，`ode_steps` 必须是正整数。**这个配置是"协议级"的，不是随便能调的参数。**

`from_dict()`（`protocol.py:60-67`）用于从 `yue2_generation_config.json` 读取模型自带配置。

#### `token_prefixes(request, tokenizer, abc_ids=None)` → `list[int]`

**这是全项目最该看懂的一个函数。** 它把"请求"翻译成"token 序列"。

| 参数 | 类型 | 说明 |
|---|---|---|
| `request` | `SongRequest` | 要编码的请求 |
| `tokenizer` | `YuE2TextTokenizer` | 文本 BPE |
| `abc_ids` | `list[int] \| None` | 已知（或已生成）的 ABC token；`None` 表示"接下来要生成" |

**输出**，取决于 `cot`：

| `cot` | `abc_ids` | 输出 |
|---|---|---|
| `off` | 任意 | `[EOD] + enc(text) + [ABC_START, ABC_END, MUSIC_START]` |
| 任意 | `None`（准备生成） | `[EOD] + enc(text) + [ABC_START]` |
| `full`/`melody` | 给了 | `[EOD] + enc(text) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]` |

还有一条安全检查（`protocol.py:124-125`）：ABC 的 token 必须落在普通文本词表内（`< EOD`），防止越界。

#### `negative_prefix(request, tokenizer, abc_ids=None)` → `list[int]`

CFG 的负分支。区别是**没有 `[Tags]` 和 `[Lyrics]`**：

- `off`：`[EOD] + enc(指令) + [MUSIC_START]`
- `full`/`melody`：`[EOD] + enc(指令) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]`

**符号模式下必须传 `abc_ids`**，否则报错（`protocol.py:133-134`）——因为负分支要保留谱子。

#### `chunk_ranges(frames, prefix_tokens, context=CONTEXT)` → `list[tuple[int, int]]`

```
size = min((context - prefix_tokens - 3) // 2, CONTEXT)
返回 [(0, size), (size, 2*size), ...]
```

算声学分块。`frames` 是语义 token 数量（= latent 帧数），`prefix_tokens` 是 AR 前缀长度。

### 7.2 `tokenization_yue2.py` —— 文本 tokenizer

**类**：`YuE2TextTokenizer`

```python
YuE2TextTokenizer(merge_file)   # merge_file = 模型目录下的 "qwen.tiktoken"
```

- **输入**：tiktoken 的合并表文件路径。
- **输出**：一个可用的 tokenizer 实例。
- **构造时做的事**（`tokenization_yue2.py:11-21`）：
  1. 读合并表，`base64` 解码每一行 → `{字节串: 等级}`。
  2. **校验数量必须正好是 151,643**，否则抛错。
  3. 挂上 208 个特殊 token（3 个空串 + `<R>` `<S>` `<X>` `<mask>` `<sep>` + 200 个 `<extra_i>`，
     其中 2 个被 `<abc>` / `</abc>` 顶掉）。
  4. 用 Qwen 的正则模式构造 `tiktoken.Encoding`。

- **方法**：

  | 方法 | 输入 | 输出 | 说明 |
  |---|---|---|---|
  | `encode(text)` | `str` | `list[int]` | 先做 NFC 规范化（让"é" 的两种写法统一），再 `encode_ordinary`（**不**添加特殊 token） |
  | `decode(ids)` | `list[int]` | `str` | 忽略越界 id，遇到非法字节用 `errors="replace"` |
  | `from_pretrained(path)` | 路径 | 实例 | 走 `resolve_model` 解析 HF repo |
  | `save_pretrained(dir)` | 目录 | `(路径,)` | 把合并表拷出去 |

> **名字骗人的地方**：这个类叫 "TextTokenizer"，注释里特意写了 *"This is not the audio audio tokenizer"*。
> **音频的 tokenizer 是 MERT2，训练时用的，推理时根本不需要加载。** 推理时你只会看到语义 token 的 id，不需要知道它是怎么编码出来的。

### 7.3 `sampling.py` —— 采样循环

#### `window_penalty(logits, recent_ids, penalty)` → `Tensor`

重复惩罚。统计最近窗口内每个 token 出现了 `freq` 次，就给它乘/除 `penalty ** freq`（正 logits 除、负 logits 乘，方向是"压低"）。

#### `distribution(logits, sampling, history, step, phase, legacy_off=False)` → `Tensor`

**把原始 logits 变成可采样的分数**。五步：

1. **词表掩码**（`sampling.py:30-35`）：只放开当前阶段合法的 token。
2. **禁止提前结束**（`sampling.py:37-38`）：`step < min_tokens` 时把结束标记打成 `-inf`。
3. **重复惩罚**（`sampling.py:39`）。
4. **温度**（`sampling.py:40-43`）：`temperature == 0` 直接返回（后面走 argmax 确定性解码）；否则除以温度。
5. **top-k + top-p**（`sampling.py:44-52`）：先砍掉 top-k 之外的全部置 `-inf`，再按累积概率砍掉 top-p 之外的。

#### `generate_tokens(model, prefix, sampling, seed, phase, ...)` → `(history, timing, truncated)`

**这是 AR 阶段的核心函数。**

| 参数 | 类型 | 说明 |
|---|---|---|
| `model` | `YuE2ForCausalLM` | 主干模型（已 `.eval()`） |
| `prefix` | `list[int]` | 前缀 token（prompt + 已有的 ABC） |
| `sampling` | `Sampling` | 采样参数 |
| `seed` | `int` | 随机种子（**两个阶段各自重置**，见 `sampling.py:70-72`） |
| `phase` | `"abc"` \| `"semantic"` | 决定结束标记和词表掩码 |
| `negative` | `list[int] \| None` | CFG 负分支 |
| `cfg_scale` | `float` | 引导强度，`1.0` = 不引导 |

**返回值**：
- `history`：`list[int]`，**不含**结束标记的生成 token 列表
- `timing`：`dict`，含 `seconds` / `prefill_seconds` / `ttft_seconds`（首 token 时间）/ `output_tokens` / `output_tps` / `prefix_tokens` / `cfg_branches` / `execution`（`cuda_graph` 或 `eager`）/ `attention`
- `truncated`：`bool`，**`True` 表示撞上 `max_tokens` 被硬截断**（即没生成出结束标记）

**流程**：

```
prefill(prefix) → logits
循环 max_tokens 次:
    logits = uncond + cfg * (cond - uncond)      # CFG
    scores = distribution(logits, ...)            # 掩码 + 惩罚 + 采样准备
    next_id = multinomial(scores.softmax(-1))    # 采样
    if next_id == end: eos = True; break
    history.append(next_id)
    把 next_id 喂回模型，更新 logits
```

**两个工程细节**：
- **CUDA Graph**（`sampling.py:86-95`）：如果后端是 `torch` 且不是 FP8，就把整个解码循环捕获成 CUDA Graph，省掉大量 kernel launch 开销。这是实测能到 107 tok/s 的原因。`torch-eager` 后端会退回普通循环（更稳但更慢）。
- **前缀长度检查**（`sampling.py:62-67`）：`len(prefix) + max_tokens > 24576` 直接报错，**不做隐式截断**。

### 7.4 `modeling_yue2.py` —— MoT 主干

#### `YuE2Config`（`modeling_yue2.py:61-117`）

继承 HuggingFace 的 `PretrainedConfig`。关键字段：

| 字段 | 默认 | 含义 |
|---|---|---|
| `hidden_size` | 2048 | 隐藏维度 |
| `num_hidden_layers` | 28 | 层数 |
| `num_attention_heads` | 16 | Q 头数 |
| `num_key_value_heads` | 8 | KV 头数（GQA：2 个 Q 头共享 1 组 KV） |
| `head_dim` | 128 | 每头维度 |
| `intermediate_size` | 6144 | SwiGLU 中间层维度 |
| `vocab_size` | 184704 | 词表大小 |
| `max_position_embeddings` | 24576 | 最大位置 |
| `rope_theta` | 1e6 | RoPE 基数 |
| `latent_dim` | 64 | 声学 latent 维度 |
| `max_latent_frames` | 24576 | 声学最大帧数 |
| `timestep_shift` | 1.0 | flow matching 时间偏移（1.0 = 无偏移） |

`to_dict()`（`modeling_yue2.py:73-75`）只序列化白名单字段，避免把推理期不需要的东西写进 config。

#### `Attention`（`modeling_yue2.py:159-213`）

```python
q = self.q_proj(x).view(B, T, num_heads, head_dim)     # [B,T,16,128]
k = self.k_proj(x).view(B, T, num_kv_heads, head_dim)    # [B,T,8,128]
v = self.v_proj(x).view(B, T, num_kv_heads, head_dim)
q, k = self.q_norm(q), self.k_norm(k)                  # RMSNorm on head_dim
q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)   # RoPE
```

**`project_qkv()` 只做投影+归一化+旋转，不做注意力、不管 KV cache、不做输出投影**——这样 `nar.py` 才能复用它来手工搭建"AR 缓存 + NAR 计算"。

`forward()` 则是标准实现：转置成 `[B,H,T,D]` → 更新 KV cache → `sdpa()` → `o_proj`。

#### `MLP`（`modeling_yue2.py:216-224`）

SwiGLU：`down_proj(silu(gate_proj(x)) * up_proj(x))`。三个矩阵，无 bias。

#### `DecoderLayer`（`modeling_yue2.py:227-304`）—— MoT 的灵魂

```python
# AR 路径
self.input_layernorm = RMSNorm(...)
self.self_attn = Attention(config)
self.post_attention_layernorm = RMSNorm(...)
self.mlp = MLP(config)

# NAR 路径（完全独立的一套）
self.nar_input_layernorm = RMSNorm(...)
self.nar_self_attn = Attention(config)
self.nar_pre_mlp_layernorm = RMSNorm(...)
self.nar_mlp = MLP(config)
```

`forward` 有两条分支：

- **`ar_mask is None`**（AR 生成时）：只走 AR 路径，普通因果 LM。
- **`ar_mask is not None`**（NAR 时）：两套都算，然后按掩码选：
  ```python
  query = torch.where(mask_4d, q_ar, q_nar)
  key   = torch.where(mask_4d, k_ar, k_nar)
  value = torch.where(mask_4d, v_ar, v_nar)
  core_out = sdpa(query, key, value, attn_mask=attention_mask)   # 一次注意力
  h = torch.where(mask_3d, self.self_attn.o_proj(core_out),
                        self.nar_self_attn.o_proj(core_out))
  mlp_out = torch.where(mask_3d, self.mlp(...), self.nar_mlp(...))
  ```

> **关键点**：注意力计算**只做一次**（因为 Q/K/V 已经按位置类型选好了），但**输出投影和 MLP 各做两遍**再按掩码选。这就是"Mixture-of-Transformers"的含义：**共享注意力计算，参数不共享**。

#### `StaticKVCache`（`modeling_yue2.py:357-400`）

一个**预分配、只追加**的 KV cache。

- 构造时一次性分配 `[B, kv_heads, max_seq_len, head_dim]` 的张量。
- `update()` 往 `[:, :, pos:end]` 里写，然后返回**视图**（`[:, :, :end]`）——不拷贝历史。
- 超出容量直接抛错（`modeling_yue2.py:388`），**不会偷偷缩短生成**。

#### `YuE2ForCausalLM`（`modeling_yue2.py:462-700`）

主模型。包含：

- `self.model = Backbone(config)` —— 28 层
- `self.lm_head = nn.Linear(2048, 184704, bias=False)` —— AR 输出头
- `self.vae2llm = nn.Linear(64, 2048)` —— **声学 latent → 语言模型空间**（把 noisy latent 变成能进 Transformer 的向量）
- `self.llm2vae = nn.Linear(2048, 64)` —— **语言模型空间 → 速度场**
- `self.time_embedder = TimestepEmbedder(2048)` —— flow matching 的时间嵌入
- `self.latent_pos_embed = AudioPositionEmbedding(24576, 2048)` —— 声学帧的**位置嵌入**（正弦，不可学习）

`forward()` 是标准因果 LM；`nar_velocity()`（`modeling_yue2.py:608-700`）是训练路径的 NAR 实现。

**`vae2llm` / `llm2vae` 这对线性层是理解 MoT 的钥匙**：
声学 latent（64 维）不是一个"词"，没法直接查 embedding 表。所以用 `vae2llm` 把它**投影到 2048 维的语言模型空间**，再**替换掉**该位置的 token embedding 送进 Transformer。出来之后再用 `llm2vae` 投影回 64 维的速度场。

### 7.5 `nar.py` —— 声学生成

#### `song_chunks(prefix, codec, seed, context=CONTEXT)` → `list[Chunk]`

| 参数 | 类型 | 说明 |
|---|---|---|
| `prefix` | `list[int]` | AR 前缀 token（prompt + ABC + 语义 token） |
| `codec` | `list[int]` | 语义 token（0–32767） |
| `seed` | `int` | 噪声种子 |

**输出**：`list[Chunk]`，每个 `Chunk` 有三个字段：

| 字段 | 类型 | 内容 |
|---|---|---|
| `ar_tokens` | `list[int]` | `prefix + [语义token + CODEC_OFFSET] + [MUSIC_END]` |
| `noise` | `Tensor[frames, 64]` | 这一块对应的初始噪声（FP32，CPU） |
| `nar_cond_end` | `int` | 默认 0；>0 时表示"NAR 只能看到前 `nar_cond_end` 个位置"（训练时做 codec dropout 用的） |

**注意**：语义 token 在这里被加上 `CODEC_OFFSET`（`nar.py:47`），变成 151853 起的"伪词表 id"，这样才能和文本 token 放进同一个序列。

#### `CachedNAR`（`nar.py:102-202`）

| 方法 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `__init__` | `model`, `Chunk` | — | 分配位置嵌入，**立即调用 `_prefill()`** |
| `_prefill()` | — | 填充 `self.cache` | 用**因果**注意力跑一遍 AR 前缀，逐层缓存 K/V |
| `velocity(state, raw_t)` | `state[frames,64]`, `raw_t: float` | `Tensor[frames,64]` | 算一次速度场 |
| `solve(steps=32, ...)` | — | `Tensor[frames,64]`（CPU FP32） | 跑完 32 步积分 |
| `close()` | — | — | 清缓存，释放显存 |

**`velocity()` 的内部（`nar.py:151-167`）**：

```python
x_nar = F.pad(state, (0, 0, 1, 1))          # 前后各补 1 帧零 = LATENT_START/END
shifted = model._shift_t_value(raw_t, ...)  # 时间处理
x = model.vae2llm(x_nar[None])              # 64 → 2048
x = x + model.time_embedder(shifted.expand(nar_length))[None]   # 加时间嵌入
x = x + self.pos_emb                        # 加声学位置嵌入
for layer, (ar_k, ar_v) in zip(model.model.layers, self.cache):
    q, k, v = layer.nar_self_attn.project_qkv(layer.nar_input_layernorm(x), self.cos, self.sin)
    k, v = torch.cat((ar_k, k[0])), torch.cat((ar_v, v[0]))   # AR 缓存 + NAR 自己
    h = self._attention(q[0], k, v)          # 无因果掩码 → 双向
    x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
    x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
return model.llm2vae(model.model.norm(x))[0, 1:-1]   # 去掉首尾帧 → 速度场
```

**`solve()` 的中点法（`nar.py:169-198`）**：

```python
state = noise
dt = 1.0 / steps
for step in range(steps):
    t = 1.0 - step * dt
    raw = logit(t).clamp(-20, 20)
    first = self.velocity(state, raw)       # 在 t 处的速度
    mid = state - first * (dt / 2)          # 先走半步
    raw_mid = logit(t - dt/2).clamp(-20, 20)
    state = state - self.velocity(mid, raw_mid) * dt   # 用半步点的速度走完整步
```

这是标准的二阶中点（midpoint）积分：比欧拉法准，代价是每步要算两次速度。32 步 × 2 次 = 64 次前向。

#### `synthesize(model, prefix, codec, seed, steps=32, context=CONTEXT, ...)` → `Tensor[frames, 64]`

顶层入口（`nar.py:227-261`）。流程：

```
chunks = song_chunks(prefix, codec, seed, context)
for chunk in chunks:
    engine = CachedNAR(model, chunk)        # 内部会做一次 AR prefill
    with _offload_ar(model, offload_ar):    # 可选：把 AR 模块挪到 CPU
        output.append(engine.solve(steps))
    engine.close()
return torch.cat(output, dim=0)             # CPU FP32
```

**`_offload_ar`（`nar.py:205-224`）**：如果开了 `offload_ar`，在 NAR 阶段把 `embed_tokens`、`lm_head`、每层的 `input_layernorm`/`self_attn`/`post_attention_layernorm`/`mlp` 搬到 CPU——**因为 NAR 用不到它们**。用完在 `finally` 里搬回来。这是显存不够时的救命开关。

#### `attention(q, k, v, causal=False, backend="sdpa", query_chunk_size=None)` → `Tensor`

一个手写的、**不物化整张掩码矩阵**的注意力。

- 输入 `[tokens, heads, dim]`，输出同形状。
- 用 `F.scaled_dot_product_attention`（PyTorch 原生融合实现，CUDA 上自动走 flash）。
- `query_chunk_size` 控制一次算多少个 query 行（**只影响临时显存，不影响可见的 key 集合**——`nar.py:3-5` 的注释特意强调了这点）。

### 7.6 `modeling_vae.py` —— 编解码器

#### `YuE2VAE`（`modeling_vae.py:374`）

| 方法 | 输入 | 输出 |
|---|---|---|
| `from_pretrained(path, decoder_only=True, device=...)` | 路径 | `YuE2VAE` 实例 |
| `encode(audio, sample=False, generator=None)` | `[B,2,N]` 波形 | `[B,64,T]` latent（**推理时用不到**） |
| `decode(latent)` | `[B,64,T]` | `[B,2,1920*T-64]` 波形，**不裁剪** |
| `decode_tiled(latent, core_frames, halo_frames, output_device, on_progress)` | `[B,64,T]` | `[B,2,total]` 波形 |
| `natural_output_length(frames)` | `int` | `int`，该帧数对应的自然输出长度 |
| `required_halo(core_frames)` | `int` | `int`，这个解码器至少需要的 halo |

**`decode_tiled` 的算法（`modeling_vae.py:539-579`）**：

```
对每个块 [start, end)：
    left  = max(0, start - halo_frames)
    right = min(frames, end + halo_frames)
    tile = self.decode(latent[..., left:right])       # 多解一点
    crop = tile[..., crop_start : crop_start + (out_end - out_start)]   # 裁掉 halo
    audio[..., out_start:out_end].copy_(crop)
```

文档字符串里明确说：**"没有交叉淡化，没有边界平滑，末尾也不补零"**——所以分块解码的结果和整首解码**逐样本相同**。

#### `YuE2VAEConfig`

包含 `decoder_config` / `encoder_config` / `downsampling_ratio`（1920）/ `audio_channels`（2）/ `decode_core_frames` / `decode_halo_frames` 等。

**两个解码器版本**：

| 模型 | 用途 |
|---|---|
| `m-a-p/YuE2-Vae` | 默认，日常更好听 |
| `m-a-p/YuE2-Vae-legacy` | **官方复现 WildSongBench 分数时用的那个** |

> 论文报告的所有 benchmark 数字都是用 `YuE2-Vae-legacy` 解出来的。你本地用默认的 `YuE2-Vae` 生成的音频，**跟论文的分数不是同一套东西**，这点在 `config.json` 里有记录：`"decoder_release": "standard"`。

### 7.7 `pipeline.py` —— 总调度

#### `YuE2Pipeline`

##### `from_pretrained(model, vae=..., device=..., memory_budget_gib=..., backend=..., quantization=..., offload_ar=..., verify_hashes=..., progress=...)` → `YuE2Pipeline`

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | `"m-a-p/YuE2-3B"` | HF repo id 或本地目录 |
| `vae` | `"m-a-p/YuE2-Vae"` | 同上 |
| `local_files_only` | `False` | 只用本地缓存 |
| `memory_budget_gib` | 24 | 显存预算 |
| `backend` | `"torch"` | `torch` / `torch-eager` / `vllm` |
| `quantization` | `"none"` | `none` / `fp8` |
| `offload_ar` | `False` | NAR 阶段卸掉 AR 模块 |
| `verify_hashes` | `True` | 校验权重 sha256 |

**做的事**：
1. 如果 `model` 目录里有 `pipeline.json`，读出里面的模型/VAE 路径和生成配置（`pipeline.py:175-182`）。
2. `resolve_model()` 解析成实际路径（`pipeline.py:184-186`）。
3. 构造实例 → 里头的 `__init__` 做：
   - 参数校验（backend / quantization / budget）；
   - `device == "auto"` 时自动选 cuda → mps → cpu；
   - **关掉 TF32、关掉 cudnn benchmark、设 `float32_matmul_precision("highest")`**（`pipeline.py:139-144`）——为了数值可复现；
   - 加载 tokenizer；
   - **校验权重 sha256**（`pipeline.py:152-154`）——这就是那 21.6 秒的来源；
   - 算 `runtime_sha256`（`pipeline.py:155`）——把 `yue2/` 包里所有 `.py` 文件哈希一遍，记进 `config.json`。**这让"我用的是哪版代码"也可追溯。**
   - CUDA 上：检查 BF16 支持（`pipeline.py:159-160`），设置显存上限：
     ```python
     budget = min((memory_budget_gib - 2) * 2**30, total - 2 * 2**30)
     torch.cuda.set_per_process_memory_fraction(min(budget / total, 1), device)
     ```

> **`--budget` 的真实语义**：实际可用显存 = `min(budget - 2, 显卡容量 - 2)` GiB。
> 所以**把 budget 设得比显卡还大没关系**，会自动夹到上限；反过来，设成 ≤ 2 会直接报错（`ValueError: Memory budget must leave room for a 2GiB reserve`）。
> 另外 `budget ≤ 12` 时，`vae_core_frames` 自动从 1024 降到 512（`pipeline.py:148`）——**分块更小 → 更省显存**。

##### `plan(style, lyrics, cot, seed, abc, cfg_scale, id)` → `SymbolicPlan`

**输入**：风格、歌词、模式、种子、（可选）外部 ABC。
**输出**：`SymbolicPlan` 对象。

三种走向（`pipeline.py:255-269`）：

| 情况 | 行为 |
|---|---|
| `cot == "off"` | 不规划，直接返回一个只有前缀的空计划 |
| 给了 `abc` | 把外部 ABC 编码成 token，作为前缀（`timing.seconds = 0`，`external_prefix_tokens = len(ids)`） |
| 其他 | 调用 `_generate()` 生成 ABC |

**`SymbolicPlan`（`pipeline.py:19-63`）** 的字段：

| 字段 | 类型 | 内容 |
|---|---|---|
| `request` | `SongRequest` | 原始请求 |
| `abc` | `str \| None` | ABC 文本 |
| `abc_ids` | `list[int]` | ABC 的 token id |
| `prefix` | `list[int]` | 完整的 AR 前缀（prompt + ABC + 标记） |
| `timing` | `dict` | 耗时、token 数、tok/s |
| `truncated` | `bool` | 是否被截断 |

**方法**：
- `save(directory)`：写 `score.abc`、`abc_tokens.npy`、`prefix.npy`、`plan.json`、`plan_manifest.json`。
- `load(directory)`（类方法，`pipeline.py:43-63`）：**恢复计划并校验完整性**——逐个文件比对 sha256，比对 `.npy` 数组和 `plan.json` 里的列表是否一致，任何不一致就抛错。

> `SymbolicPlan.load` 是"编辑工作流"的关键：你可以先只规划（快，12 秒），把 ABC 拿出去改，改完再喂回去合成，**不用重新规划**。

##### `generate_semantic(plan, sampling=None, ...)` → `SemanticResult`

| 步骤 | 细节 |
|---|---|
| 1 | 校验 `plan.prefix` 和 `token_prefixes(request, tokenizer, plan.abc_ids)` 一致（`pipeline.py:275-277`）——防止前缀被篡改 |
| 2 | 算负分支（`cfg_scale != 1` 时才有） |
| 3 | `_generate(plan.prefix, semantic_sampling, seed, "semantic", ...)` |
| 4 | 把返回的 id 减去 `CODEC_OFFSET` → `SemanticResult.tokens` |

**`SemanticResult`（`pipeline.py:66-71`）**：`plan` + `tokens` + `timing` + `truncated`。

##### `synthesize(semantic, ...)` → `np.ndarray`

调 `nar.synthesize`，返回 CPU FP32 `[frames, 64]` 的 numpy 数组。

##### `decode(latents, full=False, vae=None)` → `np.ndarray`

1. 把 AR 模型挪到 CPU 并清显存（`pipeline.py:323-326`）——**给 VAE 腾地方**。
2. 加载/复用 VAE。
3. `torch.inference_mode()` 下分块解码，`output_device="cpu"`（避免整首歌堆在显存里）。
4. 检查无非有限值，clamp 到 [-1, 1]，转 numpy。

**输出形状**：`[samples, 2]`，float32，范围 [-1, 1]。

##### `effective_config(request, abc_sampling=None, semantic_sampling=None)` → `dict`

把"这次到底用了什么配置"记下来，写进 `config.json`。包括：

- `generation`：完整的采样配置
- `overrides`：相对默认值改了什么
- `cot`、`cfg_scale`、`cfg_negative`（负分支是 `instruction_only` 还是 `same_instruction_and_exact_abc`）
- `backend`、`quantization`、`model_dtype`（bf16）、`vae_dtype`（fp32）、`vae_decode`（halo_crop）
- `device`、`memory_budget_gib`、`offload_ar`
- `runtime_sha256`：代码指纹
- `decoder_release`：`"standard"` 或 `"legacy"`
- `validation_status`：`"unvalidated"`（**诚实地告诉你：这个配置没经过官方验证**）

##### `close()` / `__enter__` / `__exit__`

释放模型、清 CUDA 缓存、支持 `with` 语法。

#### `SongResult`（`pipeline.py:74-118`）

| 字段 | 类型 | 内容 |
|---|---|---|
| `audio` | `np.ndarray` | 波形 `[samples, 2]` |
| `sample_rate` | `int` | 48000 |
| `semantic` | `SemanticResult` | 语义阶段结果 |
| `latents` | `np.ndarray` | `[frames, 64]` float32 |
| `config` | `dict` | 生效配置 |
| `weights` | `dict` | 权重指纹 |
| `timing` | `dict` | 各阶段耗时 |
| `request_identity` | `str` | 请求+配置+权重的联合 sha256 |

**属性**：`song.abc` → 谱子文本；`song.truncated` → `{"abc": bool, "semantic": bool}`。

**方法**：
- `save(path)`：只写音频。**只允许 `.flac` 或 `.wav`**（`pipeline.py:98-99`），FLAC 用 `PCM_24`，WAV 用 `FLOAT`。
- `save_artifacts(directory)` → `dict`：写全部产物，返回 `result.json` 的内容。

### 7.8 其余模块速览

#### `storage.py` —— 存储与校验

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `sha256_file(path)` | 路径 | 64 字符 hex | 分块读，不占内存 |
| `identity(value)` | 任意 JSON-able | 64 字符 hex | 对 JSON 做规范化排序后哈希 |
| `write_json(path, value)` | 路径 + 对象 | — | **原子写**：先写 `.tmp` 再 `os.replace` |
| `resolve_model(model, ...)` | repo id 或目录 | `Path` | 是目录就返回它；否则 `snapshot_download` |
| `model_identity(path, verify=True)` | 模型目录 | `dict` | 列出所有 `.safetensors` 的 sha256 + 字节数；`verify=True` 时与 `weights_manifest.json` 比对 |
| `collect_hashes(directory)` | 目录 | `dict` | 目录下所有文件的 sha256 + 大小（`result.json` 除外） |
| `verify_result(directory, expected_identity)` | 目录 | `dict` | 校验 `result.json` 里记录的每个产物都还在、没变 |

**`MODEL_FILES`（`storage.py:33-38`）** 是一个白名单：只有这些文件会被下载/导出。"额外报告、缓存和任意代码都不导出"——这是一道安全边界。

#### `quantization.py` —— FP8

| 函数 | 作用 |
|---|---|
| `prepare_fp8_ar(model, device)` | 把 AR 路径的 `nn.Linear` 换成 `FP8Linear`，权重按 per-channel 量化 |
| `restore_ar(model)` | 换回来（**NAR 阶段前必须调用**，因为 NAR 需要原始精度） |
| `quantization_status(model)` | 报告当前量化状态 |

**注意**：FP8 模式下 **CUDA Graph 会被禁用**（`sampling.py:86`），所以会慢一些。`timing["graph_fallback_reason"]` 会写明原因。

#### `cuda_graph.py` —— CUDA Graph

`GraphAR(model, prefixes, max_tokens, ...)`：把整个自回归解码循环捕获成 CUDA Graph。

| 方法 | 作用 |
|---|---|
| `prefill()` | 捕获并执行 prefill，返回正/负分支的 logits |
| `step(token)` | 在图里跑一步，返回新的 logits |
| `close()` | 释放图资源 |

CFG 模式下正负两个分支被**放在同一个 batch 里**（`sampling.py:92`），一次前向算两条分支——这是它比朴素实现快的原因之一。

#### `progress.py` —— 进度显示

`Progress(enabled=True)` 是一个上下文管理器，`stage(label, total, unit)` 返回一个 `_Stage`。

**设计原则很重要**（`pipeline.py` 里的注释）：**进度信息永远走 stderr，结果永远走 stdout。** 这样你 `> result.json` 重定向时，进度条不会污染你的结果文件。

`_Stage.advance()` 累加计数，`_Stage.finish(status="truncated")` 标记被截断。

#### `fast.py` —— 实验性 vLLM 后端

把 AR 阶段换成 vLLM 加速。需要 NVIDIA GPU，且 FP8 模式会退回 torch。

**本项目默认不用它**（`--backend` 只有 `torch` 和 `torch-eager` 两个选项）。

#### `cli.py` —— 官方命令行

```bash
yue2 doctor                    # 环境体检
yue2 generate --request examples/song.json --output outputs/song
yue2 batch --input requests.jsonl
```

`generate` 支持 `--stage plan`（只规划）和 `--stage audio`（完整），以及 `--resume`（从已保存的计划续跑）。

> 本项目**没有**用这个 CLI，而是自己写了 `scripts/infer_yue2.py`。原因是官方 CLI 依赖完整 release kit，本项目只有 wheel。

### 7.9 模块间的调用关系全景

```
scripts/infer_yue2.py
   │
   ├── yue2_local.apply_offline()           ← 判断模型是否已在本地缓存
   │
   └── yue2.YuE2Pipeline.from_pretrained()   → pipeline.py
            │
            ├── storage.resolve_model()     → 找到权重目录
            ├── storage.model_identity()    → sha256 校验
            ├── tokenization_yue2.YuE2TextTokenizer()
            │
            └── pipe(...)  = pipeline.py:378
                 │
                 ├── [1] plan()                        → protocol.token_prefixes
                 │        └── sampling.generate_tokens(model, prefix, ..., phase="abc")
                 │             ├── modeling_yue2.YuE2ForCausalLM.forward()
                 │             └── cuda_graph.GraphAR  (可选)
                 │
                 ├── [2] generate_semantic()           → protocol.negative_prefix
                 │        └── sampling.generate_tokens(..., phase="semantic")
                 │
                 ├── [3] synthesize()                  → nar.synthesize
                 │        ├── nar.song_chunks()         → protocol.chunk_ranges
                 │        └── nar.CachedNAR
                 │             ├── _prefill()           → modeling_yue2 的层
                 │             └── solve()  × 32        → velocity() × 2/步
                 │
                 └── [4] decode()                      → modeling_vae.YuE2VAE
                          └── decode_tiled()
```

---

## 第 8 章 本项目的脚本层

这一层是本项目自己写的，**逻辑很薄**，主要是命令行封装。

### 8.1 `scripts/infer_yue2.py` —— 主脚本

#### `parse_args(argv=None)` → `argparse.Namespace`

把参数分成 4 组：

| 组 | 参数 |
|---|---|
| **输入** | `--style`、`--style_txt`、`--lyrics`、`--lyrics_txt`、`--abc_txt`、`--cot` |
| **采样** | `--seed`、`--cfg_scale`、`--id` |
| **模型** | `--model`、`--vae`、`--revision`、`--vae_revision`、`--backend`、`--quantization`、`--offload_ar`、`--no_verify_hashes` |
| **运行** | `--device`、`--budget`、`--output_dir`、`--plan_only`、`--offline`、`--quiet` |

#### `read_text(path, base)` → `str`

相对路径按 `PROJECT_DIR` 解析（`infer_yue2.py:88-92`）。

#### `resolve_input(args)` → `(style, lyrics, abc)`

把 `--style`/`--style_txt` 和 `--lyrics`/`--lyrics_txt` **归一**成两个字符串：

- 风格：多个来源用 `", "` 拼接；
- 歌词：多个来源用 `"\n"` 拼接；
- `abc`：只有 `--abc_txt` 时才有，否则 `None`。

**如果 style 或 lyrics 任何一个为空，直接 `SystemExit`**（`infer_yue2.py:105-106`）并给出提示。

#### `main(argv=None)` → `int`

```
1. 解析参数
2. resolve_input()
3. 判断是否离线：
     offline = args.offline or yue2_local.apply_offline([args.model, args.vae])
   若离线 → 设 HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
4. import yue2（失败时给出"怎么装"的友好提示）
5. 计算输出目录 = output_dir / id
   若目录非空且不是 --plan_only → 报错退出（防覆盖）
6. YuE2Pipeline.from_pretrained(...)
7. 如果 --plan_only：
       plan = pipe.plan(...)
       plan.save(out_dir)
       打印 {"stage": "plan", "output": ..., "truncated": ...}
       return 0
8. 否则：
       song = pipe(style=..., lyrics=..., cot=..., seed=..., abc=..., cfg_scale=..., id=...)
       result = song.save_artifacts(out_dir)
       额外写一个 audio.mp3（soundfile 写；失败则退回 ffmpeg；再失败就跳过，不影响主结果）
       打印 JSON 摘要
       return 0
9. finally: pipe.close()
```

**几个值得注意的细节**：

- **输出目录非空保护**（`infer_yue2.py:136-137`）：这是防止你手滑覆盖掉之前满意的版本。
- **mp3 是"便利副本"**（`infer_yue2.py:175-195`）：FLAC 才是正式产物。写 mp3 失败**不会**影响主流程（`except Exception` 里只打日志）。
- **stdout / stderr 分工**：进度和提示走 stderr，结果 JSON 走 stdout。所以 `python scripts/infer_yue2.py ... > summary.json` 只会得到干净的结果。

### 8.2 `scripts/batch_yue2.py` —— 批量

- **输入**：JSONL 文件，每行一个 JSON，字段 `id`（必填、唯一）、`style`（必填）、`lyrics`（必填）、`cot`/`seed`/`cfg_scale`/`abc`（可选）。
- **关键点**：**模型只加载一次**（`batch_yue2.py:71-76`），然后逐首生成。这就是批量比单首快得多的原因。
- **容错**：某一首失败不会中断整批（`batch_yue2.py:102-106`），失败信息记进摘要。
- **输出**：每首一个子目录 + 根目录下的 `batch.json` 摘要（`{"complete": bool, "expected": N, "failed": N, "results": [...]}`）。
- **退出码**：有失败返回 1。

### 8.3 `scripts/benchmark_yue2.py` —— 性能基准

三种模式：

| `--mode` | 做什么 | 报什么 |
|---|---|---|
| `load` | 只把模型读进显存 | 加载耗时、显存峰值、权重指纹 |
| `plan` | 只跑乐谱规划 | 规划耗时、token 数、tok/s、谱子前 400 字符 |
| `e2e` | 完整跑一遍 | 分阶段耗时、音频时长、**RTF**（实时因子）、显存峰值 |

**RTF = 端到端耗时 / 音频时长**。RTF = 0.55 意味着**生成 1 分钟音乐只要 33 秒**。

### 8.4 `scripts/check_env.py` —— 环境自检

6 项检查，每项打印 `[OK ]` / `[WARN]` / `[FAIL]`：

1. Python 版本（≥ 3.10，推荐 3.12）
2. 依赖版本（9 个包的**精确版本**，与 wheel 的 METADATA 一致）
3. import 冒烟测试（真的 `import` 一遍）
4. CUDA / GPU / BF16 支持
5. HF 缓存里的模型权重是否齐全
6. libsndfile 支持哪些格式（FLAC / WAV / MP3 / OGG）

**退出码 0 = 一切就绪，1 = 有阻塞问题。**

> `check_env.py:72` 有个注释很有意思：PEP 440 规定 `2.10.0+cu128` 满足 `==2.10.0`（本地版本标识不参与比较），所以它手动用 `got.split("+")[0]` 比较。

### 8.5 `scripts/yue2_local.py` —— 本地缓存探测

| 函数 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `hf_cache_root()` | — | `Path` | `$HF_HOME/hub`，默认 `~/.cache/huggingface/hub` |
| `repo_cached(repo)` | repo id 或路径 | `bool` | 本地目录直接返回 True；否则查 `models--<repo>/snapshots/` 下有没有快照 |
| `all_cached(repos)` | 列表 | `bool` | 全部命中才算 True |
| `apply_offline(repos)` | 列表 | `bool` | 全部命中就设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` |

**为什么需要它？** `transformers` 加载模型时会去 huggingface.co 做 HEAD 校验（确认远端有没有更新）。网络不通时它会**指数退避重试**，看起来像"卡死了"。

但权重既然已经下好了，联网校验毫无意义。所以这个脚本先探测，命中缓存就自动切离线。

> 这就是 `README.md` 里说"**不用手动加 `--offline`**"的原因。

---

## 第 9 章 磁盘上的文件到底是什么

### 9.1 产物总表

```
output/my_song/
├── audio.flac        29 MB   48 kHz / 立体声 / 24 bit —— ★ 正式成品
├── audio.mp3         2.6 MB  同一份音频的 mp3 —— 便利副本
├── score.abc         1.7 KB  模型写的乐谱（旋律 + 和弦 + 段落）
├── plan.json         27 KB   请求 + 规划耗时 + token + 完整前缀
├── plan_manifest.json 341 B  规划阶段产物的 sha256
├── abc_tokens.npy    5.3 KB  ABC 的 token id（int32）
├── prefix.npy        6.1 KB  送进语义阶段的前缀 token（int32）
├── semantic.npy      15 KB   语义 token（int32）
├── latent.npy        973 KB  声学 latent（float32）
├── request.json      813 B   本次请求
├── config.json       1.0 KB  生效配置 + 代码指纹
└── result.json       3.1 KB  耗时 / 时长 / truncated / 各文件 sha256
```

### 9.2 每个 `.npy` 的精确形状

用 `output/my_song`（155.52 秒的歌）实测：

| 文件 | dtype | shape | 说明 |
|---|---|---|---|
| `abc_tokens.npy` | int32 | `(1332,)` | ABC 文本的 BPE token（不含 `ABC_END`） |
| `prefix.npy` | int32 | `(1535,)` | 语义阶段的前缀 |
| `semantic.npy` | int32 | `(3888,)` | 语义 token（已减掉 `CODEC_OFFSET`，值域 0–32767） |
| `latent.npy` | float32 | `(3888, 64)` | 声学 latent |

**验算（前缀的构成）**：

```
1 (EOD) + 199 (prompt 文本) + 1 (ABC_START) + 1332 (abc_tokens) + 2 (ABC_END, MUSIC_START) = 1535  ✓
```

（用 `token_prefixes(request, tokenizer, abc_ids)` 重算一遍，结果和 `prefix.npy` **逐个元素完全一致**。）

**验算（时长）**：

```
语义 token 数 3888 = latent 帧数 3888 = 3888 / 25 Hz = 155.52 秒
采样点 = 3888 × 1920 − 64 = 7,464,896 → / 48000 = 155.5187 秒  ✓
```

> **这张表是全文最有用的"物理量纲对照表"**：
> **25 Hz 是贯穿全系统的时钟**。语义 token、latent 帧、音频秒数三者用同一个 25 Hz 换算。
> `155.52 秒 ≈ 3888 帧 = 3888 个语义 token`。
>
> 顺带一提，**声学分块大小也能算出来**：
> `size = (24576 − 1535 − 3) // 2 = 11519` 帧 > 3888 帧，所以这首歌**只分 1 块**，不需要分块。
> （按默认上限——乐谱 ≤ 4096 token、语义 ≤ 9000 token——只要"文本 + 乐谱"的 token 总数不超过约 6,570，
> 一首歌就永远落在同一块里；分块只是给极端输入准备的安全网，普通歌根本不会触发。）

### 9.3 `result.json` 里有什么

```json
{
  "status": "complete",
  "identity": "54f45b72...",         // 请求+配置+权重的联合指纹
  "truncated": { "abc": false, "semantic": false },
  "sample_rate": 48000,
  "audio_seconds": 155.51866666666666,
  "weights": { "mot": {...sha256...}, "vae": {...sha256...} },
  "timing": {
    "abc":      { "seconds": 12.41, "output_tokens": 1333, "output_tps": 107.4, ... },
    "semantic": { "seconds": 35.76, "output_tokens": 3889, "output_tps": 108.8, ... },
    "nar_seconds": 24.43,
    "vae_seconds": 8.47,
    "load": { "resolve_and_integrity_seconds": 21.63, "mot_load_seconds": 0.36 },
    "e2e_seconds": 85.67
  },
  "artifacts": { "audio.flac": { "sha256": "...", "bytes": 29083895 }, ... }
}
```

**`timing` 里的 `output_tokens` 比 `abc_tokens.npy` 的长度多 1**——因为前者把结束标记 `ABC_END` 也算进去了（`sampling.py:141`：`count = len(history) + int(eos)`）。

### 9.4 本机实测性能（RTX 3090 24 GB）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| 解析路径 + sha256 校验 | 21.6 s | `--no_verify_hashes` 可跳过 |
| 权重加载到 GPU | 0.4 s | 显存峰值 6.76 GiB |
| 乐谱规划（AR） | 12.4 s | 1333 tokens，**107 tok/s**，CUDA Graph + flash attention |
| 语义 token（AR） | 35.8 s | 3889 tokens，**109 tok/s** |
| NAR 声学合成（32 步 flow matching） | 24.4 s | |
| VAE 解码 | 8.5 s | |
| **端到端** | **85.7 s** | 产出 **155.5 s** 音频，**RTF ≈ 0.55** |

> **RTF 0.55 是什么感觉**：等 85 秒，拿到 2 分半的歌。比实时快接近一倍。
> 那 21.6 秒的 sha256 校验占了总时间的 1/4——**赶时间就加 `--no_verify_hashes`**。

---

## 第 10 章 四种玩法与三种模式

### 10.1 三种 `cot` 模式

| `cot` | 写不写谱 | 谱里有和弦吗 | 什么时候用 |
|---|---|---|---|
| `full`（默认） | 写 | 有 | **绝大多数情况**。质量最好（论文的 49.3% vs 34.6% 就是这个） |
| `melody` | 写 | 没有 | 翻唱、外部谱、想让伴奏自由发挥 |
| `off` | 不写 | — | 快速出一版；或做"有规划 vs 无规划"的对比实验 |

**`off` 模式的技术含义**：前缀变成 `[EOD] + enc(text) + [ABC_START, ABC_END, MUSIC_START]`——ABC 段是**空的**（`protocol.py:117-118`）。模型直接开始写语义 token。

**`off` 模式的默认 `cfg_scale` 是 1.01 而不是 1.0**（`protocol.py:106`）——因为没有符号部分作为"锚"，CFG 的负分支只剩指令，权重分布变了，所以官方调了 1.01 来补偿。

### 10.2 四种玩法

#### 玩法 A：从文字生成一首歌

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output --id my_song
```

#### 玩法 B：先看谱，再决定要不要渲染

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --plan_only --output_dir output --id my_plan
```

**只跑 12 秒**（省掉 73 秒的语义 + NAR + VAE），产出 `output/my_plan/score.abc`。
用任何 ABC 查看器打开就能看到模型写的谱子。

> **这是本项目最有价值的一个功能**，因为它把"创作"变成了一个可迭代的循环：改提示词 → 12 秒 → 看谱 → 再改。

#### 玩法 C：用外部谱做 cover

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --abc_txt examples/melody.abc \
    --cot melody \
    --output_dir output --id cover_song
```

**流程是**：音频 → （SheetSage2 转谱）→ ABC → YuE2 重新演奏。

外部 ABC 必须满足（`protocol.py:99-100`）：
- `cot` 必须是 `melody` 或 `full`（不能 `off`）
- 内容非空

**且不能带和弦符号**——原因见第 5.3 节。

#### 玩法 D：改谱重录（编辑）

```
1. plan_only 生成 score.abc
2. 手工/让 AI 修改 score.abc（比如把 D 和弦改成 Dm）
3. 把改好的谱作为 --abc_txt 传回去，--cot full
```

**模型会尽量保留你没改的部分**。论文的编辑实验数据：

| 编辑类型 | 跟随度（越高越好） | 未编辑部分保留度 |
|---|---:|---:|
| 旋律 | 84.17 | 93.10 – 94.34 |
| 和声 | 79.54 | 90.16 – 94.05 |
| 节奏 | 73.43 | — |
| 调号 | 90.58 | — |
| 速度 | 95.68 | — |

> "保留度"的含义：比如你只改了和声，那旋律有 90–94 分（满分 100）保持不变。

### 10.3 快速对比：`full` vs `off`

想亲眼验证论文的"规划有用"这个结论？跑两遍：

```bash
# 有规划
python scripts/infer_yue2.py --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt --cot full --output_dir output --id cmp_full

# 无规划
python scripts/infer_yue2.py --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt --cot off  --output_dir output --id cmp_off
```

然后对比两边的 `score.abc`：`cmp_full/` 里有谱子，`cmp_off/` 里没有。

---

## 第 11 章 亲手跑一遍

### 11.1 第一步：确认环境

```bash
cd ~/distributed_machine_learning/m-a-p2
conda activate yue2
python scripts/check_env.py
```

期望最后一行是：`结果: 全部通过, 可以跑 scripts/infer_yue2.py`

### 11.2 第二步：先只规划，看看谱子（12 秒）

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --plan_only --output_dir output --id my_first_plan
```

然后：

```bash
cat output/my_first_plan/score.abc
```

**你应该能看到**：调号、拍号、速度、`% verse` / `% chorus` 段落标记、和弦符号（`D` `F#m7` `G`...）、两条声部（`V: Vocal` 和 `V: Ins`）。

如果谱子不对劲（比如调号奇怪、段落标签乱、长度太短），改 `lyrics.txt` 再跑一次——**每次只花 12 秒**。

### 11.3 第三步：渲染成音频

```bash
python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output --id my_first_song
```

听 `output/my_first_song/audio.mp3`。

### 11.4 第四步：检查产物完整性

```bash
python -c "
import json
d = json.load(open('output/my_first_song/result.json'))
print('时长:', round(d['audio_seconds'],1), '秒')
print('是否被截断:', d['truncated'])
print('端到端耗时:', round(d['timing']['e2e_seconds'],1), '秒')
print('RTF:', round(d['timing']['e2e_seconds']/d['audio_seconds'],2))
"
```

### 11.5 第五步：批量试试

```bash
python scripts/batch_yue2.py --input examples/requests.jsonl --output_dir output/batch
cat output/batch/batch.json
```

**模型只加载一次**，3 首歌大约 2 分钟。

### 11.6 复现官方 demo

```bash
python scripts/infer_yue2.py \
    --style_txt examples/official_style.txt \
    --lyrics_txt examples/official_lyrics.txt \
    --seed 12300 --budget 22 \
    --output_dir output --id tonight_awake
```

这是官方仓库 demo 请求的复现（City Pop 风格，《今晚不眠》），产出 212.1 秒的音频。

---

## 第 12 章 调参与常见问题

### 12.1 调参速查

| 想要 | 怎么做 |
|---|---|
| 换个风格 | 改 `--style` 里的标签 |
| 歌更长 | 歌词加段落（Verse/Chorus 重复） |
| 歌更短 | 歌词砍到 2–4 段 |
| 旋律更贴歌词 | `--cfg_scale 1.2`（默认 full/melody 是 1.0） |
| 不想要规划 | `--cot off` |
| 只要旋律没和弦 | `--cot melody` |
| 换一首曲子 | `--seed 1234` |
| 显存不够 | `--budget 12` → `--quantization fp8` → `--offload_ar` |
| 想更快 | 保持 `--backend torch`（CUDA Graph） |
| 少等 20 秒 | `--no_verify_hashes` |

### 12.2 关于 `--cfg_scale` 的直觉

`cfg_scale` 调大的效果是"**更用力地把注意力推向文本条件**"。

- `1.0`（默认）= 不引导，等于纯条件生成。
- `1.1`–`1.3` = 稍微更贴歌词/风格，可能牺牲一点自然度。
- 太大（> 2）= 文本条件压倒一切，容易出现"机械感"。

> **符号模式的一个额外好处**：因为负分支保留了 ABC（`protocol.py:138`），调大 `cfg_scale` **不会破坏你已经定好的旋律和和弦**。它只改变"文字对表演的影响强度"。

### 12.3 常见错误速查

| 症状 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: No module named 'yue2'` | 用错 Python | `conda activate yue2`，确认 `which python` 指向 `envs/yue2/bin/python` |
| 装 wheel 报版本冲突 | pip 自动解依赖 | `pip install --no-deps ./yue2_infer-0.1.5-py3-none-any.whl` |
| 装 torch 报找不到版本 | 没加 PyTorch 源 | `pip install torch==2.10.0 --extra-index-url https://download.pytorch.org/whl/cu128` |
| `RuntimeError: The unquantized preset requires CUDA BF16 support` | 显卡不支持 BF16 | 需要 Ampere 及以上（sm_80+）；RTX 3080/3090 是 sm_86，没问题 |
| `ValueError: Memory budget must leave room for a 2GiB reserve` | budget ≤ 2 | 实际可用 = `min(budget-2, 显存-2)`，超过卡容量会自动夹到上限 |
| `CUDA out of memory` | 显存不够 | 按顺序试 `--budget 12` → `--quantization fp8` → `--offload_ar` |
| 卡在 "Resolving model files" 反复 retry | 联网校验卡住 | 加 `--offline`（一般脚本会自动切） |
| 生成的歌很短 / mp3 小于 1 MB | `truncated=true` | 歌词写长点，或确认用的是 `--cot full` 而不是 `off` |
| 音频全是静音 | VAE 解码问题 | 试 `--vae m-a-p/YuE2-Vae-legacy` |
| `输出目录非空` 报错 | 防覆盖保护 | 换 `--id`，或 `rm -rf output/<id>` |

### 12.4 "truncated" 到底意味着什么

`result.json` 里：

```json
"truncated": { "abc": false, "semantic": false }
```

**任一为 `true` 表示模型在 `max_tokens` 处被硬截断了。**

| 字段 | 对应上限 | 撞上的后果 |
|---|---|---|
| `abc` | 4096 | 谱子写了一半就停了 → 歌会短、结构会断 |
| `semantic` | 9000 | 语义 token 写了一半 → 音频只覆盖前半段 |

**处理办法**：歌词写长一点（多几段），或者确认 `--cot full` 而不是 `off`。
`off` 模式下没有乐谱来"框定"歌曲该有多长，语义 token 的长度完全由模型自己决定，更容易撞上 9000 的上限。

---

## 第 13 章 术语表

| 术语 | 一句话解释 | 本项目里对应什么 |
|---|---|---|
| **AR (Autoregressive)** | 一个 token 一个 token 地生成，每个都依赖前面所有已生成的 | 乐谱规划、语义展开两个阶段 |
| **NAR (Non-Autoregressive)** | 所有位置同时生成 | 声学 latent 合成 |
| **MoT (Mixture-of-Transformers)** | 一层里放两套专家，按位置类型路由 | `DecoderLayer` 里的 AR/NAR 双路径 |
| **token** | 模型眼里的最小单位（一个整数） | 文本 token / 语义 token |
| **tokenizer** | 文本 ↔ token id 的转换器 | `YuE2TextTokenizer`（冻结的 qwen BPE） |
| **semantic token** | 音乐内容的离散压缩表示 | `semantic.npy`，25 Hz，32768 词表 |
| **latent** | 压缩后的连续向量表示 | `latent.npy`，`[帧数, 64]` |
| **VAE** | 变分自编码器，把波形压成 latent 再还原 | `modeling_vae.py`，1920× 压缩 |
| **Oobleck** | 一种带 Snake 激活的卷积架构 | VAE 的骨干 |
| **flow matching** | 训练一个速度场，用 ODE 把噪声积分成数据 | 阶段 3 |
| **midpoint 积分** | 二阶 ODE 求解器，每步算两次速度 | `nar.py:170` |
| **CFG** | 无条件引导，用有/无条件两个分支混合 | `--cfg_scale` |
| **negative prefix** | CFG 的"无条件"分支输入 | `protocol.py:129` |
| **prefix** | 生成前先喂给模型的 token 序列 | `prefix.npy` |
| **KV cache** | 缓存已算过的 K/V，避免重复计算 | `StaticKVCache` |
| **CUDA Graph** | 把一串 GPU 操作捕获成图，减少 launch 开销 | `cuda_graph.py` |
| **GQA** | 分组查询注意力，Q 头多于 KV 头以省显存 | 16 Q 头 / 8 KV 头 |
| **RoPE** | 旋转位置编码 | `RotaryEmbedding` |
| **RMSNorm** | 归一化激活值，不需要均值中心化 | `modeling_yue2.py:125` |
| **SwiGLU** | `silu(gate) * up` 的前馈结构 | `modeling_yue2.py:216` |
| **halo（分块解码的边距）** | 每块多解一点再裁掉，保证分块与整首结果一致 | `halo_frames = 16` |
| **RTF (Real-Time Factor)** | 端到端耗时 / 音频时长 | 本机约 0.55 |
| **truncated** | 生成撞上 token 上限被硬截断 | `result.json` |
| **ABC 记谱法** | 纯文本乐谱格式 | `score.abc` |
| **lead sheet** | 只有旋律 + 和弦的简谱 | YuE2 的谱子就是这种 |
| **best-of-8** | 生成 8 个候选，按 Musicality → Q3O → PER 挑最好的 | 论文里的最高分设置 |

---

## 第 14 章 论文数据附录

### 14.1 论文基本信息

| 项 | 值 |
|---|---|
| 标题 | *YuE2: Unifying Symbolic and Audio Music Generation at Frontier Quality* |
| arXiv | **2609.33757**（2026-09-27 提交，56 页技术报告） |
| 机构 | HKUST、M·A·P（Multimodal Art Projection）、Tokenwave.AI、NYU、Stanford 等 |
| 代码 | https://github.com/multimodal-art-projection/YuE |
| 权重 | https://huggingface.co/m-a-p/YuE2-3B |
| 在线 demo | https://map-yue2.github.io/ |
| 基准数据集 | https://huggingface.co/datasets/m-a-p/WildSongBench |
| 许可证 | 代码 Apache 2.0；**权重 CC BY-NC 4.0**（非商用） |

### 14.2 训练配方

**数据规模**

| 模型 | 音频时长 | 阶段 |
|---|---:|---|
| MERT2 | 700,000 小时 | 基础预训练 |
| SheetSage2-AR | 28,400 小时 | 自回归转谱 |
| **YuE2** | **346,000 小时** | 符号 + 音频联合生成 |

**四种训练任务混合采样**（论文 Table 14）——这是"同一个 checkpoint 能切换三种行为"的根本原因：

| 任务 | AR 预测目标 | 声学条件 |
|---|---|---|
| 完整生成 | 谱子，然后语义 token | 文字、歌词、谱子、语义 token |
| 无符号规划 | 语义 token | 文字、歌词、语义 token |
| 无语义 token | 谱子 | 文字、歌词、谱子 |
| 直接音频生成 | 无 | 文字、歌词 |

**三阶段训练日程**（论文 Table 15）

| 阶段 | 步数 | 最大上下文 | 学习率 |
|---|---:|---:|---|
| AR 预训练 | 60,000 | 16,384 | 5,000 步线性升温到 1e-4，余弦衰减到 1e-5 |
| AR–NAR 联合 | 40,000 | 24,576 | 5,000 步升温到 3e-4，余弦衰减到 1e-4 |
| 退火 | 6,000 | 24,576 | 无升温，从 ~2.726e-4 余弦衰减到 3e-5 |

**优化**：bfloat16，全局 batch 256，梯度裁剪 1.0，**64 张 H800**，Adam（β=(0.9,0.98)/wd 0.05 预训练；β=(0.9,0.95)/wd 0.1 联合+退火）。

> **一条重要的工程细节**：歌曲**不按时间切块打包**，注意力索引和位置索引在文档边界处重置。所以一整首歌能放进一个 24,576 的上下文窗口里。

### 14.3 WildSongBench 主表（论文 Table 2，节选）

| 模型 | SongBench Mus.↑ | SongBench Avg↑ | SongEval Avg↑ | PQ↑ | MuLan↑ | AMCaps↑ | Q3O↑ | PER↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YuE1 | 4.0847 | 4.9165 | 3.2150 | 7.8683 | 0.2623 | 0.2882 | 3.7301 | 0.3638 |
| SongBloom | 3.4493 | 4.2350 | 3.2051 | 8.1539 | 0.2697 | 0.1926 | 3.0287 | 0.1919 |
| LeVo 2 | 5.4590 | 6.3247 | 4.0234 | **8.3966** | 0.3542 | 0.2680 | 3.9458 | 0.2612 |
| ACE-Step 1.5 | 5.1588 | 6.0118 | 3.8465 | 8.0518 | 0.4372 | 0.3869 | 4.5809 | 0.0746 |
| HeartMuLa | 5.4963 | 6.2483 | **4.5519** | 8.2933 | 0.3823 | 0.2786 | 3.4907 | 0.1071 |
| DiffRhythm 2 | 4.4775 | 5.2428 | 3.5245 | 7.9782 | 0.3782 | 0.3255 | 4.0870 | 0.1841 |
| Muse | 5.1692 | 6.0349 | 3.7887 | 8.0517 | 0.3937 | 0.3466 | 4.4038 | 0.3342 |
| MM Music 3 | 5.3482 | 6.2830 | 4.0994 | 8.2825 | 0.3928 | 0.3609 | 4.4362 | 0.0627 |
| **YuE2** | 5.9075 | 6.7316 | 4.2625 | 8.2598 | **0.5068** | **0.4054** | 4.6819 | 0.0844 |
| **YuE2 (Bo8)** | **6.2666** | **6.9632** | 4.2960 | 8.2714 | 0.5051 | 0.3980 | **4.7009** | 0.0979 |

**怎么看这张表**：

- YuE2 在**所有被评估的公开系统**中，于 SongBench Musicality、Global Avg、MuLan、AllMusicCaps 四项领先。
- best-of-8 的均值是**全部 17 个设置里最高的**。
- 但 YuE2 **不是**每一项都第一：LeVo 2 的 PQ 更高，HeartMuLa 的 SongEval 更高，Suno v4.5/v5.5 的 PER 更低，Mureka 9 的总体质量更高。
- 参数量上，**3.58B 的 YuE2 比 8 个公开基线里的 6 个都小**，比 HeartMuLa 少 34%。

### 14.4 专家盲听（论文主实验）

351 次参与，4,439 条成对评价（其中 3,602 条用于 Arena 偏好）。62 名付费标注者有音乐学院或音频 AI 研究背景。

**vs 商用系统：**

| 对比 | YuE2 Bo8 胜率 | 对手胜率 |
|---|---:|---:|
| vs Suno v4.5 | **57.3%** | 30.5% |
| vs Suno v5 | 40.4% | 39.9%（基本打平） |
| vs Suno v6 | 31.6% | **59.3%**（Suno v6 赢） |

**六个商用系统平均的"音质"维度：YuE2 58.9%**（95% CI 55.6–62.2%）。

### 14.5 谱子与音频的一致性（论文 Table，"score–audio agreement"）

384 个生成的谱子 + 768 段录音：

| 条件 | 旋律相似度↑ | 和弦相似度↑ | 节奏 F1↑ | 调号完全一致↑ | 速度准确率↑ |
|---|---:|---:|---:|---:|---:|
| **谱子与音频对应** | **0.9464** | **0.9246** | 0.9396 | 0.9307 | 0.9869 |
| 谱子与音频错配 | 0.2160 | 0.2473 | 0.6119 | 0.3004 | 0.6806 |
| 无谱子 | 0.2055 | 0.1897 | 0.5875 | 0.2234 | 0.5366 |

**在"这段录音对应哪张谱子"的两选一任务里，准确率 100%。**
这说明谱子不是装饰——**它真的在控制声音**。

### 14.6 零样本翻唱（论文 Table，SHS100K 测试集 948 首未见作品）

| 方法 | CLEWS mAP↑ | Hit@1↑ | Discogs Hit@1↑ | MuLan↑ | Q3O↑ | PQ↑ | Mus.↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SongEcho | 0.419 | 48.4 | 16.6 | 0.366 | 4.474 | 6.862 | 3.286 |
| ACE-Step 1.5 | 0.024 | 2.4 | 0.6 | 0.166 | 4.190 | 6.918 | 3.689 |
| **YuE2（完整谱）** | **0.647** | **71.3** | **37.5** | 0.382 | 4.273 | 8.044 | 5.104 |
| YuE2（无和弦） | 0.598 | 67.3 | 23.6 | 0.417 | 4.482 | 8.117 | 5.490 |
| YuE2（无谱） | 0.006 | 0.3 | 0.2 | **0.474** | **4.837** | **8.186** | **5.691** |

> **这张表展示了经典的两难**：
> - **完整谱** → 作品身份保留得最好（CLEWS mAP 0.647），但音乐性最低（5.104）。
> - **无谱** → 音乐性最高（5.691）、最贴合目标风格，但完全认不出原作（mAP 0.006）。
> - 948 首里有 864 首（91.1%）是无谱版本音乐性更高。
>
> **这也解释了为什么 cover 模式推荐 `--cot melody`**：给旋律骨架（保留身份），但把和声交给伴奏自由发挥（换取风格适配）。

### 14.7 MERT2 / SheetSage2（论文的辅助模型）

| 模型 | 规模 | 成绩 |
|---|---:|---|
| MERT2 | 632M | MARBLE 15 项指标中 **14 项**领先；GTZAN 节奏 F1 91.72、MTG-Jamendo Top-50 AP 80.01 |
| MERT2 tokenizer | 375 bit/s | 6 项冻结探针指标中领先 5 项（EnCodec 1.5 kbit/s，LeVo 2 350 bit/s）；量化后仍保留 78.3–98.3% 的分数；**码本利用率 > 99%** |
| SheetSage2-AR | — | 15 个"基准×指标"组合中领先 12 个；**在自己的标注生成器（Prober）之上，10/15 项更好** |

---

## 附：一句话总结这份文档

**YuE2 做的事**：让一个模型先把"写什么"（ABC 乐谱：旋律、和弦、曲式）想清楚并写成人类可读的文本，再把"写什么"展开成密集的"语义 token"，最后把"怎么录"（声学 latent → 波形）交给一个并行的 flow matching 阶段。

**为什么这样更好**：因为"写歌"和"录音"是两种难度完全不同的任务。前者需要全局的、结构性的决策，适合在几千维的离散符号空间里做；后者需要的是局部、连续、听感的还原，适合在稠密的连续空间里做。**分开做，每一步都更轻松；而由于中间产物是可读的乐谱，你还能在中间插手。**

**你能用它做什么**：写歌、翻唱、改一个和弦就换风格、让 AI 帮你改谱子。全部用同一个 checkpoint。

---

*这份文档基于 `m-a-p/YuE2-3B` + `m-a-p/YuE2-Vae` 权重、`yue2_infer-0.1.5` wheel 的实际代码，以及论文 arXiv:2609.33757 撰写。所有性能数字来自本机 RTX 3090 的实测记录（`output/my_song/result.json`）。*
