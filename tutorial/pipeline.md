# YuE2 精读讲义（一）：`pipeline.py` 总调度层

> **这份讲义写给谁？**
> 写给已经读过《YuE2 入门讲义》，觉得"我大概知道它在干什么了，但真要我自己写一行代码调用它，还是心里没底"的同学。
>
> 入门讲义回答的是 **"为什么"**——为什么要写谱子、为什么要分三步、为什么要这么多校验。
> 这份讲义回答的是 **"怎么用"**——`YuE2Pipeline` 这个类，你给它什么，它还你什么；中间那几个小函数，各自有怎样的输入和输出；为什么它要在某些地方突然报错。
>
> **读法建议**：不要从第 1 行往下读。请按下面的顺序：第 1 章建立地图 → 第 2 章看懂三个数据类 → 第 3 章逐个函数拆解 → 第 4 章看真实产物 → 第 5 章动手 → 第 6 章排错 → 第 7 章自测。
> 每读完一节，请先合上文档，用自己的话复述一遍"它吃什么、吐什么"。**复述不出来，就回去再读一遍。**

---

## 目录

| 章节 | 内容 | 适合什么时候读 |
|---|---|---|
| [第 0 章](#第-0-章-这份讲义怎么用) | 怎么用、符号约定、心态准备 | 一开始 |
| [第 1 章](#第-1-章-先建立一张地图) | 先建立一张地图：pipeline.py 到底在干什么 | 一开始 |
| [第 2 章](#第-2-章-三个数据类先吃透) | 三个数据类：`SymbolicPlan` / `SemanticResult` / `SongResult` | 想搞清"结果长什么样" |
| [第 3 章](#第-3-章-逐个函数拆解) | 逐个函数拆解（本文主体） | 想动手调用时 |
| [第 4 章](#第-4-章-一次真实调用发生了什么) | 一次真实调用发生了什么（逐行时间线） | 想建立整体印象 |
| [第 5 章](#第-5-章-三条上手路线) | 三条上手路线（照着抄即可） | 立刻 |
| [第 6 章](#第-6-章-常见报错与排查手册) | 常见报错与排查手册 | 遇到问题时 |
| [第 7 章](#第-7-章-自测题与参考答案) | 自测题与参考答案 | 学完自检 |

---

## 第 0 章 这份讲义怎么用

### 0.1 代码位置与约定

- **被讲解的源文件**：`/home/luke/miniconda3/envs/yue2/lib/python3.12/site-packages/yue2/pipeline.py`（共 399 行）
- **本项目脚本**：`/home/luke/distributed_machine_learning/m-a-p2/scripts/infer_yue2.py`
- **本项目代码根目录**：`/home/luke/distributed_machine_learning/m-a-p2`
- 本文引用代码时写成 `pipeline.py:行号`，例如 `pipeline.py:255`，你可以直接打开对照着看。

### 0.2 一个必须先摆正的心态

很多同学第一次打开这类"总调度"文件，会本能地觉得它"没什么技术含量"——不就是把几个函数串起来吗？

**请务必打消这个念头。** 我带过的学生里，能把模型原理讲得头头是道的有十几个，能把 `pipeline.py` 讲清楚的不超过三个。原因是：原理是"广"的，可以查资料补；**调度层是"深"的，它藏着整个工程里所有的取舍、边界和保护措施。**

举几个真实的例子，你读完第 3 章就会回来看这一段：

- 为什么 `plan()` 生成的谱子，存盘之后能用 `load()` 原样取回，**连重新分词的抖动都没有**？
- 为什么 `generate_semantic()` 在动手之前，要先算一遍前缀再和计划里的前缀**逐个数字比对**？
- 为什么 `decode()` 开工第一件事是把模型搬到 CPU？
- 为什么 `effective_config()` 里要老老实实写上一句 `"validation_status": "unvalidated"`？

这些都不是"串起来"能解释的。这是**工程判断**。而工程判断，正是这份讲义要教的东西。

### 0.3 阅读时的三个小习惯

1. **先看签名，再看函数体。** 先问"它吃什么、吐什么"，再看"它怎么做到的"。
2. **每遇到一个 `raise`，停下来问"它在防什么"。** 这个文件里的每一条异常，都是作者踩过坑之后留下的护栏。
3. **每遇到一个 `if`，问一句"什么情况下走另一条路"。** 分支才是理解的难点。

---

## 第 1 章 先建立一张地图

### 1.1 `pipeline.py` 的定位

在 YuE2 这个项目里，`pipeline.py` 是**面向用户的门面（Facade）**。

请回忆入门讲义第 6 章的代码地图：`protocol.py` 定义"怎么说话"，`sampling.py` 管"怎么采样"，`nar.py` 管"怎么出声学 latent"，`modeling_vae.py` 管"怎么还原成波形"。这些模块**各自都能单独工作**，但谁也不知道谁的存在。

`pipeline.py` 的职责就是**把它们串起来，并且管住四件事**：

| 职责 | 具体体现 |
|---|---|
| **参数校验** | 非法组合在开工前就报错，绝不跑到一半才崩 |
| **资源管理** | 什么时候把模型搬上 GPU、什么时候搬下来、什么时候释放 |
| **阶段衔接** | 上一个阶段的产物，如何**严格地**传给下一个阶段 |
| **可追溯性** | 这次跑用的什么代码、什么权重、什么配置，全部写进磁盘文件 |

用一句更接地气的话说：**`pipeline.py` 是"总导演"，不是"演奏家"。它自己几乎不做音乐，它只保证四个乐手按顺序、按调性、不打架地演奏完一首歌。**

### 1.2 文件里有哪些角色

`pipeline.py` 一共只有 4 个"角色"，记住这 4 个就够了：

```
┌─────────────────────────────────────────────────────────────┐
│  角色 A：三个数据类（结果的信封）                             │
│    SymbolicPlan   —— "谱子阶段的结果"                          │
│    SemanticResult —— "语义阶段的结果"                          │
│    SongResult     —— "整首歌的最终结果"                         │
├─────────────────────────────────────────────────────────────┤
│  角色 B：一个主类（总调度）                                    │
│    YuE2Pipeline   —— 从加载模型到产出音频的全部方法              │
├─────────────────────────────────────────────────────────────┤
│  角色 C：三个内部方法（下划线开头，别从外面调）                 │
│    _request  _generate  _status                              │
├─────────────────────────────────────────────────────────────┤
│  角色 D：从别的模块借来的工具函数                                │
│    token_prefixes / negative_prefix / resolve_sampling ...   │
└─────────────────────────────────────────────────────────────┘
```

**关于角色 C 的学习建议**：Python 里 `_` 开头的名字是"这是内部实现，不是公开 API"的约定（业内叫 *private convention*，不是强制机制）。初学者常犯的错是直接调 `pipe._generate(...)`。**请不要这样做**——内部方法的参数含义和调用时机是有前提的，外部调用者没有那些前提。

### 1.3 一次调用的数据流全景

这是**本文最重要的一张图**，请先记住它，后面每一章都是在往这张图上填细节。

```
                        你（调用者）
                            │
              style + lyrics (+ cot / seed / abc / cfg_scale)
                            │
                            ▼
                  ┌──────────────────────┐
                  │  YuE2Pipeline.__call__│  ← 一次到位版（pipeline.py:378）
                  │   或者你手动分步调用     │
                  └───────────┬──────────┘
                              │
      ┌───────────────────────┼────────────────────────┐
      ▼                       ▼                        ▼
  ① plan()              ② generate_semantic()      ③ synthesize()
  pipeline.py:255        pipeline.py:271             pipeline.py:285
      │                       │                        │
      ▼                       ▼                        ▼
  SymbolicPlan            SemanticResult          latents: np.ndarray
  {abc, abc_ids,          {plan, tokens,           shape = [frames, 64]
   prefix, timing}         timing, truncated}       float32 / CPU
                              │                        │
                              └───────────┬────────────┘
                                          ▼
                                  ④ decode()
                                  pipeline.py:320
                                          │
                                          ▼
                                    SongResult
                              {audio, latents, config,
                               weights, timing, identity}
                                          │
                                          ▼
                              save_artifacts(dir)   ← 写盘，可复现
                              pipeline.py:103
```

**请注意第 ④ 步之前没有"写盘"，第 ④ 步之后才有。** 这意味着：**整首歌的波形在最终落盘之前，一直待在内存里。** 一首 155 秒的立体声 float32 波形约 60 MB（`155.5 × 48000 × 2 × 4` 字节），加上 latent 和其他中间物，单次调用的内存峰值在几百 MB 量级——对训练机来说不是问题，但对内存紧张的机器，请注意一次只跑一首歌。

### 1.4 `SongResult` 与 `save_artifacts` 的分工

一个新手常问："我已经有 `SongResult` 了，为什么不直接 `song.save("a.flac")` 就行了？还要 `save_artifacts` 做什么？"

**请记住这个分工：**

- `save(path)` → **只给你音频**。适合"我只想听一耳朵"。
- `save_artifacts(directory)` → **给你一整套可复现、可校验的东西**。适合"我要把这次结果存档、给别人、或者以后想复现"。

`save_artifacts` 写出的东西，是后面第 4 章和第 6 章的主角。今天先记住一句话：**`save_artifacts` 写出的目录，是这个项目里的"事实来源"。**

---

## 第 2 章 三个数据类先吃透

在讲方法之前，必须先讲清楚"方法返回的东西长什么样"。这一章是全文的基础。

### 2.1 `SymbolicPlan` —— 谱子阶段的结果

源码位置：`pipeline.py:19-63`

```python
@dataclass
class SymbolicPlan:
    request: SongRequest
    abc: str | None
    abc_ids: list[int]
    prefix: list[int]
    timing: dict = field(default_factory=dict)
    truncated: bool = False
```

**一句话概括**：*"我打算这么唱——请求是这么发的，谱子是这么写的，这是它的 token 编号，这是完整的提示前缀。"*

#### 输入输出对照

| 字段 | 类型 | 它是什么 | 从哪来 |
|---|---|---|---|
| `request` | `SongRequest` | 本次请求的完整快照（风格/歌词/模式/种子…） | 调用者 |
| `abc` | `str \| None` | ABC 乐谱**文本**。`None` 表示"没有谱子"（`cot="off"` 时） | 模型生成 或 你提供 |
| `abc_ids` | `list[int]` | ABC 文本的 **token 编号序列** | `tokenizer.encode(abc)` |
| `prefix` | `list[int]` | 喂给模型 autoregressive 阶段的**完整前缀** | `token_prefixes(...)` |
| `timing` | `dict` | 耗时、token 数、tok/s 等 | 采样模块回填 |
| `truncated` | `bool` | 是否因为撞到 `max_tokens` 上限而被**截断** | 采样模块回填 |

#### 为什么 `abc` 和 `abc_ids` 要同时存在？

这是一个非常值得学的点。请同学们对比一下：

- `abc` 是**给人看的**：你在编辑器里打开 `score.abc`，看到的是 `X:1 / T: / M:4/4 ...` 这样可读的文本。
- `abc_ids` 是**给机器用的**：`[1234, 5678, 9012, ...]` 这样的数字序列。

**为什么不能只存一个？** 因为"文本 → token"的编码过程有**规范化**（`tokenization_yue2.py:24` 里做了 `unicodedata.normalize("NFC", text)`）。你存了文本、以后再编码一次，理论上可能出现细微差异。**而 token 序列是"模型当时真正看到的"东西。**

所以这份设计是：**保存"事实"（abc_ids），同时保存"事实的可读副本"（abc）。** 两者都在，就有了互相校验的能力——这正是 `load()` 干的事。

#### `prefix` 是什么？

`prefix` 是**模型 autoregressive 阶段实际接收的输入序列**。它由 `protocol.py:115` 的 `token_prefixes()` 拼出来：

```
[EOD] + encode(请求文本) + [ABC_START] + abc_ids + [ABC_END, MUSIC_START]
```

对 `cot="off"`：
```
[EOD] + encode(请求文本) + [ABC_START, ABC_END, MUSIC_START]
```

**请务必理解 `prefix` 的严肃性。** 在第 3 章你会看到，`generate_semantic()` 会**重新算一遍**前缀，然后和 `plan.prefix` **逐个数字比对**，不一致直接报错。这不是冗余，这是**"计划与执行必须严格对应"的契约**。

#### 方法一：`save(directory)`

源码位置：`pipeline.py:28-41`

**输入**：`directory`（目录路径，`Path` 或 `str` 均可）
**输出**：`None`（副作用式写盘）

写出的文件：

| 文件 | 内容 |
|---|---|
| `score.abc` | ABC 文本（**仅当 `abc` 不为 `None` 时才写**） |
| `abc_tokens.npy` | `np.int32` 的 `abc_ids` 一维数组 |
| `prefix.npy` | `np.int32` 的 `prefix` 一维数组 |
| `plan.json` | 请求 + 耗时 + 截断标记 + `prefix` + `abc_ids` + `abc` 文本 |
| `plan_manifest.json` | 上面每个文件的 sha256 |

注意 `plan.json` 里**既有 `prefix` 列表，也有 `abc_ids` 列表**——文本和二进制各存一份。这样即使 `.npy` 损坏，`plan.json` 还在；反之亦然。**这就是"双份记录"的思路。**

#### 方法二：`load(directory)` —— 全文件最值得学的一段

源码位置：`pipeline.py:43-63`

**它是一个类方法（`@classmethod`），输入是目录，输出是一个全新的 `SymbolicPlan`。**

它的文档字符串只有一句话，但信息量极大：

> *"Restore exact planner output without decoding and retokenizing its ABC."*
> （还原**精确的**规划器输出，**不需要**解码并重新分词它的 ABC。）

请对照着读一遍 `load()` 的校验链条，我把它拆成五关：

```
第 1 关：manifest 完整性
  └─ plan_manifest.json 里必须同时有 plan.json / abc_tokens.npy / prefix.npy
     少一个 → ValueError("Incomplete saved plan")

第 2 关：文件名白名单 + 拒绝符号链接
  └─ manifest 里出现的每个名字，必须属于允许集合
     且该文件不能是 symlink
     为什么要拒 symlink？因为符号链接可以把校验指向别处的文件——
     这是典型的"路径穿越"攻击面。→ ValueError("Invalid plan artifact")

第 3 关：内容哈希一致
  └─ 每个文件重算 sha256，必须和 manifest 里记录的一致
     → ValueError("Saved plan changed; supply modified ABC as an external planner input")

第 4 关：数组与 JSON 交叉核对
  └─ 加载 .npy（allow_pickle=False！），检查：
       ndim == 1
       dtype.kind in "iu"（必须是整数，不能是浮点）
       数组内容 == plan.json 里对应的列表
     → ValueError("Saved plan token array mismatch")

第 5 关：ABC 文本一致
  └─ score.abc 的字节 == plan.json 里 abc 字段编码后的字节
     → ValueError("Saved ABC text mismatch")
```

**请注意 `allow_pickle=False` 这个参数。** 这是 `numpy.load` 的安全开关：允许 pickle 意味着加载一个文件等于**执行一段任意代码**。官方包默认关闭它，是负责的做法。你们以后处理任何"别人给的 `.npy` 文件"时，都应该记住这个默认值。

**教学要点：第 3 关的报错信息是一句很有水平的话**——*"Saved plan changed; supply modified ABC as an external planner input"*。它不只是说"错了"，它还**告诉你正确的做法**：如果你确实想改谱子，那就把它作为**外部输入**（`request.abc`）重新走一遍流程，而不是偷偷改文件。这个设计非常值得学习：**报错信息要能指导用户走向正确路径。**

### 2.2 `SemanticResult` —— 语义阶段的结果

源码位置：`pipeline.py:66-71`

```python
@dataclass
class SemanticResult:
    plan: SymbolicPlan
    tokens: list[int]
    timing: dict
    truncated: bool
```

**一句话概括**：*"谱子定了，这是照着谱子写出来的语义 token。"*

| 字段 | 类型 | 说明 |
|---|---|---|
| `plan` | `SymbolicPlan` | **原封不动地把上一步的计划带过来** |
| `tokens` | `list[int]` | 语义 token 编号（**已减去 `CODEC_OFFSET`**，见 3.5 节） |
| `timing` | `dict` | 耗时、tok/s、前缀长度、CFG 分支数、执行方式（cuda_graph / eager）等 |
| `truncated` | `bool` | 语义阶段是否被截断 |

**为什么 `plan` 要被完整带过来？** 因为 `synthesize()` 需要用 `semantic.plan.prefix`（前缀）和 `semantic.plan.request.seed`（种子）来渲染音频。**如果不带，第三个阶段就得回头去别处找第一阶段的东西——数据流就散了。** 把上游产物一路携带，是流水线代码里非常常见也非常有效的一种组织方式。

### 2.3 `SongResult` —— 整首歌的最终结果

源码位置：`pipeline.py:74-118`

```python
@dataclass
class SongResult:
    audio: np.ndarray
    sample_rate: int
    semantic: SemanticResult
    latents: np.ndarray
    config: dict
    weights: dict
    timing: dict
    request_identity: str
```

| 字段 | 类型 | 说明 | 实测参考值（`output/my_song/`） |
|---|---|---|---|
| `audio` | `np.ndarray` | **成品波形**，形状 `[samples, 2]`，float32，范围 `[-1, 1]` | 155.52 秒 → `7464896 × 2` |
| `sample_rate` | `int` | 采样率，**恒为 48000** | 48000 |
| `semantic` | `SemanticResult` | 语义阶段结果（含内部的 plan） | 3888 个 token |
| `latents` | `np.ndarray` | 声学 latent，形状 `[frames, 64]`，float32 | 3888 × 64 |
| `config` | `dict` | **这次实际生效**的完整配置 | 见 `config.json` |
| `weights` | `dict` | 主模型 + VAE 的权重指纹 | 见 `result.json` |
| `timing` | `dict` | 各阶段耗时 | 见 `result.json` |
| `request_identity` | `str` | **请求 + 配置 + 权重的联合 sha256** | 64 位十六进制 |

#### 两个便利属性

```python
@property
def abc(self):        # pipeline.py:85-87
    return self.semantic.plan.abc

@property
def truncated(self):  # pipeline.py:89-91
    return {"abc": self.semantic.plan.truncated, "semantic": self.semantic.truncated}
```

**`song.abc` 是"谱子阶段是否被截断"和"语义阶段是否被截断"的字典**，而不是一个布尔值。**这是有讲究的**：一首 155 秒的歌，如果 ABC 阶段撞了 `max_tokens` 上限（`full` 模式下 `max_tokens=4096`），那这首歌的**结构本身就是不完整的**——后面的语义 token 再生得再好，也是在一段残缺的曲子上发挥。**你需要分别知道是哪一步出的问题，才能决定怎么办。**

#### 方法一：`save(path)` —— 只写音频

源码位置：`pipeline.py:93-101`

| 项 | 说明 |
|---|---|
| **输入** | `path`：文件路径 |
| **输出** | 字符串形式的路径（`str(path)`） |
| **扩展名限制** | **只允许 `.flac` 或 `.wav`**，其它一律 `ValueError` |
| **FLAC 子类型** | `PCM_24`（24 位整数，体积小、无损） |
| **WAV 子类型** | `FLOAT`（32 位浮点，保留原始精度） |

**请注意那句报错**："Use .flac or .wav; MP3 is an optional delivery conversion"。**这是"正式产物"与"便利副本"的分界。** MP3 是有损压缩，属于"交付格式转换"，不是"生成结果本身"。`save_artifacts()` 写的是 `audio.flac`，那是正式产物；而本项目脚本 `infer_yue2.py:177-195` 额外导出的 `audio.mp3`，被明确放在 `try/except` 里——**mp3 导出失败不影响主结果**。这个分层是刻意的。

#### 方法二：`save_artifacts(directory)` —— 写全套

源码位置：`pipeline.py:103-118`

| 项 | 说明 |
|---|---|
| **输入** | `directory`：目录路径（不存在会自动创建） |
| **输出** | `dict` —— 即 `result.json` 的内容 |

写出的文件（对照 `output/my_song/` 的真实目录）：

```
output/my_song/
├── score.abc          ← 谱子（人看）
├── abc_tokens.npy     ← 谱子的 token（机器看）
├── prefix.npy         ← 完整前缀
├── plan.json          ← 计划全量快照
├── plan_manifest.json ← 计划文件的哈希
├── audio.flac         ← 成品音频（正式产物）
├── semantic.npy       ← 语义 token
├── latent.npy         ← 声学 latent
├── request.json       ← 本次请求参数
├── config.json        ← 本次生效配置
└── result.json        ← 总账：耗时、时长、截断、权重、产物哈希
```

`result.json` 的结构，值得逐字段读一遍（数据取自本项目实测产物）：

```json
{
  "status": "complete",
  "identity": "54f45b72...3cebf",
  "truncated": { "abc": false, "semantic": false },
  "sample_rate": 48000,
  "audio_seconds": 155.51866666666666,
  "weights": {
    "mot": { "files": { "model.safetensors": { "sha256": "1d55c42c...", "bytes": 7261441640 } },
             "config_sha256": "ad3477bb..." },
    "vae": { "files": { "model.safetensors": { "sha256": "807ce9d5...", "bytes": 530512720 } },
             "config_sha256": "f0191bb9..." }
  },
  "timing": {
    "abc":      { "seconds": 12.41, "output_tokens": 1333, "output_tps": 107.4, ... },
    "semantic": { "seconds": 35.76, "output_tokens": 3889, "output_tps": 108.8, ... },
    "nar_seconds": 24.43,
    "vae_seconds": 8.47,
    "load": { "resolve_and_integrity_seconds": 21.63, "mot_load_seconds": 0.36 },
    "e2e_seconds": 85.67
  },
  "artifacts": { "audio.flac": { "sha256": "9a8b8ba9...", "bytes": 29083895 }, ... }
}
```

**请把这张表和第 4 章的时间线对照着看**——`timing` 里的每一个数字，都对应流水线上的一个阶段。这是"性能调优"的入口：**你先看哪一段慢，再决定优化哪里。**

---

## 第 3 章 逐个函数拆解

本章是本文主体。每个函数我按**五要素**讲：**它是什么 → 输入 → 输出 → 它做了什么 → 它在防什么**。

### 3.1 构造函数 `YuE2Pipeline.__init__`

源码位置：`pipeline.py:121-165`

#### 签名

```python
def __init__(self, model_dir, vae_dir, *, device="auto", memory_budget_gib=24,
             backend="torch", generation_config=None, verify_hashes=True,
             vae_core_frames=None, quantization="none", offload_ar=False, progress=True):
```

**请注意 `*` 之后的所有参数都是 keyword-only（仅限关键字传参）。** 这是一个很好的 API 设计习惯：把可能被误传的参数挡在关键字后面。`model_dir` 和 `vae_dir` 是必填的位置参数，其余都有默认值。

#### 参数逐个讲

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_dir` | 必填 | 主模型（AR + NAR）目录 |
| `vae_dir` | 必填 | 声学解码器目录 |
| `device` | `"auto"` | `"auto"` 时依次尝试 cuda → mps → cpu |
| `memory_budget_gib` | `24` | 显存预算（GiB） |
| `backend` | `"torch"` | `"torch"` / `"torch-eager"` / `"vllm"` |
| `generation_config` | `None` | 采样配置；`None` 时用 `GenerationConfig()` 默认值 |
| `verify_hashes` | `True` | 是否逐文件校验权重 sha256 |
| `vae_core_frames` | `None` | VAE 分块解码的核心块长；`None` 时按显存预算自动定 |
| `quantization` | `"none"` | `"none"` / `"fp8"` |
| `offload_ar` | `False` | 声学阶段是否把主干卸到 CPU 腾显存 |
| `progress` | `True` | 是否显示进度条 |

#### 它做了 6 件事

**第 1 件事：参数校验（`pipeline.py:125-133`）**

```python
if not isinstance(progress, bool):   raise TypeError("progress must be True or False")
if backend not in {"torch","torch-eager","vllm"}:  raise ValueError(...)
if quantization not in {"none","fp8"}:  raise ValueError(...)
if not 0 < memory_budget_gib:  raise ValueError("memory_budget_gib must be positive")
```

**教学要点**：`progress` 用 `TypeError`，其余用 `ValueError`。这个区分是对的——**类型不对是调用方的笔误（TypeError），值不对是参数内容有问题（ValueError）**。Python 社区有这个不成文的惯例，写库时值得照做。

**第 2 件事：设备选择（`pipeline.py:134-138`）**

```python
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
self.device = torch.device(device)
if self.device.type == "cuda" and self.device.index is None:
    self.device = torch.device("cuda", torch.cuda.current_device())
```

第二行那个 `if` 容易被忽略，但很关键：**`torch.device("cuda")` 没有具体的卡号**，而后面所有 `torch.cuda.get_device_properties(self.device)`、`set_per_process_memory_fraction(...)` 都需要确定的卡。**统一成 `cuda:0` 这样的具体设备，是为了让后续所有 GPU 操作指向同一个地方。**

**第 3 件事：关闭一切"快但不精确"的开关（`pipeline.py:139-144`）**

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")
```

**这一段是全文件"教学价值"最高的地方之一，我请大家重点理解。**

| 开关 | 默认值 | 关掉之后 |
|---|---|---|
| `cudnn.benchmark` | `False`（但很多框架会开） | 不再自动挑最快的卷积算法 → **同一输入必得同一输出** |
| `cudnn.deterministic` | `False` | 强制确定性算法 |
| `allow_tf32`（matmul / cudnn） | `False` | 禁用 TF32 混合精度 → 精度回到完整 float32 |
| `allow_fp16_reduced_precision_reduction` | `False` | 禁止用低精度做累加 |
| `float32_matmul_precision` | `"highest"` | 矩阵乘法按最高精度算 |

**为什么一个音乐生成模型要这么较真？** 三个理由：

1. **可复现是硬需求。** 入门讲义反复强调"同样的 seed + 同样的输入 = 同样的歌"。而 TF32 会在矩阵乘法里悄悄把尾数截成 10 位（TF32 = 1 符号 + 8 指数 + 10 尾数），**同样的 seed 也可能给出不一样的 token**。
2. **TF32 的收益是"速度"，而本项目的瓶颈不在这里。** 矩阵乘法加速帮不上"一个 token 一个 token 地吐"这个瓶颈（本项目实测稳定在 107–109 token/s，见第 4 章），却能换来确定的复现性。**这是一个明确的取舍：用一点点速度换确定性。**
3. **这些是全局设置。** 所以 `__init__` 一旦执行，会影响整个进程里的 torch 行为。**副作用范围大，所以放在构造函数里、而不是散落在各处，是对的。**

**第 4 件事：加载 tokenizer、校验权重、算代码指纹（`pipeline.py:145-157`）**

```python
self.model_dir, self.vae_dir = Path(model_dir), Path(vae_dir)
self.tokenizer = YuE2TextTokenizer(self.model_dir / "qwen.tiktoken")
with self._status("Verifying model files"):
    self.weights = {"mot": model_identity(self.model_dir, verify_hashes),
                    "vae": model_identity(self.vae_dir, verify_hashes)}
self.runtime_sha256 = identity({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
```

**`runtime_sha256` 值得专门讲。** 它把 `yue2/` 包目录下**所有 `.py` 文件**的哈希汇总成一个指纹，最终写进 `config.json`。这解决了一个很实际的问题：

> "三个月后我想复现今天这首听起来不错的歌——我当时的代码是哪个版本？"

有了这个字段，答案就在 `config.json` 里。**"代码版本"从此和"权重版本"、"配置版本"一样，是一个可追溯的事实。** 这三样东西合起来，才构成一次可复现的运行。

**`verify_hashes=True` 的代价**：实测 `result.json` 里 `load.resolve_and_integrity_seconds = 21.63` 秒——**这 21 秒几乎全是算 7.2 GB 权重的哈希**。如果只是快速试听，用 `--no_verify_hashes` 可以省掉。**但请想清楚你在换什么：省下的是完整性保证。**

**第 5 件事：显存预算的换算（`pipeline.py:158-165`）**

```python
total = torch.cuda.get_device_properties(self.device).total_memory
budget = min((self.memory_budget_gib - 2) * 2**30, total - 2 * 2**30)
if budget <= 0:
    raise ValueError("Memory budget must leave room for a 2GiB reserve")
torch.cuda.set_per_process_memory_fraction(min(budget / total, 1), self.device)
```

**请把这条公式翻译成中文：**

```
实际可用显存 = min(你设的预算 - 2 GiB, 显卡总容量 - 2 GiB)
```

那 2 GiB 是**给系统留的余量**（显示、驱动、其他进程）。所以：

| 你怎么设 | 实际结果 |
|---|---|
| `budget = 24`，显卡 24 GB | `min(22, 22) = 22` GiB |
| `budget = 48`，显卡 24 GB | `min(46, 22) = 22` GiB（**自动夹到上限，设大无害**） |
| `budget = 2`，任意显卡 | `min(0, ...) = 0` → **直接报错** |
| `budget = 1.5` | `not 0 < 1.5` 为 False → **直接报错** |

**"设大无害、设小报错"** 这个行为非常体贴：用户不需要知道显卡多大。

**第 6 件事：`vae_core_frames` 的自动取值（`pipeline.py:148`）**

```python
self.vae_core_frames = vae_core_frames if vae_core_frames is not None else (512 if memory_budget_gib <= 12 else 1024)
```

**显存紧张 → 分块更小 → 峰值更低。** 512 帧的块大约是 1024 帧块的一半。**这是一个把"显存预算"翻译成"解码策略"的转换点**，也是全文件里少见的"资源预算 → 算法参数"的联动。想深入了解这个块长如何影响解码，看 `nar.py` 与 `modeling_vae.py` 的 `decode_tiled`。

#### 输出

**构造函数没有返回值，它造出的是一个 `YuE2Pipeline` 实例。** 但请记住，这个实例构造完**并没有把模型加载到 GPU 上**（`self._model = None`），**权重校验花了 21 秒，但 7 GB 权重还在硬盘上。** 真正的加载发生在第一次需要跑模型的时候（`_load_model`，`pipeline.py:210-223`）。

**这是一个重要的性能观念：构造 pipeline ≠ 开始推理。** 构造花的是"校验时间"，推理花的是"计算时间"。

### 3.2 `from_pretrained` —— 推荐的入口

源码位置：`pipeline.py:167-189`

```python
@classmethod
def from_pretrained(cls, model="m-a-p/YuE2-3B", *, vae="m-a-p/YuE2-Vae",
                    revision=None, vae_revision=None, local_files_only=False,
                    token=None, cache_dir=None, progress=True, **kwargs):
```

#### 输入输出

| 项 | 说明 |
|---|---|
| **输入** `model` | HF 仓库 ID（如 `"m-a-p/YuE2-3B"`）或本地目录路径 |
| **输入** `vae` | VAE 的仓库 ID 或路径 |
| **输入** `revision` / `vae_revision` | 指定仓库的某个 commit / tag |
| **输入** `local_files_only` | 只用本地缓存，不联网 |
| **输入** `token` | HuggingFace 访问令牌（私有仓库用） |
| **输入** `cache_dir` | 下载缓存目录 |
| **输入** `progress` | 是否显示进度 |
| **输入** `**kwargs` | **原样透传给 `__init__`**（`device`、`backend`、`memory_budget_gib`…） |
| **输出** | `YuE2Pipeline` 实例 |

#### 它做的三件事

**第 1 件：识别"本地导出的 pipeline 目录"（`pipeline.py:175-182`）**

```python
saved = Path(model) / "pipeline.json"
if saved.is_file():
    metadata = json.loads(saved.read_text())
    parent = Path(model)
    model = parent / metadata["model"]
    if vae == "m-a-p/YuE2-Vae":
        vae = parent / metadata["vae"]
    kwargs.setdefault("generation_config", GenerationConfig.from_dict(metadata["generation_config"]))
```

这段在配合 `save_pretrained()`（3.3 节）使用。**如果你用 `save_pretrained` 导出了一整套东西，那么 `from_pretrained("那个目录")` 就能把它整个装回来**——模型、VAE、生成配置，一样不缺。

**注意 `kwargs.setdefault` 这个写法**：只有当调用者**没有**显式指定 `generation_config` 时，才用目录里保存的那份。**调用者的显式设置永远优先。**

**第 2 件：解析模型路径（`pipeline.py:183-186`）**

```python
hub = dict(local_files_only=local_files_only, token=token, cache_dir=cache_dir)
with Progress(enabled=progress).stage("Resolving model files"):
    model_path = resolve_model(model, revision=revision, **hub)
    vae_path  = resolve_model(vae,   revision=vae_revision, **hub)
```

`resolve_model`（`storage.py:43-54`）的逻辑很简单：**是目录就返回它；否则去 HuggingFace 下载。**下载时用的是**白名单**（`MODEL_FILES`，`storage.py:33-38`）——只有明确列出的文件才会被拉下来，注释写得很清楚：

> *"Explicit public model files. Extra reports, caches and arbitrary code are not exported."*

**这既是体积优化，也是安全边界。**

**第 3 件：构造 + 记录耗时（`pipeline.py:187-189`）**

```python
result = cls(model_path, vae_path, progress=progress, **kwargs)
result.load_timing["resolve_and_integrity_seconds"] = time.perf_counter() - start
```

**教学要点**：注意 `cls(...)` 而不是 `YuE2Pipeline(...)`。**这是 `classmethod` 的标准写法**——将来即使有子类，调用 `SubClass.from_pretrained()` 也会造出 `SubClass`，而不是硬编码的父类。**这是一个很容易被忽略、但很"专业"的细节。**

### 3.3 `save_pretrained` —— 导出一整套

源码位置：`pipeline.py:198-208`

| 项 | 说明 |
|---|---|
| **输入** | `directory`：目标目录 |
| **输出** | `None` |
| **前置条件** | **目标目录必须为空**，否则 `FileExistsError` |

```python
if directory.exists() and any(directory.iterdir()):
    raise FileExistsError("save_pretrained needs an empty pipeline destination")
```

**请体会这条规则的用意**：它**不覆盖、不合并**。目录非空就拒绝。理由是——你要么完全用一份新的，要么什么都别动。**半覆盖的目录是最难排查的状态**（不知道哪些文件是新的、哪些是旧的）。**宁可报错，也不给你一个说不清状态的结果。**

写出的结构：

```
<directory>/
├── YuE2-3B/          ← 主模型（按 MODEL_FILES 白名单导出）
├── YuE2-Vae/         ← VAE
└── pipeline.json     ← {"model": "YuE2-3B", "vae": "YuE2-Vae",
                          "generation_config": {...}, "source_weights": {...}}
```

`copy_model_files`（`storage.py:57-102`）在做导出时有一个**特别值得学的安全处理**（`storage.py:94-95`）：

```python
# Use the installed, reviewed implementation instead of arbitrary source-folder code.
shutil.copyfile(Path(__file__).with_name(code_name), destination / code_name)
```

**翻译**：源模型目录里可能带着一份 `modeling_yue2.py`（自定义模型代码）。导出时，**官方包不复制那份文件，而是复制自己这份"经过审阅的实现"**。

**为什么？这是防止供应链攻击。** 一个模型仓库可以附带任意 Python 文件；如果你 `trust_remote_code=True` 直接加载，它就能在你机器上执行任意代码。官方包的做法是：**权重来自仓库，代码只来自官方包。** 这是一条清晰的安全边界。

### 3.4 三个内部方法

#### 3.4.1 `_status(label, *, total=None, unit=None)`

源码位置：`pipeline.py:191-196`

```python
@contextmanager
def _status(self, label, *, total=None, unit=None):
    # Display state never enters the request/config identity or model RNG.
    with Progress(enabled=self.progress) as reporter:
        with reporter.stage(label, total=total, unit=unit) as stage:
            yield stage
```

**它是一个上下文管理器（`@contextmanager` 装饰的生成器）。** 用起来是这样：

```python
with self._status("Decoding audio", total=tiles, unit="chunks") as status:
    ...
    status.update(1)      # 设置绝对进度
    status.advance()      # 相对进度 +1
    status.finish(status="truncated")
```

**请务必读那句注释**：

> *"Display state never enters the request/config identity or model RNG."*
> （显示状态**永远不进入**请求/配置的身份计算，也**永远不进入**模型的随机数生成。）

**翻译成大白话：进度条是"给人看的旁白"，它绝不能影响"给机器看的结果"。**

为什么这条如此重要？因为 `request_identity` 是 `sha256(请求 + 配置 + 权重)`。如果进度条的累计计数被不小心塞进了配置字典里，那么**同一首歌在开了进度条和关了进度条的情况下会得到不同的 `identity`**——这就破坏了可复现性。**进度显示是"旁白"，旁白不能入正文。**

> **教学要点：这是本项目给我印象最深的一条工程原则。我把它推荐给你们作为准则：任何"展示层"的状态（进度、日志、计时动画），都必须与"计算层"的状态严格隔离。** 很多真实项目在这里栽过跟头——日志里多记了一个时间戳，导致缓存键变了，缓存永远不命中。

**顺带一个使用约定**：进度信息走 **stderr**，结果走 **stdout**。这样你 `> result.json` 重定向时，进度条不会污染结果文件。

#### 3.4.2 `_request(style=None, lyrics=None, *, tags=None, **kwargs)`

源码位置：`pipeline.py:225-231`

```python
def _request(self, style=None, lyrics=None, *, tags=None, **kwargs):
    if style is not None and tags is not None and style != tags:
        raise ValueError("style and tags are aliases and cannot disagree")
    style = tags if style is None else style
    if style is None or lyrics is None:
        raise ValueError("Provide style and lyrics")
    return SongRequest(style=style, lyrics=lyrics, **kwargs)
```

**这是全文件最短、却最值得学的函数之一。**

`style` 和 `tags` 是**同一个东西的两个名字**（别名）。为什么会有别名？看命名来源：`style` 是模型时代的叫法，`tags` 更贴近"逗号分隔的标签串"这个实际形态。两个都留着是为了方便。

**但别名有个经典陷阱**：如果两个都给了，而且**不一样**怎么办？两种做法：

- ❌ 随便挑一个（用户会困惑："我明明传了两个，到底用了哪个？"）
- ✅ **报错**（本文件的选择）

**教学要点：别名冲突时，宁可报错也不要猜。** 猜错了，用户要等到听完整首歌才发现参数不对；当场报错，一秒钟就知道。这个原则在 CLI 工具、配置解析里同样适用。

返回的是一个 `SongRequest`（`protocol.py:80-112`），它在构造时还会做一轮完整校验（`cot` 合法、`seed` 是 `[0, 2**63)` 的整数、`id` 是文件名安全的标识符、`abc` 与 `cot` 的相容性、`cfg_scale ∈ [0, 20]`）。

#### 3.4.3 `_generate(prefix, sampling, seed, phase, **kwargs)`

源码位置：`pipeline.py:233-253`

| 项 | 说明 |
|---|---|
| **输入** `prefix` | AR 前缀（`list[int]`） |
| **输入** `sampling` | 采样配置（`Sampling` 对象） |
| **输入** `seed` | 随机种子 |
| **输入** `phase` | `"abc"` 或 `"semantic"`——**决定用哪套采样规则和哪个结束符** |
| **输入** `**kwargs` | 透传给底层，含 `on_token`、`cancelled`、`negative`、`cfg_scale` |
| **输出** | 三元组 `(ids, timing, truncated)` |

```python
model = self._load_model() if self.backend != "vllm" else None
callback = kwargs.pop("on_token", None)
label = "Planning score" if phase == "abc" else "Generating song"
with self._status(label, unit="tokens") as status:
    def on_output(token_phase, token):
        status.advance()
        if callback is not None:
            callback(token_phase, token)
    observed = on_output if self.progress else callback
    if self.backend == "vllm":
        from .fast import generate_vllm
        result = generate_vllm(self, prefix, sampling, seed, phase, on_token=observed, **kwargs)
    else:
        result = generate_tokens(model, prefix, sampling, seed, phase,
                                 use_cuda_graph=self.backend != "torch-eager", on_token=observed, **kwargs)
    if result[2]:
        status.finish(status="truncated")
    return result
```

**这里有四个值得学的点：**

**（1）`phase` 是一等公民。** 同一个函数，靠 `phase` 参数切换两套完全不同的行为：

| | `phase="abc"` | `phase="semantic"` |
|---|---|---|
| 允许的 token | 普通文本词表（`< EOD`） | 音频 codec 区间 `[CODEC_OFFSET, CODEC_OFFSET+CODEC_SIZE)` |
| 结束符 | `ABC_END` (151848) | `MUSIC_END` (151852) |
| 进度标签 | "Planning score" | "Generating song" |

**这个"用参数把两种模式收进一个函数"的手法，是本文件 API 设计的核心。** 你会看到 `plan()` 和 `generate_semantic()` 长得极像，就是因为它们最后都调 `_generate`。

**（2）`observed = on_output if self.progress else callback`** —— **这是一行非常漂亮的代码。**

它的意思是：**进度开启时，用"既推进度、又转给用户回调"的包装函数；进度关闭时，直接用用户回调（或 `None`）。**

为什么要这么绕？因为如果直接传 `on_output`，那 `progress=False` 时每生成一个 token 也要走一次 Python 函数调用和一次加锁——**在几十万 token 的规模上，这是纯粹的浪费**。而用户自己的 `on_token` 回调是**外部契约**，即使进度关闭也必须被调用。**所以：内部状态按需付费，外部契约始终兑现。**

**（3）`status.finish(status="truncated")`** —— 截断要**如实上报**，不能装作正常结束。

**（4）`from .fast import generate_vllm` 是函数内导入。** 这不是随意的写法——`fast.py` 要 import vLLM，而 vLLM 是可选依赖。**把可选依赖的导入推迟到真正需要它的那一行**，才能让不用 vLLM 的人完全不受影响。（`__init__.py:5-16` 用 `__getattr__` 做了同样的事：`from yue2 import YuE2Pipeline` 不会连带导入 torch 以外的重型模块。）

### 3.5 `plan()` —— 第一阶段：写谱子

源码位置：`pipeline.py:255-269`

```python
def plan(self, style=None, lyrics=None, *, tags=None, request=None, abc_sampling=None,
         cancelled=None, on_token=None, **kwargs):
```

#### 输入输出

| 项 | 说明 |
|---|---|
| **输入** `style` / `tags` | 风格描述（二者别名） |
| **输入** `lyrics` | 歌词正文 |
| **输入** `request` | **可以直接传一个构造好的 `SongRequest`**（与上面三个二选一） |
| **输入** `abc_sampling` | 覆盖 ABC 阶段的采样配置（`Sampling` 或 `dict`） |
| **输入** `cancelled` | 可调用对象，返回 `True` 则中断（抛 `InterruptedError`） |
| **输入** `on_token` | 每产出一个 token 就回调一次 `(phase, token)` |
| **输出** | `SymbolicPlan` |

#### 它的三条分支

```
                 plan()
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
 request.cot == "off" ?    request.abc 不为 None ?
      │ 是                    │ 是
      ▼                       ▼
 不规划，直接返回           用你给的谱子
 abc=None, abc_ids=[]       encode → ids → 组前缀
      │                       │
      └───────────┬───────────┘
                  ▼ 否
        真的调用模型生成 ABC
```

**分支一：`cot == "off"`（`pipeline.py:258-259`）**

```python
if request.cot == "off":
    return SymbolicPlan(request, None, [], token_prefixes(request, self.tokenizer))
```

注意：**它返回的 `SymbolicPlan` 是"合法的空计划"**，不是 `None`。`abc=None`、`abc_ids=[]`，但 `prefix` 完整正确。**这样后面 `generate_semantic()` 拿到的接口和 `full` 模式完全一样**——下游代码不需要知道上游有没有规划。**这是一个非常好的设计：用"空但合法"代替"特殊情况"。**

**分支二：提供了外部 ABC（`pipeline.py:260-264`）**

```python
with self._status("Using provided score"):
    ids = self.tokenizer.encode(request.abc)
    return SymbolicPlan(request, request.abc, ids, token_prefixes(request, self.tokenizer, ids),
                        {"seconds": 0., "output_tokens": 0, "external_prefix_tokens": len(ids)})
```

**教学要点一**：耗时被记成 `{"seconds": 0., "output_tokens": 0}`。**这个 0 是诚实的**——没跑模型，就是没跑模型。**不要为了"看起来跑过了"而伪造一个耗时。**

**教学要点二**：额外记了一个 `external_prefix_tokens`。**为什么？** 因为这一支没有 `output_tokens`，如果你只看 `output_tokens` 会以为是 0，得再翻一个字段才知道是"外部提供"。**在指标里区分"没做"和"做了但结果是 0"，是一种成熟的做法。**

**分支三：真的去生成（`pipeline.py:265-269`）**

```python
sampling = resolve_sampling(abc_sampling, self.generation_config.abc)
ids, timing, truncated = self._generate(token_prefixes(request, self.tokenizer), sampling,
                        request.seed, "abc", cancelled=cancelled, on_token=on_token)
return SymbolicPlan(request, self.tokenizer.decode(ids), ids,
                    token_prefixes(request, self.tokenizer, ids), timing, truncated)
```

**请注意这里的 `token_prefixes` 被调用了两次，参数不同：**

- 第一次：`token_prefixes(request, self.tokenizer)` —— **不带 `abc_ids`**，意思是"接下来要生成 ABC"，所以前缀只到 `[ABC_START]` 为止。
- 第二次：`token_prefixes(request, self.tokenizer, ids)` —— **带上刚生成的 `ids`**，意思是"ABC 已经有了"，所以前缀补全成 `[ABC_START] + ids + [ABC_END, MUSIC_START]`。

**这是整个 pipeline 里最精妙的一处设计**，请务必理解：

> **规划阶段的前缀和执行阶段的前缀，天然不同。** 规划时模型只能看到"文字 + 我要开始写谱了"；执行时模型必须看到"文字 + 完整的谱子 + 现在开始写音乐"。

`token_prefixes` 用"**第三个参数在不在**"来区分这两种语义（`protocol.py:115-126`），**没有引入额外的模式标志位**。**用参数的有无来表达状态，是 Python 里很地道的做法**——`None` 天然是一个合法状态。

**`resolve_sampling` 是什么？**（`protocol.py:70-77`）

```python
def resolve_sampling(value, default):
    if value is None:      return default
    if isinstance(value, Sampling): return value
    if isinstance(value, dict):    return Sampling(**{**asdict(default), **value})
    raise TypeError("Sampling must be a Sampling object or a dictionary of overrides")
```

**它允许你传 `None`（用默认）、传 `Sampling` 对象（全量覆盖）、或者传一个字典（部分覆盖，其余沿用默认）。** 第三种最实用：

```python
abc_sampling={"temperature": 0.9}   # 只改温度，其他 6 个参数保持默认
```

### 3.6 `generate_semantic()` —— 第二阶段：写语义 token

源码位置：`pipeline.py:271-283`

```python
def generate_semantic(self, plan, *, sampling=None, cancelled=None, on_token=None):
```

#### 输入输出

| 项 | 说明 |
|---|---|
| **输入** `plan` | **必须是 `pipe.plan()` 返回的 `SymbolicPlan`**（否则 `TypeError`） |
| **输入** `sampling` | 覆盖语义阶段采样配置 |
| **输入** `cancelled` / `on_token` | 同上 |
| **输出** | `SemanticResult` |

#### 它的五步

**第 1 步：类型检查（`pipeline.py:272-273`）**

```python
if not isinstance(plan, SymbolicPlan):
    raise TypeError("Pass the SymbolicPlan returned by pipe.plan()")
```

**教学要点**：错误信息是**祈使句 + 明确出处**——"传 `pipe.plan()` 返回的 `SymbolicPlan`"。**好的报错信息应该让调用方一眼知道该改哪一行。**

**第 2 步：前缀一致性校验（`pipeline.py:274-277`）** ★

```python
request = plan.request
expected = token_prefixes(request, self.tokenizer, plan.abc_ids)
if expected != plan.prefix:
    raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
```

**请在这里停一停，想一想：为什么要在跑模型之前做这个检查？**

场景是这样的：你 `plan()` 完，把计划存盘，交给同事；同事手滑改了 `plan.json` 里的 `abc_ids`，或者把 `abc` 文本改了但没同步 `abc_ids`；又或者你用了 `abc_sampling` 覆盖了采样但没意识到前缀也变了。

**如果不检查**：你会带着一个"自相矛盾"的前缀去生成语义 token——模型看到的前缀和它应该看到的不一样。**结果不会报错，而是生成一首歌——但那首歌的调式、和弦走向，可能莫名其妙。** 这种 bug 极难排查。

**如果检查**：立刻 `ValueError`，告诉你"计划和请求/ABC 不符"。

**教学要点：这是我最想传授给你的一条工程经验——**

> **在流水线里，阶段之间传递的每一份数据，都应该在交接处被重新验证一次。**
>
> 理由很简单：**数据可能被外部改过**（人改的文件、别的进程、缓存）。你在上一阶段写进去时是对的，不代表现在还是对的。

**校验的方法也很讲究**：不是"检查我记的标记位"，而是**"用源数据重新推导一遍，然后和传递下来的结果比对"**。这叫**重算校验**，比"相信一个标志位"强得多。

**第 3 步：准备负分支（`pipeline.py:278-279`）**

```python
sampling = resolve_sampling(sampling, self.generation_config.semantic)
negative = negative_prefix(request, self.tokenizer, plan.abc_ids) if request.guidance != 1 else None
```

**CFG（Classifier-Free Guidance）** 的思路是：同时算"有条件"和"无条件"两个分支，然后用

```
logits = unconditional + cfg_scale × (conditional − unconditional)
```

把结果往"有条件"的方向推。`cfg_scale` 越大，推得越狠。

**注意 `!= 1` 这个判断**：`cfg_scale` 等于 1 时，公式退化（`uncond + 1×(cond−uncond) = cond`），**负分支纯属浪费算力**。所以这里直接置 `None`，**根本不构造负前缀**。

**符号模式下负分支必须带 ABC**（`protocol.py:133-134` 会报错）——因为负分支要**保留谱子**，只是去掉歌词和风格。**这样"引导"的方向才是"让歌词和风格去贴合谱子"，而不是"让谱子乱掉"。**

`cfg_negative` 这个字段最终会被写进 `config.json`，值为 `"instruction_only"`（`cot="off"`）或 `"same_instruction_and_exact_abc"`（符号模式）。**它明确记录了"负分支长什么样"**，因为这直接影响结果，而别人拿到你的 `config.json` 是没法自己推出来的。

**第 4 步：生成（`pipeline.py:280-282`）**

```python
ids, timing, truncated = self._generate(plan.prefix, sampling, request.seed, "semantic",
                negative=negative, cfg_scale=request.guidance, legacy_off=request.cot == "off",
                cancelled=cancelled, on_token=on_token)
```

**注意 `legacy_off=request.cot == "off"`。** 这是一个"历史行为兼容开关"：`off` 模式的采样用 BF16 而不是 FP32 计算，并且 top-p 裁剪时不屏蔽最前面的几个 token。**这是为了精确复现官方预设的历史数值行为。**

> **教学要点：这类"看起来很奇怪但必须保留"的代码，在真实项目里很常见。** 它存在的唯一理由是"改了就和历史结果对不上"。**读这种代码时，不要试图"优化"它，要先搞清楚它为什么必须是这样。**

**第 5 步：编号偏移（`pipeline.py:283`）** ★

```python
return SemanticResult(plan, [int(t) - CODEC_OFFSET for t in ids], timing, truncated)
```

**这一行做了两件事，请都记住：**

**（1）减 `CODEC_OFFSET`（= 151853）**：模型输出的 codec token 编号是在**完整词表**里的编号（`151853 ~ 184620`）。存盘时把它们**平移回 0 ~ 32767**，是为了让 `semantic.npy` 更紧凑、更易读，也方便其他工具直接用。

**（2）`int(t)`**：把 `torch` 标量转成 Python `int`。**为什么强调这一点？** 因为如果直接存 numpy 数组，元素 dtype 可能是 `int32` 或 `int64`；但转成 Python `int` 后，`np.save(..., dtype=np.int32)` 存出来的就是**固定 32 位**，跨机器一致。**在"要存盘、要跨机器比对"的地方，显式指定 dtype 是基本功。**

### 3.7 `synthesize()` —— 第三阶段：出声学 latent

源码位置：`pipeline.py:285-304`

```python
def synthesize(self, semantic, *, cancelled=None):
```

| 项 | 说明 |
|---|---|
| **输入** `semantic` | **必须是 `generate_semantic()` 返回的 `SemanticResult`** |
| **输入** `cancelled` | 中断检查 |
| **输出** | `np.ndarray`，形状 `[frames, 64]`，dtype **float32**，在 **CPU** 上 |

#### 它的六步

**第 1 步：vLLM 后端需要先关闭（`pipeline.py:287-289`）**

```python
if self.backend == "vllm":
    from .fast import close_vllm
    close_vllm(self)
```

**为什么？** 因为 vLLM 会**独占显存**（它自己预分配了一大块做 KV cache）。如果不关掉，声学阶段就没显存可用。**这是多后端共存时的资源交接问题。**

**第 2 步：类型检查（`pipeline.py:290-291`）**

```python
if not isinstance(semantic, SemanticResult):
    raise TypeError("Pass the SemanticResult returned by generate_semantic()")
```

**第 3 步：前缀一致性校验（`pipeline.py:292-293`）** ★

```python
if token_prefixes(semantic.plan.request, self.tokenizer, semantic.plan.abc_ids) != semantic.plan.prefix:
    raise ValueError("Semantic result does not retain the request's exact prefix")
```

**和 `generate_semantic` 里那次一模一样的校验，为什么要做两遍？**

**这是本文件里我要求你们特别注意的一个设计。** 原因是：**两处校验守护的是两道不同的门。**

- `generate_semantic` 的校验守护的是"**入口**"：确认你传进来的计划是自洽的。
- `synthesize` 的校验守护的是"**出口**"：确认**已经生成的**语义结果，其携带的计划仍然自洽。

**中间可能发生什么？** 有人可能这样写：

```python
sem = pipe.generate_semantic(plan)
sem.plan.prefix = [999, 999, 999]     # 手滑改坏了
latents = pipe.synthesize(sem)        # ← 如果没有第二次校验，这里就悄悄用错前缀渲染了
```

**更现实的场景**：你的代码中间做了某种"优化"，比如重新构造了一个 `SemanticResult`，或者从磁盘反序列化时丢了字段。**第二道门确保无论中间发生了什么，最终渲染用的前缀一定是和请求一致的那个。**

> **教学要点：记住这个模式——**
> **每一次"跨越阶段边界"，就校验一次。不要指望"我刚检查过了，后面就不会坏"。**
> 这是防御性编程的核心思想：**信任边界处做校验，内部假设自洽。**

**第 4 步：恢复 FP8 权重（`pipeline.py:294-296`）**

```python
if self.quantization != "none":
    from .quantization import restore_ar
    restore_ar(self._model)
```

**FP8 量化只对 AR（自回归、生成 token）阶段有效；声学阶段需要原始精度。** 所以进入声学阶段前必须把权重换回去。**量化是"局部"的，不是"全局"的——这一点在很多量化方案里都成立，务必记住。**

**第 5 步：加载模型（`pipeline.py:297`）**

```python
model = self._load_model(for_nar=True)
```

注意 `for_nar=True`——**加载时不能套用 FP8 量化**（和上面的 `restore_ar` 是配套的）。

**第 6 步：真正的合成（`pipeline.py:298-304`）**

```python
with self._status("Synthesizing audio", unit="steps") as status:
    report = (lambda completed, total: status.update(completed, total=total)) if self.progress else None
    result = synthesize(model, semantic.plan.prefix, semantic.tokens,
                        semantic.plan.request.seed, steps=self.generation_config.ode_steps,
                        context=self.generation_config.context, offload_ar=self.offload_ar,
                        cancelled=cancelled, on_progress=report)
    return result.detach().float().cpu().numpy()
```

几个细节：

- **`steps=self.generation_config.ode_steps`（默认 32）** —— 这是 ODE 求解器的步数。**步数越多越准，但也越慢**，这是一个明确的精度/耗时权衡（32 步是官方预设）。
- **`report = (...) if self.progress else None`** —— 和 `_generate` 里同一个思路：**内部进度按需付费**。
- **`.detach().float().cpu().numpy()`** —— 离开计算图（不需要梯度）、转 fp32、搬到 CPU、变成 numpy。**`.detach()` 一定要有**，否则你会得到一个带着整张计算图的张量，内存占用爆炸。

**输出形状再强调一次**：`[frames, 64]`。**`frames` 的数量等于语义 token 的数量**（实测：3888 个语义 token → 3888 帧）。**64 是声学 latent 的通道数。** 这个对应关系是理解整个 NAR 阶段的关键。

### 3.8 `decode()` —— 第四阶段：还原成波形

源码位置：`pipeline.py:320-356`

```python
def decode(self, latents, *, full=False, vae=None):
```

| 项 | 说明 |
|---|---|
| **输入** `latents` | `[T, 64]` 或 `[1, 64, T]` |
| **输入** `full` | `True` = 一次性解码整段（快但费显存）；`False` = 分块解码（默认） |
| **输入** `vae` | 可指定用别的解码器（路径或已加载对象）；`None` = 用 `self.vae_dir` |
| **输出** | `np.ndarray`，形状 `[samples, 2]`，float32，范围 `[-1, 1]` |

#### 它的五步

**第 1 步：给 VAE 腾地方（`pipeline.py:322-326`）** ★

```python
with self._status("Loading audio decoder"):
    if self._model is not None:
        self._model.to("cpu")
    if self.device.type == "cuda":
        torch.cuda.empty_cache()
```

**这是全文件最"反直觉"的一段，请重点理解。**

明明下一步要用 GPU 跑 VAE，为什么**第一步是把模型从 GPU 搬走**？

**答案：GPU 显存是有限的、且是共享的。** 7 GB 的主模型还在显存里，VAE 解码时就没有足够的空间了——尤其在长音频上。**所以这里的策略是"用完就还"**：AR/NAR 阶段用完显存，就把模型挪回 CPU 挂着，等需要时再搬回来。

`self._model.to("cpu")` **不删除模型**，只是换了个存放位置——所以下次用的时候还在，只是要重新搬一次。这个"**搬而不是删**"的选择，是为了避免反复的磁盘 IO。**这是显存与速度之间的一次典型权衡，注释都没写，但代码写得很清楚。**

**`torch.cuda.empty_cache()` 紧跟其后**：把 PyTorch 缓存里已经空闲的显存块还给驱动。**注意 `empty_cache()` 只是释放"缓存"，不是"正在用的显存"。** 所以必须先 `.to("cpu")` 再 `empty_cache()`，顺序不能反。

**第 2 步：准备 VAE（`pipeline.py:327-333`）**

```python
if vae is not None:
    model = YuE2VAE.from_pretrained(vae, decoder_only=True, device=self.device)
else:
    if self._vae is None:
        self._vae = YuE2VAE.from_pretrained(self.vae_dir, decoder_only=True, device="cpu",
                                            local_files_only=True)
    model = self._vae.to(self.device)
```

**`decoder_only=True`** —— 只加载解码器部分，扔掉编码器。**我们只需要"从 latent 还原成波形"这一个方向。** 省显存，省时间。

注意默认路径是**先在 CPU 上加载，再 `.to(device)`**。**为什么不在目标设备上直接加载？** 因为直接加载到 GPU 会在加载过程中产生一份 CPU 副本 + 一份 GPU 副本，峰值内存翻倍。**先 CPU 后 GPU 是更稳的做法。**

**第 3 步：形状归一化（`pipeline.py:334-338`）**

```python
z = torch.as_tensor(latents, dtype=torch.float32)
if z.ndim == 2 and z.shape[1] == 64:
    z = z.T.unsqueeze(0)
if z.ndim != 3 or z.shape[0] != 1 or z.shape[1] != 64:
    raise ValueError("Expected latents [T,64] or [1,64,T]")
```

**它接受两种形状**，都归一化成 `[1, 64, T]`（PyTorch 卷积要的 `[batch, channel, time]`）：

| 你给的 | 它理解成 |
|---|---|
| `[T, 64]` | 转置成 `[64, T]`，再补 batch 维 → `[1, 64, T]` |
| `[1, 64, T]` | 原样保留 |

**为什么要支持两种？** 因为 `synthesize()` 返回的是 `[T, 64]`（numpy 习惯，时间在前），而 PyTorch 习惯 `[C, T]`。**这是一个"在两套惯例之间做翻译"的地方。** 接受两种输入、只输出一种，是很体贴的 API 设计。

**第 4 步：解码（`pipeline.py:339-349`）**

```python
try:
    tiles = 1 if full else (z.shape[-1] + self.vae_core_frames - 1) // self.vae_core_frames
    with self._status("Decoding audio", total=tiles, unit="chunks") as status:
        report = (...) if self.progress else None
        with torch.inference_mode():
            if full:
                audio = model.decode(z.to(self.device)).cpu()
                status.update(1)
            else:
                audio = model.decode_tiled(z, core_frames=self.vae_core_frames, halo_frames=16,
                                           output_device="cpu", on_progress=report)
```

**教学要点一：`tiles` 的算法。**

```
tiles = ceil(T / core_frames)
```

`core_frames` 默认 1024（显存 ≤ 12 GiB 时是 512）。**155 秒的音频有 3888 帧，所以默认会切成 4 块。**

**教学要点二：`halo_frames=16`。** 每块解码时，会**多带 16 帧上下文**（前后各 16），解完再把这段"halo"裁掉。**为什么？** 因为 VAE 的卷积核有感受野，块与块的边界会因为缺少上下文而产生接缝。**多算一点再裁掉，是消除接缝的标准做法。** `config.json` 里记着 `"vae_decode": "halo_crop"`——**这就是那个技术的名字。**

**教学要点三：`output_device="cpu"`。** 解码输出**直接写进 CPU 内存**，不经过 GPU。**为什么？** 因为一整首歌的波形在 GPU 上会占一大块显存，而这些数据我们马上就要交给 CPU 用。**让数据"只走必要的路"，是显存工程的基本功。**

**教学要点四：`torch.inference_mode()`。** 比 `no_grad()` 更彻底——连自动求导的中间状态都不记录。**推理时必须用。**

**第 5 步：校验与归一（`pipeline.py:350-352`）**

```python
if not torch.isfinite(audio).all():
    raise ValueError("VAE produced non-finite audio")
return audio[0].float().clamp(-1, 1).T.contiguous().numpy()
```

- **`isfinite` 检查**：`NaN` 或 `Inf` 会让音频变成刺耳的噪声，或者干脆播放失败。**在输出边界上做一次"数值是否合法"的检查，是负责任的做法。**
- **`.clamp(-1, 1)`**：浮点解码可能产生略微超出 `[-1, 1]` 的值（比如 1.0000001）。**不裁剪的话，某些播放器会削波失真。**
- **`.T` 转置**：从 `[channels, samples]` 转成 `[samples, channels]`，**和 numpy 的"每行一个采样点"习惯对齐**。
- **`.contiguous()`**：转置后内存不连续，`.contiguous()` 把它整理成连续内存，**后续的 numpy 转换和写盘会更快**。

**最后 `finally` 块（`pipeline.py:353-356`）**：

```python
finally:
    model.to("cpu")
    if self.device.type == "cuda":
        torch.cuda.empty_cache()
```

**注意用的是 `finally`——无论解码成功还是抛异常，VAE 都会被搬回 CPU 并释放缓存。** **这就是"用完就还"的纪律性实现。** 如果这里写成普通的 `model.to("cpu")`（而不是放在 `finally`），一旦解码中途报错，VAE 就会**一直占着显存**，后续操作可能直接 OOM。**这是一个很小、但非常能体现工程素养的细节。**

### 3.9 `effective_config()` —— 把"这次到底用了什么"记下来

源码位置：`pipeline.py:358-376`

```python
def effective_config(self, request, abc_sampling=None, semantic_sampling=None):
```

**输入**：一个 `SongRequest`，可选的两个采样覆盖。
**输出**：一个 `dict`，最终被写进 `config.json`。

#### 它算了两样东西

**（1）`generation`：完整的生效配置**

```python
config = self.generation_config.to_dict()
for key, sampling in (("abc", abc_sampling), ("semantic", semantic_sampling)):
    if sampling is not None:
        config[key] = dataclasses.asdict(resolve_sampling(sampling, getattr(self.generation_config, key)))
```

**（2）`overrides`：相对默认值改了什么**

```python
defaults = GenerationConfig().to_dict()
overrides = {k: v for k, v in config.items() if defaults.get(k) != v}
```

**这个 `overrides` 字段设计得非常妙。** 想想看：`config.json` 里存了完整的 `generation`（16 个采样参数 + 4 个全局参数），你下次看这个文件，怎么知道哪些是被特别调过的？

- 如果只存 `generation`：你得逐个和默认值比。
- 如果只存 `overrides`：你不知道完整配置是什么。

**两个都存，就完美了。** 实测 `output/my_song/config.json` 里 `"overrides": {}` ——因为那次运行用的是纯默认配置。**空字典本身就是信息："这次什么都没改"。**

#### 剩下的字段

| 字段 | 含义 | 实测值 |
|---|---|---|
| `cot` | 规划模式 | `"full"` |
| `cfg_scale` | 引导强度 | `1.0` |
| `cfg_negative` | 负分支形态 | `"same_instruction_and_exact_abc"` |
| `backend` / `quantization` | 运行后端与量化 | `"torch"` / `"none"` |
| `model_dtype` / `vae_dtype` | 数值精度 | `"bfloat16"` / `"float32"` |
| `vae_decode` | 解码策略 | `"halo_crop"` |
| `vae_core_frames` / `vae_halo_frames` | 分块参数 | `1024` / `16` |
| `device` | 实际设备 | `"cuda:0"` |
| `memory_budget_gib` | 显存预算 | `24.0` |
| `offload_ar` | 是否卸载主干 | `false` |
| `runtime_sha256` | **代码版本指纹** | `facebc61...` |
| `decoder_release` | 解码器版本 | `"standard"` |
| `validation_status` | **是否经过官方验证** | `"unvalidated"` |

**请特别看最后两个字段，这是本文件"诚实性设计"的集中体现。**

- **`decoder_release`**：从 VAE 的 `config.json` 里读 `release_variant` 字段。它会告诉你是 `"standard"` 还是 `"legacy"`——**官方 benchmark 用的是 legacy 解码器**（见 `scripts/infer_yue2.py:65` 的说明），**用错解码器，指标就不可比**。

- **`validation_status: "unvalidated"`**：**这句话是诚实的自我标注。** 它在说：*"我把我用的所有配置都记下来了，但我没有承诺这个配置组合被官方验证过。"*

> **教学要点：这是我最想推荐给你们的一条职业素养——**
> **当一个系统有很多参数、且参数组合会影响结果质量时，请主动记录"这个组合有没有被验证过"。**
> 不要让使用者误以为"我记录了所有配置，所以结果一定可靠"。**记录的完备性和结果的可靠性是两回事。** 明确区分这两者，是一个负责任的库和一份负责任的文档之间的差别。

**顺带一提**：`effective_config` 里的 `cfg_scale` 判定逻辑是这样的：

```python
if request.guidance != (1.01 if request.cot == "off" else 1.0):
    overrides["cfg_scale"] = request.guidance
```

**只有当用户显式改了 `cfg_scale`，它才会出现在 `overrides` 里。** 默认值不算"改动"。**同一个道理：区分"设成了默认值"和"没设"。**

### 3.10 `__call__()` —— 一次到位

源码位置：`pipeline.py:378-398`

```python
def __call__(self, style=None, lyrics=None, *, tags=None, abc_sampling=None,
             semantic_sampling=None, cancelled=None, on_token=None, **kwargs):
```

**这个方法让 `pipe` 本身变成"可调用对象"**，用法是：

```python
song = pipe(style="...", lyrics="...")
```

**这叫 `__call__` 协议。** Python 里 `obj(...)` 会自动调用 `obj.__call__(...)`。**这是 Python 里表达"这个对象本身就是一个操作"的标准手法。**

#### 它的十步

```python
request = self._request(style, lyrics, tags=tags, **kwargs)                        # 1
config  = self.effective_config(request, abc_sampling, semantic_sampling)          # 2
request_id = identity({"request": request.to_dict(), "config": config, "weights": self.weights})  # 3
start = time.perf_counter()                                                        # 4
plan = self.plan(request=request, abc_sampling=abc_sampling, cancelled=cancelled, on_token=on_token)  # 5
semantic = self.generate_semantic(plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token)  # 6
nar_start = time.perf_counter()                                                   # 7
latents = self.synthesize(semantic, cancelled=cancelled)                          # 8
nar_seconds = time.perf_counter() - nar_start                                      # 9
if cancelled is not None and cancelled():
    raise InterruptedError("Cancelled before VAE")                                  # 10
vae_start = time.perf_counter()
audio = self.decode(latents)
timing = {"abc": plan.timing, "semantic": semantic.timing, "nar_seconds": nar_seconds,
          "vae_seconds": time.perf_counter() - vae_start, "load": dict(self.load_timing),
          "e2e_seconds": time.perf_counter() - start}
Progress(enabled=self.progress).complete(len(audio) / 48000, timing["e2e_seconds"],
                                        truncated=plan.truncated or semantic.truncated)
return SongResult(audio, 48000, semantic, latents, config, self.weights, timing, request_id)
```

**请注意第 1 步：`request = self._request(style, lyrics, tags=tags, **kwargs)`。**

而 `plan()` 内部也会调 `_request`。**这里会不会重复构造？** 不会——因为 `plan()` 是这样调用的：

```python
plan = self.plan(request=request, abc_sampling=abc_sampling, ...)
```

**`request` 已经被显式传进去了**，所以 `plan()` 里的 `request = request or self._request(...)` 会直接用传进来的那个（`pipeline.py:257`）。**`or` 短路在这里正好发挥了作用。** 这是一个很干净的写法：**既能自己造请求，也能接受外部造好的请求。**

**第 3 步：`request_id` 的构成。**

```python
identity({"request": request.to_dict(), "config": config, "weights": self.weights})
```

`identity()`（`storage.py:19-21`）做的事是：把字典**规范化成 JSON（键排序、无空格、禁止 NaN）**，然后算 sha256。

**请注意 `allow_nan=False`。** 如果配置里出现了 `NaN` 或 `Infinity`，JSON 序列化会直接报错，而不是生成一个"看起来正常"的哈希。**哈希函数最怕的就是输入不唯一——那会让"身份"失去意义。**

**这三个东西合在一起就是"一次运行的完整身份"：**

| 组成 | 回答的问题 |
|---|---|
| `request` | 我要生成什么？ |
| `config` | 我用了什么参数、什么代码、什么设备？ |
| `weights` | 我用的是哪份权重？ |

**三样都一样，才是"同一首歌"。** 少一样，就可能是另一首歌。

**第 10 步：中断检查的位置很有意思。**

```python
if cancelled is not None and cancelled():
    raise InterruptedError("Cancelled before VAE")
```

**为什么在 VAE 之前单独检查一次？** 因为 VAE 解码是**整段一起算**的，**一旦开始就没有中间点可以中断**。前面三个阶段都是 token 级或 step 级可中断的。**所以在进入一个"不可中断的长任务"之前，必须检查一次。**

**这个细节体现的是对"取消语义"的周全考虑**：一个好的取消机制，不只是"能停"，还要"在还来得及的时候停"。**如果在不可中断的阶段里才检查，用户按下 Ctrl+C 就得干等几十秒白跑。**

**最后一行 `Progress(...).complete(...)` 的 `len(audio) / 48000`** —— 用波形长度除以采样率得到音频秒数。**注意这里硬编码了 48000**，因为 `__call__` 内部固定按 48 kHz 输出。

**`truncated=plan.truncated or semantic.truncated`** —— 汇总两阶段的截断状态。**任一阶段被截断，整首歌就算截断。**

### 3.11 `close()` 与上下文管理器

源码位置：`pipeline.py:306-318`

```python
def close(self):
    if self.backend == "vllm":
        from .fast import close_vllm
        close_vllm(self)
    self._model, self._vae = None, None
    if self.device.type == "cuda":
        torch.cuda.empty_cache()

def __enter__(self): return self
def __exit__(self, *exc): self.close()
```

**`close()` 做三件事**：关掉 vLLM、把模型引用置 `None`、清 CUDA 缓存。

**`self._model = None` 的效果是"下次再用会重新加载"。** 这就是 `_load_model` 里那个判断的由来（`pipeline.py:211`）：

```python
loading = self._model is None or next(self._model.parameters()).device != self.device
```

**它同时检查两件事**：模型没加载，**或者**模型在别的设备上。**第二种情况很常见**——`decode()` 把模型搬到 CPU 之后，如果下一步又要跑 AR（用户自己手动分步调用时就可能发生），`_load_model` 会自动把它搬回来。**"设备漂移自动纠正"这个小设计，让手动分步调用变得安全。**

**上下文管理器支持**让代码更干净：

```python
with YuE2Pipeline.from_pretrained() as pipe:
    song = pipe(style=..., lyrics=...)
# 退出 with 时自动 close()
```

**注意 `__exit__(self, *exc)` 的 `*exc`** —— 它接收异常信息但**返回 `None`**，意味着**异常会继续向上传播**（不会被吞掉）。**这是正确的写法**：清理资源 ≠ 掩盖错误。

### 3.12 `_load_model()` —— 延迟加载

源码位置：`pipeline.py:210-223`

```python
def _load_model(self, for_nar=False):
    loading = self._model is None or next(self._model.parameters()).device != self.device
    with self._status("Loading model") if loading else nullcontext():
        if self._model is None:
            from .modeling_yue2 import YuE2ForCausalLM
            start = time.perf_counter()
            self._model = YuE2ForCausalLM.from_pretrained(self.model_dir, local_files_only=True,
                          torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
            self.load_timing["mot_load_seconds"] = time.perf_counter() - start
        if self.quantization == "fp8" and not for_nar:
            from .quantization import prepare_fp8_ar
            prepare_fp8_ar(self._model, self.device)
        self._model.to(self.device)
    return self._model
```

**这个函数体现了一条重要的性能原则：延迟加载（lazy loading）。**

构造函数里花 21 秒校验哈希，但**权重不加载**。真正的加载推迟到第一次真的要跑模型的时候。实测 `mot_load_seconds = 0.36` 秒（权重文件已在本地）。

**为什么要延迟？** 因为有些工作流**根本不需要跑模型**——比如你只想加载一个已保存的计划看看谱子。**能不做的事就不做，这是最有效的优化。**

还有几个细节：

- **`with self._status("Loading model") if loading else nullcontext():`** —— 一个非常漂亮的**三元表达式 + 上下文管理器**组合。需要显示进度时用 `_status`，不需要时用 `nullcontext()`（**空上下文，什么都不做**）。**用 `nullcontext` 而不是 `if/else` 写两遍代码，保持了单一代码路径。**
- **`torch_dtype=torch.bfloat16`** —— 主干用 BF16。**为什么是 BF16 而不是 FP16？** BF16 的指数位和 FP32 一样宽，**动态范围更大**，不容易溢出。**在长序列上这是更安全的选择。**
- **`low_cpu_mem_usage=True`** —— 加载时不先在 CPU 上凑出完整副本，**省内存**。
- **`.eval()`** —— 关掉 dropout 等训练期行为。**推理必须调这个。**
- **`local_files_only=True`** —— **模型文件已经在本地了（`from_pretrained` 阶段解析好了），不需要再联网确认一遍。** 这在离线环境下是刚需。

### 3.13 `YuE2Pipeline` 方法速查表

| 方法 | 输入 | 输出 | 一句话 |
|---|---|---|---|
| `__init__` | 目录 + 若干配置 | 实例 | 校验参数、选设备、关快不精的开关、校验权重 |
| `from_pretrained` | 仓库 ID / 目录 | 实例 | **推荐入口**，会解析下载路径、读 `pipeline.json` |
| `save_pretrained` | 空目录 | — | 导出模型 + VAE + 配置 |
| `_status` | 标签 | 上下文管理器 | 进度显示（内部） |
| `_request` | style/lyrics | `SongRequest` | 别名处理 + 参数校验（内部） |
| `_generate` | 前缀/采样/种子/阶段 | `(ids, timing, truncated)` | 真正的自回归采样（内部） |
| `plan` | 风格/歌词 | `SymbolicPlan` | **阶段一：写谱** |
| `generate_semantic` | `SymbolicPlan` | `SemanticResult` | **阶段二：写语义 token** |
| `synthesize` | `SemanticResult` | `np.ndarray [T,64]` | **阶段三：出声学 latent** |
| `decode` | `np.ndarray` | `np.ndarray [S,2]` | **阶段四：还原波形** |
| `effective_config` | 请求 + 采样覆盖 | `dict` | 记录这次实际生效的配置 |
| `__call__` | 风格/歌词 | `SongResult` | **一次跑完四个阶段** |
| `close` | — | — | 释放资源 |
| `__enter__`/`__exit__` | — | — | 支持 `with` 语法 |

---

## 第 4 章 一次真实调用发生了什么

我们用 `output/my_song/result.json` 里的**真实数据**，把整条流水线走一遍。请对照着看，这会让你对每个数字都有实感。

### 4.1 时间线

| 时刻 | 发生了什么 | 代码位置 | 实测耗时 |
|---|---|---|---|
| T0 | `from_pretrained()` 开始 | `pipeline.py:167` | — |
| | 解析模型路径 / 校验 7.2 GB 权重 sha256 | `pipeline.py:184-186`, `152-154` | **21.63 s** |
| T1 | 构造完成（模型**还没**上 GPU） | `pipeline.py:122-165` | （含在上面） |
| T2 | `pipe(...)` 被调用 | `pipeline.py:378` | — |
| | 算 `request_id` | `pipeline.py:382` | ~0 s |
| | **阶段一**：AR 生成 ABC，1333 token | `pipeline.py:266` | **12.41 s** |
| | ↳ prefill | | 0.80 s |
| | ↳ 首 token 延迟 (TTFT) | | 0.96 s |
| | ↳ 吞吐 | | 107.4 token/s |
| | **阶段二**：AR 生成语义 token，3889 token | `pipeline.py:280` | **35.76 s** |
| | ↳ prefill | | 0.36 s |
| | ↳ TTFT | | 0.38 s |
| | ↳ 吞吐 | | 108.8 token/s |
| | **阶段三**：NAR 声学生成 | `pipeline.py:300` | **24.43 s** |
| | **阶段四**：VAE 解码 | `pipeline.py:392` | **8.47 s** |
| T3 | 全部完成 | `pipeline.py:398` | **e2e = 85.67 s** |

### 4.2 从时间线上能读出什么

**（1）总时长 85.67 秒里，21.63 秒花在"校验文件"上，占 25%。**

这是**纯 I/O 的 SHA256 计算**，不产生任何音乐。**如果你要批量生成 100 首歌，这 21 分钟可以省下来**（`verify_hashes=False`）。但代价是：你不再知道权重是否被损坏过。**这是一个你自己要做的取舍，库把开关交给你，而不是替你决定。**

**（2）两个 AR 阶段的吞吐几乎一样（107.4 vs 108.8 token/s）。**

**这说明什么？** 说明瓶颈不在"算得多快"，而在"一个一个地吐 token"——**这是自回归解码的本质限制**。阶段二 token 数是阶段一的三倍（3889 vs 1333），耗时也差不多是三倍（35.76 vs 12.41）。

**（3）`execution: "cuda_graph"`，`attention: "flash"`。**

说明 `backend="torch"`（不是 `torch-eager`），CUDA Graph 捕获生效，FlashAttention 也在用。**如果你把这三个数字换成 `eager` / `sdpa`，吞吐会明显下降。** **`result.json` 把这些实现细节也记了下来，因为它们也影响结果。**

**（4）`cfg_branches: 1`。**

说明这次运行**没有启用 CFG 双分支**——因为 `cfg_scale=1.0`，而 `1.0` 时负分支被跳过了（3.6 节）。**单分支 = 一半的显存和近乎两倍的速度。** 想用 CFG 就得付这份代价。

**（5）`nar_seconds = 24.43` vs `vae_seconds = 8.47`。**

声学生成（NAR）比解码（VAE）贵了近 3 倍。**如果你的优化时间有限，请优化 NAR 阶段。**

### 4.3 内存里发生了什么

```
峰值 1：__init__ 校验权重        →  CPU 内存，逐块读 8 MB
峰值 2：_load_model              →  7.2 GB 权重在 GPU（bf16）
峰值 3：AR 生成                  →  KV cache（静态分配，上下文 24576）
峰值 4：synthesize (NAR)         →  主干仍在 GPU + NAR 的中间量
峰值 5：decode (VAE)             →  主干已搬回 CPU，VAE 独占显存
                                    解码输出直接写 CPU（output_device="cpu"）
```

**请特别注意峰值 2 到峰值 5 的转变**：**AR 阶段和 VAE 阶段几乎是"接力"使用显存的，而不是"叠加"。** 这就是 `decode()` 开头那句 `self._model.to("cpu")` 的意义。

**这也解释了 `offload_ar=True` 这个选项存在的理由**（`pipeline.py:302` 传给 `nar.synthesize`）：在 NAR 阶段把主干彻底卸到 CPU，**用速度换显存**。显存不够时开它。

---

## 第 5 章 三条上手路线

现在轮到动手了。我给你三条路线，从简单到复杂。**请按顺序做，每条都能独立跑通。**

### 5.1 路线 A：一条命令跑完（先用这个）

```bash
cd /home/luke/distributed_machine_learning/m-a-p2

conda activate yue2

python scripts/infer_yue2.py \
    --style_txt examples/style.txt \
    --lyrics_txt examples/lyrics.txt \
    --output_dir output \
    --id my_tutorial_song
```

**这个脚本内部就是第 3 章讲的那几个方法**（`scripts/infer_yue2.py:144-174`）：

```python
pipe = YuE2Pipeline.from_pretrained(...)       # 3.2
song = pipe(style=style, lyrics=lyrics, ...)   # 3.10
result = song.save_artifacts(out_dir)          # 2.3
```

**请特别注意脚本第 207 行的 `finally: pipe.close()`。** **不管成功还是失败，模型都会被释放。** 这是使用 pipeline 的标准礼仪。

**输出**：`output/my_tutorial_song/` 下会有第 2.3 节列出的 11 个文件。

### 5.2 路线 B：分步调用（理解流水线的关键一步）

```python
from yue2 import YuE2Pipeline

pipe = YuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    vae="m-a-p/YuE2-Vae",
    device="cuda",
    memory_budget_gib=24,
    backend="torch",
    progress=True,
)

STYLE = "mandarin pop, female vocal, warm acoustic guitar, soft drums"
LYRICS = "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n再唱一遍就好"

# ---- 阶段一：只看谱子（十几秒）----
plan = pipe.plan(style=STYLE, lyrics=LYRICS, cot="full", seed=831001)
print(plan.abc)              # 打印谱子
print(plan.timing)           # 耗时、token 数
plan.save("output/plan_only")   # 存盘

# ---- 阶段二：接着往下走 ----
semantic = pipe.generate_semantic(plan)
print(len(semantic.tokens))  # 语义 token 数量

# ---- 阶段三：声学 latent ----
latents = pipe.synthesize(semantic)
print(latents.shape)         # [frames, 64]

# ---- 阶段四：波形 ----
audio = pipe.decode(latents)
print(audio.shape)           # [samples, 2]

# ---- 汇总 ----
song = pipe(style=STYLE, lyrics=LYRICS, cot="full", seed=831001)
song.save_artifacts("output/manual")

pipe.close()
```

**请注意最后两行——我把完整流程又跑了一遍，只为拿到 `SongResult`。** 你可能会问："能不能用前面分步的产物拼一个 `SongResult`？"

**答案是：可以，但你得自己填八个字段，包括重新算 `config` 和 `request_id`。** 这就是 `__call__` 存在的价值。**分步调用是给你"探索"用的，`__call__` 是给你"生产"用的。**

### 5.3 路线 C：改谱子再演奏（最实用的工作流）

这是 `SymbolicPlan` 设计出来的**真正目的**。

```python
from yue2 import YuE2Pipeline
from yue2.pipeline import SymbolicPlan

pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae", progress=False)

STYLE = "mandarin pop, female vocal, warm acoustic guitar"
LYRICS = "[Verse]\n晚风轻轻吹过窗前\n[Chorus]\n再唱一遍就好"

# 1) 只规划，把谱子存下来
plan = pipe.plan(style=STYLE, lyrics=LYRICS, cot="melody", seed=831001)
plan.save("output/my_work")

# 2) 人为编辑谱子（用任何文本编辑器）
#    编辑 output/my_work/score.abc

# 3) 把改好的谱子作为【外部输入】重新走一遍
with open("output/my_work/score.abc", encoding="utf-8") as f:
    my_abc = f.read()

song = pipe(style=STYLE, lyrics=LYRICS, cot="melody", abc=my_abc, seed=831001)
song.save_artifacts("output/my_cover")
```

**请特别注意第 3 步：改过的谱子是通过 `abc=` 参数传进去的，而不是通过"改文件"传进去。**

**这正是 `SymbolicPlan.load` 那句报错信息指引我们的做法**（`pipeline.py:54`）：

> *"Saved plan changed; supply modified ABC as an external planner input"*
> （保存的计划已被修改；请把改动过的 ABC 作为**外部规划器输入**提供）

**为什么不能直接改文件然后 `load()`？** 因为 `load()` 会**严格校验所有文件的 sha256**——你改了 `score.abc`，哈希对不上，`load()` 就会拒绝。**这个"拒绝"是故意的。**

**更深一层的原因**：如果你能随便改 `plan_manifest.json` 里的哈希，就等于**任何人都能篡改一份计划而系统察觉不到**。**"计划要么原封不动，要么作为新输入重新走"这条规矩，保护的是整条流水线的可信性。**

**如果你确实想用 `load()`，那也有它的正确用法**——加载一份**未被篡改**的计划，用于复现或继续：

```python
plan = SymbolicPlan.load("output/my_work")
semantic = pipe.generate_semantic(plan)     # 前缀校验会自动通过
```

### 5.4 三个常用参数组合

| 需求 | 参数 | 说明 |
|---|---|---|
| **默认：完整规划** | `cot="full"` | 谱子带和弦符号，最"完整" |
| **翻唱 / 改编已有旋律** | `cot="melody"` + `abc=谱子` | 谱子里**不要带和弦符号** |
| **最快出活** | `cot="off"` | 不写谱，直接出声；**但不能同时给 `abc`** |

**`cot` 的选择会显著影响耗时和音色。** 实测数据（`cot="full"`）：ABC 阶段 12.41 秒 + 语义阶段 35.76 秒。`cot="off"` 时**省掉整个 ABC 阶段**（但语义阶段的 token 数会不同，因为 prefix 不同）。

> **教学提醒：`cot="off"` 时 `request.guidance` 的默认值是 1.01 而不是 1.0**（`protocol.py:106`）。**这个 0.01 不是笔误**，是官方预设的一部分。**如果你手动传 `cfg_scale`，请想清楚你在覆盖什么。**

---

## 第 6 章 常见报错与排查手册

我把实际会遇到的错误按"你看到什么 → 什么意思 → 怎么办"整理成表。

| 报错 | 触发位置 | 原因 | 怎么办 |
|---|---|---|---|
| `Memory budget must leave room for a 2GiB reserve` | `pipeline.py:164` | `memory_budget_gib ≤ 2` | 调大到 3 以上；注意实际可用 = `min(预算-2, 显卡容量-2)` |
| `The unquantized preset requires CUDA BF16 support` | `pipeline.py:160` | 显卡不支持 BF16 | 换卡，或用 `quantization="fp8"`（但需自行验证质量） |
| `backend must be torch, torch-eager, or vllm` | `pipeline.py:129` | 后端名拼错 | 检查 `--backend` |
| `progress must be True or False` | `pipeline.py:126` | 传了非布尔值 | 用 `True`/`False` |
| `style and tags are aliases and cannot disagree` | `pipeline.py:227` | 同时传了不同的 `style` 和 `tags` | 只传其中一个 |
| `Provide style and lyrics` | `pipeline.py:230` | 缺参数 | 两个都要给 |
| `Pass the SymbolicPlan returned by pipe.plan()` | `pipeline.py:273` | 类型不对 | 别自己造 `SymbolicPlan` |
| `Pass the SemanticResult returned by generate_semantic()` | `pipeline.py:291` | 类型不对 | 同上 |
| `Plan prefix disagrees with request/exact ABC IDs` | `pipeline.py:277` | **计划与请求/ABC 不自洽** | 常见于手改过 `plan.json`；请通过 `abc=` 重新传入 |
| `Semantic result does not retain the request's exact prefix` | `pipeline.py:293` | **中间环节改坏了前缀** | 检查你在两阶段之间做了什么 |
| `Saved plan changed; supply modified ABC as an external planner input` | `pipeline.py:54` | 存盘后被改过 | 见 5.3 节，改用 `abc=` 传入 |
| `Incomplete saved plan` / `Invalid plan artifact` / `Saved plan token array mismatch` | `pipeline.py:49/51/59` | 计划文件缺失/被篡改/数组与 JSON 不符 | 目录完整性被破坏，重新生成 |
| `Prefix + requested generation budget exceeds 24576` | `sampling.py:63` | 前缀 + `max_tokens` 超上下文 | 缩短歌词/谱子，或调小 `max_tokens` |
| `CFG requires a negative prefix` | `sampling.py:65` | `cfg_scale≠1` 但没有负前缀 | 检查 `cot` 模式与 `guidance` |
| `Expected latents [T,64] or [1,64,T]` | `pipeline.py:338` | 传错了 latent 形状 | 检查 `synthesize()` 的输出 |
| `VAE produced non-finite audio` | `pipeline.py:351` | 解码出了 NaN/Inf | 通常是显存不足或权重损坏；尝试调小 `vae_core_frames` 或 `memory_budget_gib` |
| `Use .flac or .wav; MP3 is an optional delivery conversion` | `pipeline.py:99` | `save()` 用了别的扩展名 | 用 `.flac`（正式产物）或 `.wav` |
| `save_pretrained needs an empty pipeline destination` | `pipeline.py:201` | 目标目录非空 | 换空目录 |
| `Cancelled before VAE` / `Cancelled during abc` | `pipeline.py:390` 等 | 你自己传的 `cancelled` 回调返回了 `True` | 正常的中断行为，不是 bug |
| `ValueError: Save operation failed` 附近 / `RuntimeError: CUDA out of memory` | — | 显存不足 | 调小 `memory_budget_gib`、开 `offload_ar`、或 `quantization="fp8"` |

### 排查思路：三问法

遇到 pipeline 里的报错，按这个顺序问三句：

**第一问：这是"输入不对"还是"环境不对"？**
- 前者是**你的调用写错了**（参数、类型、形状）。
- 后者是**机器/权重/显存的问题**。
- 上面表格里，前半部分基本都是前者，后半部分基本都是后者。

**第二问：报错发生在哪个阶段？**
对照 4.1 的时间线：
- 在 21 秒左右 → **加载/校验阶段**（哈希、路径、显存预算）
- 在 30 秒左右 → **ABC 阶段**（前缀、上下文长度、采样参数）
- 在 60 秒左右 → **语义阶段**
- 在 80 秒左右 → **NAR / VAE 阶段**（显存、形状、量化）

**第三问：这次的 `config.json` 和 `request.json` 写出来了吗？**
如果写出来了，把它们和上次能跑成功的那次对比。**这是最快的定位方法。**

---

## 第 7 章 自测题与参考答案

先自己作答，再看答案。

### 第 1 组：读代码

**Q1.** `pipe.plan(style="pop", lyrics="...")` 和 `pipe.plan(request=SongRequest(style="pop", lyrics="..."))` 有什么本质区别？

**Q2.** `plan()` 内部调用了两次 `token_prefixes()`，两次的参数不同。分别是什么？为什么要调两次？

**Q3.** 为什么 `generate_semantic()` 和 `synthesize()` 里**各做了一次**前缀校验？只做一次不够吗？

**Q4.** `SongResult.save()` 只允许 `.flac` / `.wav`，为什么不允许 `.mp3`？

**Q5.** `decode()` 为什么要先把模型 `.to("cpu")`？

**Q6.** `runtime_sha256` 是干什么的？如果它不存在，会损失什么？

### 第 2 组：写代码

**Q7.** 写一段代码：只生成乐谱并存盘，**不加载模型**（不花那 21 秒）。

**Q8.** 写一段代码：加载一份已存盘的计划并继续渲染。

**Q9.** `effective_config()` 里，`"overrides": {}` 是什么意思？什么情况下它不为空？

**Q10.** `cot="off"` 时，`plan()` 返回的 `SymbolicPlan` 里，`abc` 和 `abc_ids` 分别是什么？为什么不是 `None` / `[]` 就完事？

### 第 3 组：理解与判断

**Q11.** 有人报告："我用同样的 seed，跑出了不一样的歌。" 请列出至少三个可能的原因（按可能性排序）。

**Q12.** `result.json` 里 `"validation_status": "unvalidated"` 是什么意思？它是在说这次生成的音频有问题吗？

**Q13.** 如果你想要"先生成 10 首不同的歌（seed 不同），但只让其中一个跑完整流程"，应该怎么组织代码？

---

### 参考答案

**A1.** 实质上没有区别。`plan()` 第一行是 `request = request or self._request(...)`（`pipeline.py:257`），传了 `request` 就直接用，没传才自己造。**区别在于：传 `request` 时，你可以提前构造好并校验 `SongRequest`（比如 `cot`/`seed`/`id` 的合法性在更早的地方就报错了）。**

**A2.** 第一次：`token_prefixes(request, self.tokenizer)`——**不带** `abc_ids`，前缀只到 `[ABC_START]`，因为接下来要**生成**谱子。
第二次：`token_prefixes(request, self.tokenizer, ids)`——**带上**刚生成的 `ids`，前缀补成 `[ABC_START] + ids + [ABC_END, MUSIC_START]`，因为接下来要在**已知谱子**的前提下生成音乐。
**必须调两次，因为"规划时"和"执行时"模型看到的前缀天然不同。**

**A3.** 两次校验守护的是**两道不同的门**：`generate_semantic` 的校验守入口（确认传入的计划自洽）；`synthesize` 的校验守出口（确认已经生成的语义结果所携带的计划仍然自洽）。**中间可能发生任何事**——有人改了 `plan.prefix`、代码做了反序列化、人工构造了 `SemanticResult`。**"我刚检查过"这种假设，在跨阶段的流水线里不成立。**

**A4.** 因为**MP3 是有损压缩，属于"交付格式转换"，不是"生成结果本身"**。官方产物是 FLAC（PCM_24 无损）。如果 `save()` 允许写 mp3，那么"这次生成的到底是什么"就有歧义了——mp3 采样率可能变化、比特率可能不同。**本项目脚本额外导出的 `audio.mp3` 是明确的"便利副本"，且失败不影响主结果。**

**A5.** 因为**显存有限且共享**。7 GB 的主模型占着显存，VAE 解码就没空间了（尤其长音频）。策略是"**用完就还**"：先把模型搬回 CPU 挂着（不删除，下次还能搬回来），再 `empty_cache()` 把空闲缓存还给驱动。**注意顺序不能反——必须先搬走再清缓存。**

**A6.** 它是 `yue2/` 包下所有 `.py` 文件的哈希指纹，用来回答："**三个月后我用的是哪版代码？**" 如果没有它，你只能知道用了什么权重、什么配置，**却不知道当时跑的代码是什么版本**。而代码是会变的——同一个权重配不同代码，结果可能不同。**没有它，可复现性就缺了一块。**

**A7.**
```python
from yue2 import YuE2Pipeline
from yue2.pipeline import SymbolicPlan

# 只用 SymbolicPlan.load，不需要 pipeline 实例！
plan = SymbolicPlan.load("output/my_work")
print(plan.abc)
```
**注意：`SymbolicPlan.load` 是类方法，不需要 `pipe` 实例。** 它只读文件、校验哈希、重建计划对象。**但如果要继续生成音频，那就还是需要 `pipe`**（需要 tokenizer 和模型）。

**A8.**
```python
from yue2 import YuE2Pipeline
from yue2.pipeline import SymbolicPlan

pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae")

plan = SymbolicPlan.load("output/my_work")     # 完整性校验在这里
semantic = pipe.generate_semantic(plan)        # 前缀校验在这里
latents  = pipe.synthesize(semantic)           # 前缀校验又在这里
audio    = pipe.decode(latents)

pipe.close()
```
**请注意：`load()` 校验了文件哈希，`generate_semantic` 和 `synthesize` 又各自校验了前缀。** 三道关卡，**一道都不会少**。

**A9.** `overrides` 记录的是"**相对默认值改了什么**"。为空表示**这次运行用的是纯默认配置，什么都没改**。
不为空的典型场景：改了 `abc_sampling={"temperature": 0.9}`、改了 `semantic_sampling`、显式传了 `cfg_scale`（默认值 1.0/1.01 不算改动）。
**空字典本身也是信息。**

**A10.** `abc` 是 `None`，`abc_ids` 是 `[]`。
**但 `prefix` 必须完整正确**——因为 `cot="off"` 时，前缀是 `[EOD] + encode(text) + [ABC_START, ABC_END, MUSIC_START]`，**注意这里 ABC 段是"空"的，但标记符仍然要在**（`protocol.py:117-118`）。
**如果返回一个"什么都不带"的对象，`generate_semantic()` 就得写一堆 if/else 来处理特殊情况。** 返回"**空但合法**"的对象，**下游代码完全不需要知道自己上游有没有规划**——这是"用统一接口消化差异"的经典做法。

**A11.** 按可能性排序：

1. **设备/后端不同**：换了显卡（3090 → 4090）、`backend` 从 `torch` 换成 `torch-eager`（CUDA Graph 捕获与否会影响数值）、`device` 从 `cuda` 换成 `cpu`。**`config.json` 里能查出来。**
2. **代码版本不同**：包被升级过，**`runtime_sha256` 会变**。`sampling.py` 里那些"保留历史数值行为"的开关（`legacy_off`、BF16 CFG 运算顺序）一改，输出就变。
3. **权重不同**：重新下载的模型权重不一样（`weights` 字段的 sha256 会变），或者 VAE 换了版本（`decoder_release` 从 `standard` 变成 `legacy`）。
4. **请求内容不同**：哪怕只是一个空格不同——`tokenizer.encode` 会先做 NFC 规范化，**词表边界可能因此移动**，后面的 token 序列就全变了。
5. **采样参数不同**：温度、top_p、top_k 任一变化都会改变采样路径。**注意 `overrides` 字段专门用来查这个。**

**排查顺序建议**：先 diff 两次的 `config.json`，再 diff `request.json`，最后看 `weights`。**这三个文件能覆盖上面 5 条中的前 5 条。**

**A12.** **不是在说音频有问题。** 它的意思是：**"我把我用的所有配置都记下来了，但我没有承诺这个配置组合被官方验证过。"**
具体来说，官方发布时验证过的是**默认配置**。如果你改了 `cfg_scale`、改了采样温度、换了 `decoder_release`——**这个组合就超出了官方验证范围**。**记录完备 ≠ 结果可靠**，这两件事必须分开说。
**这是一个负责任的库应有的自我标注：它告诉你边界在哪里，让你自己判断要不要跨过去。**

**A13.**
```python
import copy
from yue2 import YuE2Pipeline

pipe = YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae")

plans = []
for i in range(10):
    req = SongRequest(style=STYLE, lyrics=LYRICS, seed=1000 + i)   # 不同 seed
    plans.append(pipe.plan(request=req))

# 从中挑一个
chosen = plans[3]
semantic = pipe.generate_semantic(chosen)
latents  = pipe.synthesize(semantic)
audio    = pipe.decode(latents)

pipe.close()
```
**关键点：`plan()` 阶段是快的（12 秒左右），后三个阶段是贵的（68 秒左右）。** 所以"多生成几个谱子备选，挑一个再渲染"是划算的。
**更省的做法**：先用 `plan_only` 跑 10 次，把 10 张谱子拿给人看，选定后再跑完整流程——**这也正是 `plan()` / `generate_semantic()` 被拆成两个方法的原因。**

---

## 附录 A：本文涉及的外部函数速查

这些函数定义在别的模块，但 `pipeline.py` 大量调用。**理解它们的输入输出，是读懂 pipeline 的前提。**

| 函数 | 所在模块 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| `token_prefixes(request, tokenizer, abc_ids=None)` | `protocol.py:115` | 请求 + tokenizer + 可选 ABC ids | `list[int]` | 拼出 AR 前缀；**`abc_ids` 在不在，语义完全不同** |
| `negative_prefix(request, tokenizer, abc_ids=None)` | `protocol.py:129` | 同上 | `list[int]` | CFG 的负分支前缀 |
| `resolve_sampling(value, default)` | `protocol.py:70` | `None` / `Sampling` / `dict` | `Sampling` | 采样配置的三态合并 |
| `resolve_model(model, revision, ...)` | `storage.py:43` | 仓库 ID 或目录 | `Path` | 是目录就返回，否则下载 |
| `model_identity(path, verify=True)` | `storage.py:105` | 模型目录 | `dict` | 各 `.safetensors` 的 sha256 + 字节数 |
| `identity(value)` | `storage.py:19` | 任意 JSON-able | 64 位 hex | 规范化 JSON 后算 sha256 |
| `write_json(path, value)` | `storage.py:24` | 路径 + 对象 | — | **原子写**（先写 `.tmp` 再 `os.replace`） |
| `collect_hashes(directory, exclude=("result.json",))` | `storage.py:125` | 目录 | `dict` | 目录下所有文件的 sha256 + 大小 |
| `sha256_file(path)` | `storage.py:11` | 文件路径 | 64 位 hex | 分块读，不占内存 |
| `copy_model_files(source, destination)` | `storage.py:57` | 源目录 + 空目标目录 | — | 按白名单导出；**代码只取官方包那份** |
| `generate_tokens(model, prefix, sampling, seed, phase, ...)` | `sampling.py:57` | 模型 + 前缀 + 采样 + 种子 + 阶段 | `(ids, timing, truncated)` | 真正的自回归采样循环 |
| `synchronize(device)` | `sampling.py:8` | 设备 | — | 同步 CUDA / MPS |
| `Progress(enabled=...)` | `progress.py` | 开关 | 上下文管理器 | 进度显示；**走 stderr，不污染 stdout** |
| `synthesize(model, prefix, codec, seed, ...)` | `nar.py:228` | 模型 + 前缀 + codec + 种子 | `[frames, 64]` CPU float32 | 声学生成（flow matching） |
| `YuE2VAE.from_pretrained(..., decoder_only=True)` | `modeling_vae.py` | 目录 | VAE 模型 | 只加载解码器 |
| `YuE2ForCausalLM.from_pretrained(...)` | `modeling_yue2.py` | 目录 | 主干模型 | 加载 AR + NAR 主干 |
| `prepare_fp8_ar(model, device)` / `restore_ar(model)` | `quantization.py` | 模型 | — | FP8 量化 / 还原 |

---

## 附录 B：一条时间线的复述练习

请合上文档，用自己的话把这张表讲出来（**不看原文**）：

```
from_pretrained()
  └─ 解析路径 → 校验 7.2GB 权重哈希（21.6s）→ 构造实例（模型还没上 GPU）
plan()
  └─ 前缀①（无ABC）→ AR 生成 ABC（12.4s）→ 前缀②（含ABC）
generate_semantic()
  └─ 校验前缀② → 必要时算负分支 → AR 生成语义 token（35.8s）→ 减去 CODEC_OFFSET
synthesize()
  └─ 校验前缀 → FP8 则还原权重 → NAR 声学生成（24.4s）→ [frames, 64]
decode()
  └─ 模型搬回 CPU 腾显存 → VAE 分块解码（8.5s）→ 裁 halo → 检查 NaN → clamp → [samples, 2]
__call__ 组装
  └─ config = effective_config(...) → identity = sha256(请求+配置+权重) → SongResult
save_artifacts()
  └─ 11 个文件 + 每个文件的 sha256
close()
  └─ 释放模型、清 CUDA 缓存
```

**能顺下来，你就把 `pipeline.py` 吃透了。**

---

## 写在最后

`pipeline.py` 只有 399 行，比 `modeling_yue2.py`（33 KB）和 `modeling_vae.py`（25 KB）都短。

**但它是整个项目里"信息密度"最高的一个文件。** 因为它不实现任何数学，它实现的是**判断**：什么时候该报错、什么时候该搬运、什么时候该记录、什么时候该放手。

**给你三条带走的准则：**

1. **在阶段边界上校验数据，不要相信"我刚检查过"。** ——`pipeline.py:275-277` 和 `pipeline.py:292-293` 是同一个思想的两处体现。
2. **展示层和计算层严格隔离。** ——进度条不进身份哈希，日志不进结果文件。
3. **诚实地记录，包括记录你的不确定性。** ——`"validation_status": "unvalidated"` 这一行，比十页文档更能建立信任。

**祝你写得开心。跑出第一首歌的那一刻，请回头看看第 4 章那张时间线——你会知道你刚才那 85 秒里，每一秒都在干什么。**
