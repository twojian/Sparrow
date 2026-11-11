# t2a 复现入口（合成数据链路验证）

这是 t2a 这一条链路的复现入口：录音先由冻结的 Mimi 压成连续潜变量，再照流匹配从噪声逐步生成，最后由 Paraformer 听写判分。数据、参数与下面所有结论都基于合成数据，用于验证"数据 → 编码 → 训练 → 评测 → 推理"是否闭合，不代表模型能力。

## 环境

- **Python 环境**：Python 3.12 虚拟环境，装好 [requirements.txt](requirements.txt)（在 PyTorch 2.9 环境下验证）。
- **额外依赖**：`asr.py` 判分还需要 `funasr`、`pypinyin`、`librosa`、`torchaudio`、`modelscope`；其中 `torch`、`torchaudio`、`modelscope` 未写进 `requirements.txt`，由环境自带。
- **模型权重**：冻结的 FLUX.2 图像 VAE、Mimi、Whisper、Paraformer-zh 首次运行会自动下载，无需手动准备。
- **目录约定**：命令都在本目录（`code\t2a`）下执行，产物写到 `..\..\repro\work\`。

## 音频数据契约

训练与评测只读一份 `voice.npz`，共 5 个键：

| 键 | 形状 / 类型 | 含义 |
| --- | --- | --- |
| `wav` | `[432, 49920]` `float32` | 24 kHz × 2.08 s = 49920 采样点，去首尾静音后只补尾部静音 |
| `animal` | `[432]` `int64` | 词表 `ANIMALS` 的序号（猫、狗、兔、鸟、鱼、熊） |
| `voice` | `[432]` `str` | 音色名，共 12 个：9 个训练音色、3 个测试音色，按说话人划分 |
| `test` | `[432]` `bool` | `True` 为测试集，108 条；训练集 324 条 |
| `name` | `[432]` `str` | 原始文件名 |

该文件不入库，需自备；也可以用 `data.py voice` 从 wav 目录重新整理。

## 四步

四步按依赖顺序执行：先造数据，再训判卷器，然后编码缓存，最后训主模型。

**1) 建目录、造数据**：

```powershell
# 产物目录不会自动建父目录，先建好
New-Item -ItemType Directory -Force -Path ..\..\repro\work\data, ..\..\repro\work\cache, ..\..\repro\work\judges, ..\..\repro\work\runs, ..\..\repro\work\gen_test | Out-Null

# 卡通小世界：训练 / 测试用不同种子（t2a 只用其中标签与测试集做评测）
python data.py world ..\..\repro\work\data\world_test.npz   2000  500 900000000
python data.py world ..\..\repro\work\data\world_train.npz 60000 10000 0

# 把自备的 voice.npz 放进数据目录
Copy-Item data\voice.npz ..\..\repro\work\data\voice.npz
```

**2) 训练判卷器**：

```powershell
python train_judges.py img --data ..\..\repro\work\data --out ..\..\repro\work\judges --steps 2000
python train_judges.py vid --data ..\..\repro\work\data --out ..\..\repro\work\judges --steps 1500
```

`encode.py` 与 `sparrow.py` 的评测段都会无条件加载 `judge_img.pt` 与 `judge_vid.pt`，即使只编码 `aud`、只练 `t2a` 也要先备好这两个。

**3) 编码缓存**（只编 `aud`，即 Mimi 潜变量）：

```powershell
python encode.py --data ..\..\repro\work\data --judges ..\..\repro\work\judges --out ..\..\repro\work\cache --parts aud
```

**4) 训练主模型**：

```powershell
python sparrow.py `
  --data ..\..\repro\work\data --cache ..\..\repro\work\cache --judges ..\..\repro\work\judges `
  --out ..\..\repro\work\runs\t2a --tasks t2a --eval_tasks t2a --steps 8000 --eval_every 1000
```

产物为 `final.pt`、`report.json`、`samples/`；每 1000 步另有中途评测。训练录音按固定随机数做四份时间平移增强，只挪入尾部静音，不截字。要接着一份已有权重继续训，加 `--init <final.pt>`；要同时复习画图，把 `--tasks` 改为 `t2i,t2a` 并用 `--bs_task t2i=<每轮题数>`。

## 推理

```powershell
python infer.py ..\..\repro\work\runs\t2a\final.pt t2a 小猫 ..\..\repro\work\gen_test\cat.wav
```

默认生成 30 步、引导强度 3、噪声种子 0；可加 `--steps`、`--cfg`、`--seed` 覆盖，或用 `--noise` 指定起点噪声。六种名字为小猫、小狗、小兔、小鸟、小鱼、小熊。生成时会在输出旁另存 `<输出>.noise.pt`。

## 实测（合成数据）

| 项 | 数值 |
| --- | --- |
| 参数规模 | 16.04M（`dim=384`、8 层、6 头） |
| t2a 序列长度 | 107（`<bos>` + `<aud>` + 104 个潜变量向量） |
| 训练耗时 | 8000 步 12895.3 s（1.61 s/步）；续训 2000 步 3223.1 s（1.61 s/步） |
| 训练峰值显存 | 8000 步 2.56 GB；续训 2000 步 2.62 GB |
| 念名字正确率 | 8000 步 `sound_acc` 91.67% / `exact_acc` 90.00%；续训 2000 步 94.58% / 92.92%（240 题，`eval_seed 0`） |
| 评测耗时（240 题） | 首次 `--eval_only` 353.3 s / 4.78 GB（评的是 `weights\` 里那份参考实现权重，不是本地训练产物）；8000 步末评 132.2 s / 4.73 GB；续训末评 135.7 s / 4.79 GB |
| 单条推理耗时 | 未记录：只做过出声冒烟，没有单独计时 |

判分口径：念名字由 Paraformer-zh 听写，转写拼音（不计声调）与目标词相同即算对，另报逐字全对比例。

## 限制

- **封闭词表**：只认 6 种动物名，不是自由语音指令。
- **定长 2.08 秒**：去静音后装不下就丢弃，不截字。
- **单一音色来源**：12 个音色全部来自合成录音，说话人覆盖有限。
- **只训了声音生成**：`sparrow.py` 里还有 t2i / t2v 与看图、听声、看视频等路径，t2v 与理解那几条路径还没训、也没评测。

## 数据、许可与署名

代码采用 [Apache-2.0](LICENSE)。卡通小世界由 [world/world.py](world/world.py) 生成。`data/voice.npz` 是 432 条合成录音，测试说话人未用于训练。

- **Mimi — Kyutai**：[模型与来源](https://huggingface.co/kyutai/mimi)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。用作冻结的声音编解码器；取跳过码本查表的连续潜变量，再由 Mimi 解码。本目录未重新分发 Mimi 参数。
- **FLEURS — Google**：[普通话数据来源](https://huggingface.co/datasets/google/fleurs)，[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)。使用其中真人朗读作为音色参考，由 CosyVoice 合成动物名；随后去静音、补至 2.08 秒并做识别把关。本目录分发合成后的录音，不含原始朗读。
- **CosyVoice2-0.5B、CosyVoice-300M-SFT — FunAudioLLM / 通义**：[项目](https://github.com/FunAudioLLM/CosyVoice)，[CosyVoice2-0.5B 模型卡](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)、[CosyVoice-300M-SFT 模型卡](https://huggingface.co/FunAudioLLM/CosyVoice-300M-SFT)均标注 Apache-2.0。用于造录音与候选音色；保留的 12 个音色来自 FLEURS 参考音。
- **Whisper-large-v3-turbo — OpenAI**：[来源](https://huggingface.co/openai/whisper-large-v3-turbo)，MIT；合成录音时当筛子。
- **Paraformer-zh — FunASR**：[来源](https://huggingface.co/funasr/paraformer-zh)，Apache-2.0；用于最终把关和念名字判卷。
- **FLUX.2 图像 VAE — Black Forest Labs**：[来源](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/main/vae)，Apache-2.0；用于图像编码与还原。
