# t2i 复现入口（合成数据链路验证）

这是 t2i 这一条链路的复现入口：数据、参数与下面所有数字都基于合成数据，用于验证"数据 → 编码 → 训练 → 评测 → 推理"是否闭合，不代表模型能力。

## 环境

- **Python 环境**：自己准备一个 Python 3.12 虚拟环境并装好依赖（`pip install -r ..\..\repro\requirements-lock.txt`），下文命令都在该环境已激活的终端里执行。
- **模型权重**：冻结的编码器与 tokenizer 首次运行会自动下载，无需手动准备。
- **目录约定**：命令都在本目录（`code\t2i`）下执行，产物写到 `..\..\repro\work\`。

## 四步

四步按依赖顺序执行：先把数据造出来，再训判卷器（编码与评测都依赖它），然后编码缓存，最后训主模型。

**1) 建数据**（世界图与视频，训练 / 测试用不同种子）：

```powershell
python data.py world ..\..\repro\work\data\world_test.npz   2000  500 900000000
python data.py world ..\..\repro\work\data\world_train.npz 60000 10000 0
```

**2) 训练判卷器**：

```powershell
python train_judges.py img --data ..\..\repro\work\data --out ..\..\repro\work\judges --steps 2000
python train_judges.py vid --data ..\..\repro\work\data --out ..\..\repro\work\judges --steps 1500
```

**3) 编码缓存**（只编 img，约 242 MiB）：

```powershell
python encode.py --data ..\..\repro\work\data --judges ..\..\repro\work\judges --out ..\..\repro\work\cache --parts img
```

**4) 训练主模型**：

```powershell
python sparrow.py `
  --data ..\..\repro\work\data --cache ..\..\repro\work\cache --judges ..\..\repro\work\judges `
  --out ..\..\repro\work\runs\ep02 --tasks t2i --steps 4000 --eval_every 250 --bs 128
```

## 出图 / 推理

```powershell
python infer.py ..\..\repro\work\runs\ep02\final.pt t2i "红色 小猫" ..\..\repro\work\gen_test\red_cat.png
```

可加 `--steps`（默认 30）、`--cfg`（默认 3.0）、`--seed`（默认 0）；同一提示 + 同一种子结果可复现。

## 实测（合成数据）

| 项 | 数值 |
| --- | --- |
| 参数规模 | 16.04M（`dim=384`、8 层、6 头） |
| 批大小 / 步数 | 128 / 4000，每 250 步评测 384 题 |
| 纯训练耗时 | 3119.2 s（约 52 分钟） |
| 训练峰值显存 | 4.60 GB（超出 4 GB 级单卡预算，由驱动回退承载，单步约 0.78 s） |
| 末评 `both_acc` | 100%（颜色与动物同时正确） |
| `nn_dist_gen_mean` | 2917.362 |
| 单图推理 | 约 2.8 s / 0.45 GB |

## 限制

- **封闭词表**：输入只认 8 种颜色 × 6 种动物 = 48 种组合，外加固定动作词与结构符号，不是自由句子。
- **分辨率 64×64**：细节有限，换构图、换画风都不行。
- **容量小**：16M 参数只够学会这套封闭组合与固定模板。
- **显存紧**：4 GB 级单卡预算下批大小 128 已接近上限，单步约 0.78 s；再往上加批大小会先撞显存。
- **精度**：bf16 在无张量核的硬件上无加速，混合精度只省显存、不提速。
- **只训了这一条链路**：t2a / t2v 与其他理解任务的代码路径在 `sparrow.py` 里，但未训练、未评测。

更完整的设计取舍见 `..\..\repro\t2i-实现与进度.md`，分步复现文档见 `..\..\repro\docs\`。
