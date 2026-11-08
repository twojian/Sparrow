"""A3 数值等效性探针:同一个 512 行样本,「一次 bs512 反向」vs「4 次 bs128 累积(各损失/4)」,
比较逐样本 loss 与最终梯度。只做一次前向/反向,不训练、不写产物。"""
import json
import sys

import torch

sys.path.insert(0, r"<code_dir>\t2i\code")
import sparrow as S

dev = "cuda"
CACHE = r"<repo>\repro\work\cache"
lay = S.Layout("t2i")
data = S.Cache(CACHE, {"img"})

torch.manual_seed(0)
model = S.Sparrow(384, 8, 6).to(dev)
B = 512

slots, segs = data.get("t2i", B)                     # 取同一批 512 行,两条路径共用
segs = [s.float() for s in segs]
g = torch.Generator(device=dev).manual_seed(1234)     # 两条路径共用同一组 t / 噪声
t = torch.sigmoid(torch.randn(B, device=dev, generator=g))
drop = (torch.rand(B, device=dev, generator=g) < 0.1) if lay.slots else None
tok = S.fill_tok(lay, slots, B, drop)
x1 = segs[-1]
eps = torch.randn(x1.shape, device=dev, generator=g)
noisy = (1 - t[:, None, None]) * eps + t[:, None, None] * x1


def run(chunk):
    """chunk=None: 一次 512;chunk=128: 4 次累积。返回逐样本 loss 列表与梯度。"""
    for p in model.parameters():
        p.grad = None
    losses = []
    if chunk is None:
        parts = [(slice(0, B), 1)]
    else:
        parts = [(slice(s, s + chunk), B // chunk) for s in range(0, B, chunk)]
    for sl, div in parts:
        with torch.autocast("cuda", torch.bfloat16):
            _, v = model(lay, tok[sl], [x[sl] for x in segs[:-1]] + [noisy[sl]], t[sl])
        l = torch.nn.functional.mse_loss(v.float(), (x1 - eps)[sl])
        (l / div).backward()
        losses.append(l.item())
    grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
    return losses, grads


l_big, g_big = run(None)
l_micro, g_acc = run(128)
assert len(g_big) == len(g_acc)

num = sum(((a - b) ** 2).sum().item() for a, b in zip(g_big, g_acc)) ** 0.5
den = sum((a ** 2).sum().item() for a in g_big) ** 0.5
maxabs = max((a - b).abs().max().item() for a, b in zip(g_big, g_acc))
print(json.dumps({
    "loss_one_pass_512": round(l_big[0], 6),
    "loss_mean_of_four_128": round(sum(l_micro) / len(l_micro), 6),
    "loss_per_micro": [round(x, 6) for x in l_micro],
    "grad_rel_l2_diff": f"{num / den:.3e}",
    "grad_max_abs_diff": f"{maxabs:.3e}",
    "grad_norm_512": round(den, 4),
}, ensure_ascii=False, indent=1))
