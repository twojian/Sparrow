"""D 轴探针:torch.compile 在 Windows + 无张量核显卡上的可用性、channels_last 是否有收益。
只做前向/反向微基准,不训练、不写报告。"""
import json
import sys
import time
import traceback

import torch

sys.path.insert(0, r"<code_dir>\t2i\code")
import sparrow as S

dev = "cuda"
DATA = r"<repo>\repro\work\data"
CACHE = r"<repo>\repro\work\cache"
lay = S.Layout("t2i")
data = S.Cache(CACHE, {"img"})


def step(model, bs=128, dtype=torch.bfloat16):
    slots, segs = data.get("t2i", bs)
    segs = [s.float() for s in segs]
    t = torch.sigmoid(torch.randn(bs, device=dev))
    tok = S.fill_tok(lay, slots, bs, (torch.rand(bs, device=dev) < 0.1) if lay.slots else None)
    x1, eps = segs[-1], torch.randn_like(segs[-1])
    segs[-1] = (1 - t[:, None, None]) * eps + t[:, None, None] * x1
    with torch.autocast("cuda", dtype):
        _, v = model(lay, tok, segs, t)
    loss = torch.nn.functional.mse_loss(v.float(), x1 - eps)
    loss.backward()
    for p in model.parameters():
        p.grad = None
    return loss.item()


def bench(model, n=10, bs=128, dtype=torch.bfloat16):
    for _ in range(3):
        step(model, bs, dtype)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        step(model, bs, dtype)
    torch.cuda.synchronize()
    return (time.time() - t0) / n


torch.manual_seed(0)
m = S.Sparrow(384, 8, 6).to(dev)
out = {"eager_bf16_s": round(bench(m), 4)}

# --- channels_last ---
cl = S.Sparrow(384, 8, 6).to(dev).to(memory_format=torch.channels_last)
out["channels_last_bf16_s"] = round(bench(cl), 4)
out["channels_last_note"] = "主干只有 Linear/Embedding/LayerNorm/attention,无 4D 卷积"

# --- torch.compile ---
try:
    cm = torch.compile(m)
    t0 = time.time()
    step(cm)
    torch.cuda.synchronize()
    out["compile_first_step_s"] = round(time.time() - t0, 1)
    out["compile_bf16_s"] = round(bench(cm, n=10), 4)
    out["compile_ok"] = True
except Exception as e:
    out["compile_ok"] = False
    out["compile_error_type"] = type(e).__name__
    out["compile_error"] = str(e)[:800]
    out["compile_trace_tail"] = traceback.format_exc().strip().splitlines()[-1][:300]

print(json.dumps(out, ensure_ascii=False, indent=1))

# --- VAE 解码侧的 channels_last(评测路径) ---
try:
    import frozen
    vae = frozen.ImageVAE()
    lat = torch.randn(64, 32, 8, 8, device=dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        vae.decode(lat)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            vae.decode(lat)
        torch.cuda.synchronize()
        base = (time.time() - t0) / 3
    vae.m = vae.m.to(memory_format=torch.channels_last)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        vae.decode(lat)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            vae.decode(lat)
        torch.cuda.synchronize()
        clv = (time.time() - t0) / 3
    print(json.dumps({"vae_decode_base_s": round(base, 4), "vae_decode_channels_last_s": round(clv, 4)}, indent=1))
except Exception as e:
    print(json.dumps({"vae_probe_error": f"{type(e).__name__}: {e}"[:500]}, ensure_ascii=False))
