"""B 轴热态对照探针:同一份训练代码路径(train()),串行跑 bf16 → fp16 → bf16(做温度漂移的夹逼),
只跑 100 步、不评测,比较单步耗时与训练峰值。"""
import argparse
import json
import sys

import torch

sys.path.insert(0, r"<code_dir>\t2i\code")
import sparrow as S

CACHE = r"<repo>\repro\work\cache"
data = S.Cache(CACHE, {"img"})
lays = {t: S.Layout(t) for t in S.TASKS}


def run(precision, steps=100, bs=128, accum=1):
    torch.manual_seed(0)
    model = S.Sparrow(384, 8, 6).to("cuda")
    a = argparse.Namespace(tasks=["t2i"], eval_tasks=["t2i"], bs=bs, bs_vid=128, lr=6e-4, ema=0.999,
                           cfg=3.0, steps=steps, eval_every=0, grad_accum=accum, precision=precision,
                           weights={"t2a": 4.0}, init=None, data="", cache="", judges="", out="", eval_only=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sec, log = S.train(a, model, data, lays)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    return {"precision": precision, "steps": steps, "train_seconds": sec,
            "per_step": round(sec / steps, 3), "peak_GB": peak, "log": log}


for p in ("bf16", "fp16", "bf16"):
    print(json.dumps(run(p), ensure_ascii=False), flush=True)
