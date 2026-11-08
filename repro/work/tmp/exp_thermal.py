"""热控消融编排:每组先等温度降到 <=65C 再开跑,全程 5 秒一次采样 GPU 状态。

组:bt_bf16(凉,100) -> bt_fp32(凉,100) -> bt_fp16(凉,100) -> a0_heat(凉,200,看升温曲线) -> bt_bf16_hot(热机,100,冷启动对照)
每组都走真实 train() 代码路径、只跑 t2i、不评测,单步耗时可比。
"""
import argparse
import json
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, r"<code_dir>\t2i\code")
import sparrow as S

CACHE = r"<repo>\repro\work\cache"
OUT_CURVE = r"<repo>\repro\work\tmp\thermal_curve.csv"
OUT_RUNS = r"<repo>\repro\work\tmp\thermal_runs.json"

data = S.Cache(CACHE, {"img"})
lays = {t: S.Layout(t) for t in S.TASKS}

stop = False


def snap():
    """返回 (temp, sm_mhz, power_w, throttle_mask)。"""
    q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,"
                        "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    parts = [x.strip() for x in q.split(",")]
    return int(parts[0]), int(parts[1]), parts[2], parts[3]


def sampler():
    with open(OUT_CURVE, "w", encoding="utf-8") as f:
        f.write("ts_epoch,temp,sm_mhz,power_w,throttle_mask\n")
        while not stop:
            try:
                t, c, p, m = snap()
                f.write(f"{time.time():.0f},{t},{c},{p},{m}\n")
                f.flush()
            except Exception:
                pass
            time.sleep(5)


def temp_now():
    return snap()[0]


def wait_cool(target=65, cap=600):
    t0 = time.time()
    while time.time() - t0 < cap:
        if temp_now() <= target:
            return True
        time.sleep(10)
    return False


def run(precision, steps, bs=128):
    torch.manual_seed(0)
    model = S.Sparrow(384, 8, 6).to("cuda")
    a = argparse.Namespace(tasks=["t2i"], eval_tasks=["t2i"], bs=bs, bs_vid=128, lr=6e-4, ema=0.999,
                           cfg=3.0, steps=steps, eval_every=0, grad_accum=1, precision=precision,
                           weights={"t2a": 4.0}, init=None, data="", cache="", judges="", out="",
                           eval_only=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sec, log = S.train(a, model, data, lays)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model
    torch.cuda.empty_cache()
    return {"precision": precision, "steps": steps, "train_seconds": sec,
            "per_step": round(sec / steps, 3), "peak_GB": peak, "log": log}


GROUPS = [("bt_bf16", "bf16", 100, True),
          ("bt_fp32", "fp32", 100, True),
          ("bt_fp16", "fp16", 100, True),
          ("a0_heat", "bf16", 200, True),
          ("bt_bf16_hot", "bf16", 100, False)]

results = []
threading.Thread(target=sampler, daemon=True).start()
try:
    for name, prec, steps, cool in GROUPS:
        cooled = wait_cool() if cool else False
        start_temp = temp_now()
        t_start = time.time()
        r = run(prec, steps)
        t_end = time.time()
        r.update({"group": name, "start_temp": start_temp, "end_temp": temp_now(),
                  "cooled_to_65": cooled, "t_start": t_start, "t_end": t_end})
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
finally:
    stop = True
    time.sleep(0.5)
    json.dump(results, open(OUT_RUNS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("DONE", flush=True)
