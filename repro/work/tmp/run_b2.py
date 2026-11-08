"""B2 凉机重跑:等温度 <=65C 且空闲 >=180s 后,用原命令跑 500 步 fp16(含末评)。
全程 5s 采样;若温度连续 3 次 >=88C 则终止进程并标记「熔断」,不烧卡。
"""
import json
import subprocess
import sys
import threading
import time

PY = r"<venv>\Scripts\python.exe"
ROOT = r"<repo>"
TMP = ROOT + r"\repro\work\tmp"
OUT_CURVE = TMP + r"\b2_thermal.csv"
OUT_JSON = TMP + r"\b2_result.json"

CMD = [PY, "-u", r"<code_dir>\t2i\code\sparrow.py",
       "--data", ROOT + r"\repro\work\data",
       "--cache", ROOT + r"\repro\work\cache",
       "--judges", ROOT + r"\repro\work\judges",
       "--out", ROOT + r"\repro\work\runs\abl_b2_cool",
       "--tasks", "t2i", "--steps", "500", "--bs", "128", "--precision", "fp16"]

stop = False
state = {"peak": 0, "max_mask": "0x0", "hot_run": 0, "fuse": False}


def snap():
    q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,"
                        "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in q.split(",")]
    return int(p[0]), int(p[1]), p[2], p[3]


def sampler():
    with open(OUT_CURVE, "w", encoding="utf-8") as f:
        f.write("ts_epoch,temp,sm_mhz,power_w,throttle_mask\n")
        while not stop:
            try:
                t, c, pw, m = snap()
                f.write(f"{time.time():.0f},{t},{c},{pw},{m}\n")
                f.flush()
            except Exception:
                pass
            time.sleep(5)


def temp_now():
    return snap()[0]


threading.Thread(target=sampler, daemon=True).start()
while temp_now() > 65:
    time.sleep(10)
time.sleep(180)                     # 组间至少 3 分钟冷却
start_temp = temp_now()
t_start = time.time()
proc = subprocess.Popen(CMD, stdout=open(TMP + r"\b2_train.log", "w", encoding="utf-8"),
                        stderr=subprocess.STDOUT, cwd=r"<code_dir>\t2i\code")
hot = 0
while proc.poll() is None:
    time.sleep(5)
    t, c, pw, m = snap()
    state["peak"] = max(state["peak"], t)
    if m != "0x0000000000000001":
        state["max_mask"] = m
    if t >= 88:
        hot += 1
        state["hot_run"] = max(state["hot_run"], hot)
        if hot >= 3:
            proc.terminate()
            state["fuse"] = True
            break
    else:
        hot = 0
proc.wait()
stop = True
res = {"cmd": " ".join(CMD), "start_temp": start_temp, "end_temp": temp_now(),
       "peak_temp": state["peak"], "max_throttle_mask": state["max_mask"], "fuse_terminated": state["fuse"],
       "exit_code": proc.returncode, "wall_seconds": round(time.time() - t_start, 1),
       "t_start": t_start, "t_end": time.time()}
json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True)
