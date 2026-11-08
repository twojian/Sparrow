"""A1 干净复测:bs256 + bf16 + 200 步 + 温度采样(不评测,acc 沿用之前的 500 步结果)。
用途:给 A1 补一份带温度/时钟记录的时序数据,消除「500 步平均值是否被热漂移污染」的疑问。
"""
import json
import subprocess
import threading
import time

PY = r"<venv>\Scripts\python.exe"
ROOT = r"<repo>"
TMP = ROOT + r"\repro\work\tmp"
CURVE = TMP + r"\a1_thermal.csv"
RES = TMP + r"\a1_result.json"

CMD = [PY, "-u", r"<code_dir>\t2i\code\sparrow.py",
       "--data", ROOT + r"\repro\work\data", "--cache", ROOT + r"\repro\work\cache",
       "--judges", ROOT + r"\repro\work\judges", "--out", ROOT + r"\repro\work\runs\abl_a1_clean",
       "--tasks", "t2i", "--steps", "200", "--bs", "256", "--eval_every", "0"]

stop = False
st = {"peak": 0, "masks": set()}


def snap():
    q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,"
                        "clocks_throttle_reasons.active", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip()
    p = [x.strip() for x in q.split(",")]
    return int(p[0]), int(p[1]), float(p[2]), p[3]


def sampler():
    with open(CURVE, "w", encoding="utf-8") as f:
        f.write("ts_epoch,temp,sm_mhz,power_w,throttle_mask\n")
        while not stop:
            try:
                t, c, pw, m = snap()
                f.write(f"{time.time():.0f},{t},{c},{pw},{m}\n")
                f.flush()
                st["peak"] = max(st["peak"], t)
                st["masks"].add(m)
            except Exception:
                pass
            time.sleep(5)


threading.Thread(target=sampler, daemon=True).start()
while snap()[0] > 65:
    time.sleep(10)
time.sleep(180)
t0 = time.time()
start_temp = snap()[0]
proc = subprocess.Popen(CMD, stdout=open(TMP + r"\a1_train.log", "w", encoding="utf-8"),
                        stderr=subprocess.STDOUT, cwd=r"<code_dir>\t2i\code")
proc.wait()
stop = True
end_temp = snap()[0]
json.dump({"cmd": " ".join(CMD), "start_temp": start_temp, "end_temp": end_temp,
           "peak_temp": st["peak"], "masks_seen": sorted(st["masks"]),
           "wall_seconds": round(time.time() - t0, 1)}, open(RES, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(json.load(open(RES, encoding="utf-8")), ensure_ascii=False), flush=True)
