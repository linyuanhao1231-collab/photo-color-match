# /// script
# dependencies = ["numpy", "opencv-python-headless", "scipy"]
# ///
import sys, io, contextlib, numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import os
os.chdir(__import__("pathlib").Path(__file__).resolve().parent.parent)  # 讓 "sample/..." 相對路徑一律以 repo 根目錄為準
from colormatch import imread, fit_lut, apply_lut

def de(a,b):
    la=cv2.cvtColor(a.astype(np.float32),cv2.COLOR_RGB2Lab); lb=cv2.cvtColor(b.astype(np.float32),cv2.COLOR_RGB2Lab)
    return np.sqrt(((la-lb)**2).sum(-1))

P = lambda k: (imread(f"sample/{k}.jpg"), imread(f"sample/{k}_truth.jpg"))
pairs = {"人像": (imread("sample/before.jpg"), imread("sample/after.jpg")),
         "風景": P("t1_landscape"), "街景": P("t2_street")}

sets = [("僅人像", ["人像"]), ("人像+風景", ["人像","風景"]), ("人像+風景+街景", ["人像","風景","街景"])]
tests = [("t1_landscape","冷色風景"), ("t2_street","高反差街景"), ("t3_underexposed","曝光不足人像")]

print(f"{'訓練配對':<18}{'涵蓋率':>9}", end="")
for _,zh in tests: print(f"{zh:>12}", end="")
print()
print("-"*66)
for label, keys in sets:
    bs = [pairs[k][0] for k in keys]; as_ = [pairs[k][1] for k in keys]
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        lut,_ = fit_lut(bs, as_, n=17, lam=0.15)
    cov = buf.getvalue().split("(")[1].split(")")[0]
    print(f"{label:<18}{cov:>9}", end="")
    for name,_ in tests:
        t, truth = imread(f"sample/{name}.jpg"), imread(f"sample/{name}_truth.jpg")
        held = name.startswith("t1") and "風景" in keys or name.startswith("t2") and "街景" in keys
        d = de(apply_lut(t,lut,17), truth)
        print(f"{d.mean():9.2f}{'*' if held else ' '}  ", end="")
    print()
print("\n* = 該場景在訓練集內（非公平測試）")
print("基準線(不處理)：風景 9.73 / 街景 6.98 / 曝光不足 6.64")
