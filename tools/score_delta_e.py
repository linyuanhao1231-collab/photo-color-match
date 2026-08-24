# /// script
# dependencies = ["numpy", "opencv-python-headless", "scipy"]
# ///
import sys, numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import os
os.chdir(__import__("pathlib").Path(__file__).resolve().parent.parent)  # 讓 "sample/..." 相對路徑一律以 repo 根目錄為準
from colormatch import imread, fit_lut, apply_lut, reinhard, read_cube

def de(a, b):
    """CIE76 ΔE，感知色差。~1 = 剛好看得出來，>5 = 明顯不同。"""
    la = cv2.cvtColor(a.astype(np.float32), cv2.COLOR_RGB2Lab)
    lb = cv2.cvtColor(b.astype(np.float32), cv2.COLOR_RGB2Lab)
    return np.sqrt(((la-lb)**2).sum(-1))

before, after = imread("sample/before.jpg"), imread("sample/after.jpg")
print("=== 網格大小 / 平滑強度 掃描（對未見場景的泛化能力）===\n")
tests = [("t1_landscape","冷色風景"), ("t2_street","高反差街景"), ("t3_underexposed","曝光不足人像")]

print(f"{'設定':<16}", end="")
for _, zh in tests: print(f"{zh:>14}", end="")
print(f"{'訓練殘差':>12}")
print("-"*76)

for n, lam in [(9,0.05),(17,0.02),(17,0.05),(17,0.15),(17,0.5),(33,0.15)]:
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        lut, resid = fit_lut(before, after, n=n, lam=lam)
    cov = buf.getvalue().split("：")[1].split(" ")[0]
    print(f"n={n:<3} lam={lam:<6}", end="")
    for name, _ in tests:
        t, truth = imread(f"sample/{name}.jpg"), imread(f"sample/{name}_truth.jpg")
        d = de(apply_lut(t, lut, n), truth)
        print(f"  ΔE {d.mean():5.2f}/{np.percentile(d,95):5.2f}", end="")
    print(f"   {resid:.4f}   [{cov}]")

print("\n(格式：平均ΔE / 95百分位ΔE)\n")
print("=== 方法 A (Reinhard) 對照 ===")
for name, zh in tests:
    t, truth = imread(f"sample/{name}.jpg"), imread(f"sample/{name}_truth.jpg")
    d = de(reinhard(t, after), truth)
    print(f"  {zh:<12} ΔE {d.mean():6.2f} / {np.percentile(d,95):6.2f}")

print("\n=== 未套用任何處理（基準線）===")
for name, zh in tests:
    t, truth = imread(f"sample/{name}.jpg"), imread(f"sample/{name}_truth.jpg")
    d = de(t, truth)
    print(f"  {zh:<12} ΔE {d.mean():6.2f} / {np.percentile(d,95):6.2f}")
