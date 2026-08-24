# /// script
# dependencies = ["numpy", "opencv-python-headless"]
# ///
"""
產生合成測試素材：一組已知 grade 的 before/after，加三張帶真值的測試場景。
用來驗證 colormatch.py 的擬合準確度（真值已知，才能算 ΔE）。

跑法：uv run tools/gen_synthetic.py
"""
from pathlib import Path

import cv2
import numpy as np

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample"

H, W = 600, 900


def scene(kind, seed):
    r = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32)
    if kind == "portrait":  # 暖色調、大片膚色、暗背景
        img[...] = [0.78, 0.60, 0.48]
        m = ((x - W * 0.5) ** 2 / (W * 0.22) ** 2 + (y - H * 0.55) ** 2 / (H * 0.42) ** 2) < 1
        img[~m] = [0.12, 0.13, 0.16]
    elif kind == "landscape":  # 冷色調、亮天空
        img[..., 0] = 0.35 + 0.45 * (1 - y / H)
        img[..., 1] = 0.45 + 0.42 * (1 - y / H)
        img[..., 2] = 0.62 + 0.33 * (1 - y / H)
        g = y > H * 0.62
        img[g] = [0.22, 0.34, 0.18]
    else:  # 高反差街景
        img[...] = [0.45, 0.45, 0.47]
        for _ in range(28):
            cx, cy = r.integers(0, W), r.integers(0, H)
            cv2.circle(img, (int(cx), int(cy)), int(r.integers(30, 110)), [float(v) for v in r.random(3)], -1)
    img += r.normal(0, 0.015, img.shape).astype(np.float32)  # 一點雜訊
    return np.clip(img, 0, 1)


def grade(img):
    """已知的『調色風格』：S 形對比 + 青色陰影 + 橘色高光 + 降飽和。"""
    x = np.clip(img, 0, 1)
    s = x * x * (3 - 2 * x)  # smoothstep 對比
    x = x + 0.85 * (s - x)
    lum = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    x = x + (1 - lum) * np.array([-0.06, 0.015, 0.075], np.float32)  # 陰影推青
    x = x + lum * np.array([0.05, 0.012, -0.055], np.float32)  # 高光推橘
    gray = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    x = gray[..., None] + 0.88 * (x - gray[..., None])  # 降飽和
    return np.clip(x, 0, 1)


def save(path, img):
    cv2.imwrite(str(path), cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    base = scene("portrait", 1)
    save(SAMPLE_DIR / "before.jpg", base)
    save(SAMPLE_DIR / "after.jpg", grade(base))
    for name, kind, seed, gain in [
        ("t1_landscape", "landscape", 2, 1.0),
        ("t2_street", "street", 3, 1.0),
        ("t3_underexposed", "portrait", 4, 0.55),
    ]:
        s = np.clip(scene(kind, seed) * gain, 0, 1)
        save(SAMPLE_DIR / f"{name}.jpg", s)
        save(SAMPLE_DIR / f"{name}_truth.jpg", grade(s))  # 真值：同一個 grade 直接套上去
    print(f"寫入 {SAMPLE_DIR}/")


if __name__ == "__main__":
    main()
