# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "scipy"]
# ///
"""
colormatch.py — 驗證原型：能否把一組照片自動套成同一個調性？

兩種方法，直接對照：
  A. Reinhard  — 只靠一張「已調好」的參考照，做 Lab 空間統計匹配。
                 缺點：把場景內容和調色風格混在一起。
  B. LUT fit   — 從 (原檔, 調好的檔) 配對反解出 3D LUT，也就是 grade 本身。
                 可外掛曝光/白平衡正規化，處理現場光線飄移。

用法：
  # 一次比較兩種方法，輸出對照圖 + grade.cube
  uv run colormatch.py validate --before orig.jpg --after graded.jpg --test a.jpg b.jpg

  # 實際批次套用
  uv run colormatch.py learn --before orig.jpg --after graded.jpg --out grade.cube
  uv run colormatch.py apply --cube grade.cube --input ./raw_jpgs --out ./done \
                             --normalize --ref-before orig.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------- 影像 IO / 色彩空間

_NON_SRGB_MARKERS = [b"Adobe RGB", b"ProPhoto", b"Display P3", b"Apple RGB", b"ColorMatch"]


def _warn_if_not_srgb(path, buf):
    """
    粗略掃描檔案內嵌的 ICC profile 描述字串。整套工具的數學完全沒有讀取/校正
    色彩空間，所以嵌了非 sRGB profile 的檔案會被當成 sRGB 硬算 —— 不會報錯，
    只會讓學出來的 grade 悄悄跑掉。這裡只做得到「提醒」，做不到「校正」。
    """
    for marker in _NON_SRGB_MARKERS:
        if marker in buf:
            print(
                f"  [!] {path} 內嵌了 {marker.decode()} 色彩描述檔，但這個工具全程假設 sRGB。"
                f"請重新以 sRGB 匯出，否則結果不可信。",
                file=sys.stderr,
            )
            return


def imread(path):
    """讀成 RGB float [0,1]。用 fromfile 以支援非 ASCII 路徑。"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    _warn_if_not_srgb(path, buf.tobytes())
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"讀不到影像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def imwrite(path, rgb):
    bgr = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise SystemExit(f"寫不出影像: {path}")
    buf.tofile(str(path))


def srgb_to_linear(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def luminance(lin_rgb):
    return lin_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------- 方法 A：Reinhard

def reinhard(target, ref, strength=1.0):
    """Lab 空間逐通道均值/標準差匹配。"""
    t = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_RGB2Lab)
    r = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_RGB2Lab)
    out = t.copy()
    for c in range(3):
        mt, st = t[..., c].mean(), t[..., c].std() + 1e-6
        mr, sr = r[..., c].mean(), r[..., c].std() + 1e-6
        out[..., c] = (t[..., c] - mt) * (sr / st) + mr
    out = t + strength * (out - t)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_Lab2RGB), 0, 1)


# ---------------------------------------------------------------- 方法 B：3D LUT 擬合

def _diff_operator(n):
    """3D 網格上的一階差分算子 D；D^T D 就是 Laplacian，當平滑先驗用。"""
    idx = np.arange(n ** 3).reshape(n, n, n)  # [b, g, r]
    rows, cols, vals, r0 = [], [], [], 0
    for axis in range(3):
        a = np.moveaxis(idx, axis, 0)
        u, v = a[:-1].ravel(), a[1:].ravel()
        rr = np.arange(r0, r0 + len(u))
        rows += [rr, rr]
        cols += [u, v]
        vals += [np.ones(len(u)), -np.ones(len(u))]
        r0 += len(u)
    return sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(r0, n ** 3),
    ).tocsr()


def _identity_lut(n):
    """單位 LUT，形狀 (n^3, 3)，flat index = ir + ig*n + ib*n^2。"""
    ax = np.linspace(0, 1, n, dtype=np.float32)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1)


def _trilinear_corners(img, n):
    """回傳 8 個角點的 (flat_index, weight)。"""
    g = np.clip(img, 0, 1) * (n - 1)
    i0 = np.minimum(np.floor(g).astype(np.int32), n - 2)
    f = g - i0
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = ((1 - f[..., 0]) if dr == 0 else f[..., 0]) * \
                    ((1 - f[..., 1]) if dg == 0 else f[..., 1]) * \
                    ((1 - f[..., 2]) if db == 0 else f[..., 2])
                flat = (i0[..., 0] + dr) + (i0[..., 1] + dg) * n + (i0[..., 2] + db) * n * n
                yield flat, w


def fit_lut(before, after, n=17, lam=0.15, mu=1e-3, max_samples=2_000_000, seed=0):
    """
    從像素對應 before -> after 擬合 3D LUT。

    關鍵：對「偏移量」y = x - identity 求解，而不是直接對 x。

        sum_v w_v ||y_v - (t_v - id_v)||^2  +  lam*||D y||^2  +  mu*||y||^2

    調色本質上是一個平滑的偏移場，不是一個平滑的函數。直接對 x 做平滑
    等於要求 LUT 平坦，那會跟斜率為 1 的單位變換衝突，讓沒有樣本的顏色
    被外推到嚴重過衝。改成對偏移量平滑之後，沒資料的區域會延續鄰近的
    偏移、再由 mu 緩緩衰減回不變，這才是色彩外推該有的行為。
    """
    pairs = [(before, after)] if isinstance(before, np.ndarray) else list(zip(before, after))
    bs, as_ = [], []
    for bi, ai in pairs:
        if bi.shape != ai.shape:
            ai = cv2.resize(ai, (bi.shape[1], bi.shape[0]), interpolation=cv2.INTER_AREA)
            print(f"  [!] before/after 尺寸不同，已縮放對齊 {bi.shape[1]}x{bi.shape[0]}")
        bs.append(bi.reshape(-1, 3))
        as_.append(ai.reshape(-1, 3))
    b, a = np.concatenate(bs), np.concatenate(as_)
    if len(b) > max_samples:
        sel = np.random.default_rng(seed).choice(len(b), max_samples, replace=False)
        b, a = b[sel], a[sel]

    N = n ** 3
    W = np.zeros(N, dtype=np.float64)
    T = np.zeros((N, 3), dtype=np.float64)
    for flat, w in _trilinear_corners(b, n):
        np.add.at(W, flat, w)
        for c in range(3):
            np.add.at(T[:, c], flat, w * a[:, c])

    covered = (W > 1e-8).sum()
    tgt = np.zeros((N, 3))
    nz = W > 1e-8
    tgt[nz] = T[nz] / W[nz, None]

    wn = W / max(W.mean(), 1e-12)          # 正規化，讓 lam 與樣本數無關
    D = _diff_operator(n)
    ident = _identity_lut(n)
    A = (sparse.diags(wn) + lam * (D.T @ D) + mu * sparse.identity(N)).tocsc()
    rhs = wn[:, None] * (tgt - ident)      # 資料項也改成偏移量

    y = np.column_stack([spsolve(A, rhs[:, c]) for c in range(3)]).astype(np.float32)
    lut = np.clip(ident + y, 0, 1)

    resid = float(np.sqrt(np.mean((apply_lut(b, lut, n) - a) ** 2)))
    print(f"  LUT {n}^3：{covered}/{N} 網格點有資料 ({covered/N:.1%})，擬合殘差 RMS = {resid:.4f}")
    if resid > 0.05:
        print("  [!] 殘差偏高 — 這個 grade 可能含局部調整(遮罩/筆刷)，單一全域 LUT 表達不了。")
    return lut, resid


def apply_lut(img, lut, n):
    out = np.zeros_like(img, dtype=np.float32)
    for flat, w in _trilinear_corners(img, n):
        out += lut[flat] * w[..., None]
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- .cube 讀寫

def write_cube(path, lut, n, title="colormatch"):
    with open(path, "w") as f:
        f.write(f'TITLE "{title}"\nLUT_3D_SIZE {n}\n')
        f.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        # .cube 規格：red 變化最快，正好等於我們的 flat index 排列
        for v in lut:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def read_cube(path):
    vals, n = [], None
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith("LUT_3D_SIZE"):
            n = int(line.split()[-1])
        elif line[0].isdigit() or line[0] in "-.":
            vals.append([float(x) for x in line.split()[:3]])
    if n is None or len(vals) != n ** 3:
        raise SystemExit(f"{path} 不是合法的 3D .cube")
    return np.array(vals, dtype=np.float32), n


# ---------------------------------------------------------------- 曝光 / 白平衡正規化

def baseline_stats(img):
    """在 linear 光空間量測：照明色偏 + 中位亮度。"""
    lin = srgb_to_linear(img)
    lum = luminance(lin)
    hi = lum >= np.percentile(lum, 90)          # 亮部估照明色
    if hi.sum() < 64:
        hi = np.ones_like(lum, dtype=bool)
    illum = lin[hi].reshape(-1, 3).mean(axis=0) + 1e-6
    return {"illum": illum / illum.mean(), "med_lum": float(np.median(lum)) + 1e-6}


def normalize(img, ref_stats, strength=0.7):
    """
    把單張照片的曝光/白平衡「部分」拉向基準。

    刻意不做滿：只想修掉測光飄移與環境光變化這類非預期差異，
    保留攝影者刻意的明暗差。strength 就是這條線畫在哪裡。
    """
    s = baseline_stats(img)
    wb = (ref_stats["illum"] / s["illum"]) ** strength
    ev = (ref_stats["med_lum"] / s["med_lum"]) ** strength
    lin = srgb_to_linear(img) * (wb * ev).astype(np.float32)
    return linear_to_srgb(lin).astype(np.float32)


# ---------------------------------------------------------------- 對照圖

def contact_sheet(panels, path, height=420):
    tiles = []
    for label, img in panels:
        h, w = img.shape[:2]
        t = cv2.resize(img, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
        t = (np.clip(t, 0, 1) * 255).astype(np.uint8)
        cv2.rectangle(t, (0, 0), (t.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(t, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    sheet = np.hstack([np.pad(t, ((0, 0), (0, 6), (0, 0))) for t in tiles])
    imwrite(path, sheet.astype(np.float32) / 255.0)


# ---------------------------------------------------------------- 子指令

def cmd_validate(args):
    before, after = imread(args.before), imread(args.after)
    print(f"學習 grade：{Path(args.before).name} -> {Path(args.after).name}")
    lut, _ = fit_lut(before, after, n=args.size, lam=args.lam)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_cube(out_dir / "grade.cube", lut, args.size, Path(args.after).stem)
    ref_stats = baseline_stats(before)

    # 健全性檢查：LUT 套回原檔應該要重現調好的成品
    contact_sheet(
        [("before", before), ("after (真值)", after), ("LUT 套回 before", apply_lut(before, lut, args.size))],
        out_dir / "00_sanity.jpg",
    )

    for i, p in enumerate(args.test, 1):
        t = imread(p)
        panels = [
            ("原始", t),
            ("A: Reinhard", reinhard(t, after)),
            ("B: LUT", apply_lut(t, lut, args.size)),
            ("B+正規化", apply_lut(normalize(t, ref_stats, args.strength), lut, args.size)),
        ]
        name = f"{i:02d}_{Path(p).stem}.jpg"
        contact_sheet(panels, out_dir / name)
        print(f"  -> {out_dir / name}")

    print(f"\n對照圖與 grade.cube 都在 {out_dir}/")
    print("先看 00_sanity.jpg：第 2、3 張若肉眼看不出差別，代表 LUT 忠實抓到了這個 grade。")


def cmd_learn(args):
    before, after = imread(args.before), imread(args.after)
    lut, _ = fit_lut(before, after, n=args.size, lam=args.lam)
    write_cube(args.out, lut, args.size, Path(args.after).stem)
    print(f"已寫出 {args.out}（可直接丟進 Lightroom / Premiere / DaVinci）")


def cmd_apply(args):
    lut, n = read_cube(args.cube)
    files = sorted(p for p in Path(args.input).iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"{args.input} 裡沒有影像")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_stats = baseline_stats(imread(args.ref_before)) if args.normalize else None
    if args.normalize and not args.ref_before:
        raise SystemExit("--normalize 需要同時給 --ref-before（學習時用的那張原檔）")

    for p in files:
        img = imread(p)
        if ref_stats:
            img = normalize(img, ref_stats, args.strength)
        imwrite(out_dir / p.name, apply_lut(img, lut, n))
        print(f"  {p.name}")
    print(f"\n{len(files)} 張輸出到 {out_dir}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="比較兩種方法並輸出對照圖")
    v.add_argument("--before", required=True, help="未調色的原檔")
    v.add_argument("--after", required=True, help="同一張、已調好的成品")
    v.add_argument("--test", nargs="+", required=True, help="要試套的其他照片")
    v.add_argument("--out-dir", default="./out")
    v.add_argument("--strength", type=float, default=0.7)
    v.set_defaults(func=cmd_validate)

    l = sub.add_parser("learn", help="只輸出 .cube")
    l.add_argument("--before", required=True)
    l.add_argument("--after", required=True)
    l.add_argument("--out", default="grade.cube")
    l.set_defaults(func=cmd_learn)

    a = sub.add_parser("apply", help="批次套用 .cube")
    a.add_argument("--cube", required=True)
    a.add_argument("--input", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--normalize", action="store_true", help="先做曝光/白平衡正規化")
    a.add_argument("--ref-before", help="正規化的基準（學習時的原檔）")
    a.add_argument("--strength", type=float, default=0.7)
    a.set_defaults(func=cmd_apply)

    for p in (v, l):
        p.add_argument("--size", type=int, default=17, help="LUT 網格邊長，預設 17")
        p.add_argument("--lam", type=float, default=0.05, help="平滑強度，預設 0.05")

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
