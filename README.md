# photo-color-match

驗證原型：**能不能從一組「原檔 / 調好的檔」自動學出調色風格，套用到整批照片上？**

```bash
uv run colormatch.py validate --before 原檔.jpg --after 調好.jpg --test a.jpg b.jpg c.jpg
```
（`uv` 會自動裝好 numpy / opencv / scipy，不用先建環境）

---

## 驗證結論（用合成資料，有已知真值）

方法是造一個已知的 grade（S 形對比 + 陰影推青 + 高光推橘 + 降飽和），
套在不同場景上當真值，再看程式能不能只從一組配對把它反解回來。
誤差用 CIE76 ΔE：**~1 = 剛好看得出來，>5 = 明顯不同**。

| | 冷色風景 | 高反差街景 | 曝光不足人像 |
|---|---|---|---|
| 不做任何處理（基準線） | 9.73 | 6.98 | 6.64 |
| A. Reinhard 統計轉移 | **38.28** | **52.36** | **16.54** |
| B. LUT 擬合（1 組配對，涵蓋 2.4%） | 6.85 | 6.93 | 3.17 |
| B. LUT 擬合（3 組配對，涵蓋 41.9%） | **1.37** | **1.57** | 2.96 |

### 1. Reinhard 統計轉移不能用 — 比不處理還糟三倍以上

它把**場景內容**和**調色風格**混在一起。參考照是人像、目標是風景，
兩者的色彩統計本來就天差地遠，硬套只會把畫面拉爛。
原始構想「拿一張調好的照片當參考」這條路，量出來是死的。

### 2. LUT 擬合可行，但成敗完全取決於「色彩涵蓋率」

落在訓練配對色彩範圍內的顏色 → **ΔE 1.4，肉眼幾乎分不出來**。
落在範圍外的顏色 → 跟不處理差不多，因為那是純外推，猜不到。

一張人像只涵蓋 17³ 網格的 2.4%。加到三組配對才 41.9%。

**這直接定義了產品邊界：**
- ✅ 可行 —— 同場次批次一致化（婚禮、活動、商攝）。同樣的光、同樣的膚色、
  同樣的衣服，涵蓋率天然就高，這正是最花體力的情境。
- ❌ 不可行 —— 「把我的風格套到任何照片上」的通用 preset 產生器。做不到，別賣這個。

### 3. 正則項的寫法是整個演算法的關鍵

第一版用 `||D·x||²` 平滑 LUT 輸出本身，結果 ΔE 高達 20–29（比不處理還爛）。
原因：單位變換的斜率是 1，而「平滑輸出」的先驗要求 LUT 平坦 —— 兩者直接衝突，
沒有樣本的顏色被外推到嚴重過衝。

改成平滑**偏移量** `y = x - identity`：

```
min_y  Σ w_v ||y_v − (t_v − id_v)||²  +  λ||D y||²  +  μ||y||²
```

同一組資料，ΔE 從 20.13 掉到 6.85。調色本質上是一個平滑的**偏移場**，
不是一個平滑的函數 —— 這一點寫錯，整個工具就沒有價值。

---

## 在 Lightroom Classic 使用 grade.cube

已查證：Lightroom Classic 7.3 版起原生支援直接匯入 `.cube`，不用額外轉檔。

1. Develop 模組 → Basic 面板上方的 **Profile Browser**
2. 右上角 `+` → **Import Profiles** → 選 `grade.cube`
3. 會出現在 Profile Browser 底下、以你匯入資料夾命名的新分類，套用方式跟套相機 profile 一樣

這其實是比「preset」更合理的位置——它變成疊在其他 Develop 調整**之下**的基礎呈現，
而不是覆蓋掉你其他設定的獨立 preset。

想要更進階版本（帶白平衡感知、能配合遮罩，即 Adobe 的 "Enhanced Profile"）才需要
額外經過 Photoshop 的 Camera Raw 濾鏡轉檔，那是加分項不是必要路徑。網格大小 17 或 33
都在相容範圍內（Enhanced Profile 上限 32），預設的 `--size 17` 沒問題。

Sources:
- [How To Install 3D LUTs in Lightroom Classic](https://proedu.com/blogs/photoshop-skills/how-to-install-3d-luts-in-lightroom-classic-a-step-by-step-guide)
- [How To Convert A LUT To A Lightroom Camera Profile](https://scottdavenportphoto.com/blog/how-to-convert-a-lut-to-a-lightroom-camera-profile)

---

## v1 範圍：不做 RAW，只吃 Lightroom 匯出的 JPG

刻意的取捨，不是還沒做完：

這個工具要處理的是「調色前 vs 調色後」這一層轉換，不是 RAW 解碼。
Lightroom 自己的 Develop 引擎已經處理好 demosaic 跟色彩空間轉換了 —— 工具只要接手
**匯出之後**的 JPG 就好，完全不用碰 RAW、不用管相機的色彩科學、不用嵌入 ICC profile 邏輯。

代價是你要在匯出時保持紀律：**before/after/target 全部用同一種色彩空間匯出**
（建議固定用 sRGB，最通用）。工具內部沒有讀 ICC profile，色彩空間不一致的話，
`fit_lut` 學到的偏移量會是錯的，而且不會有任何錯誤訊息提醒你 —— 這是目前最大的
「安靜失敗」風險，回家測試時務必檢查匯出設定。

如果之後真的需要跳過匯出這一步、直接吃 RAW，再加 `rawpy` 不遲，但那是完全不同的
工作量等級（demosaic、機身色彩校正、更大的檔案），不要在驗證出真正的產品價值之前碰它。

---

## 回家測試前：Lightroom 匯出檢查清單

為了讓第一次拿真實照片測試就成功，匯出時這幾點要一致：

- [ ] **色彩空間固定 sRGB**（Export 面板 → File Settings → Color Space）
- [ ] **Before 用完全沒調過的版本**（右鍵 Develop Settings → Reset，或用尚未進 Develop 的原始匯入版本）
- [ ] **After 用你已經調好、滿意的成品**
- [ ] Before 跟 After **必須是同一張照片**，不要裁切、不要旋轉（尺寸不同工具會自動縮放對齊，但構圖位移會讓像素對應錯位）
- [ ] Output Sharpening、雜訊消除等匯出時的處理，**before/after 兩邊開關要一致**（最簡單是兩邊都關），否則這些差異會被誤學成「調色風格」的一部分
- [ ] 3–5 張同場次、涵蓋不同膚色/場景/光線的 before-after 配對會比只有 1 張準很多（見上面「色彩涵蓋率」那節），如果手上有多張都調過同一風格，一起丟進去

---

## 還沒驗證的部分

**曝光/白平衡正規化**（`normalize()`）只做過健全性檢查，沒有真正驗證。
合成資料在這裡幫不上忙：我的真值刻意保留了曝光不足，但實際使用時你反而
**希望**它被拉齊 —— 真值定義本身就跟需求相反。

這一段（哪些差異該修掉、哪些是攝影師刻意的）是整個產品最主觀、最需要
真實照片去調的地方。`--strength` 參數就是這條線畫在哪裡，預設 0.7 是猜的。

**下一步該做的，是拿一場真實拍攝的原始檔 + 你調好的成品跑一次。**

---

## 指令

```bash
# 比較兩種方法，輸出對照圖 + grade.cube
uv run colormatch.py validate --before 原檔.jpg --after 調好.jpg --test *.jpg

# 只產生 .cube（可直接丟進 Lightroom / Premiere / DaVinci）
uv run colormatch.py learn --before 原檔.jpg --after 調好.jpg --out grade.cube

# 批次套用
uv run colormatch.py apply --cube grade.cube --input ./原始 --out ./成品 \
                           --normalize --ref-before 原檔.jpg
```

參數：`--size` LUT 網格邊長（預設 17）、`--lam` 平滑強度（預設 0.15）、
`--strength` 正規化力道（預設 0.7）。

## 已知限制

- 只吃 JPG/PNG/TIFF，不支援 RAW（見上方「v1 範圍」— 刻意決定，不是缺陷）。
- 不讀 ICC profile，色彩空間不一致會安靜失敗（無錯誤訊息，結果就是錯的）。
  務必確保 before/after/target 匯出時用同一種色彩空間，見上方檢查清單。
- 3D LUT 是全域變換，表達不了局部調整。如果你的 grade 用了遮罩或筆刷，
  擬合殘差會偏高，程式會警告 —— 那代表這張不適合當訓練配對。
- before/after 必須是同一張、沒有裁切位移。尺寸不同會自動縮放對齊。
