"""图像分割：阈值分割 / 区域生长 / 颜色量化聚类。

纯 Pillow 实现：

- threshold：全局（Otsu 自动或手定）二值化 -> 前景/背景两个区域。
- region   ：从灰度相似度出发做区域生长，得到多个空间区域。
- color    ：颜色量化（中位切分）-> 每个主色掩码做连通域 -> 颜色聚类区域。

输出：半透明彩色覆盖层（每区域一色）+ 区域边界 + 区域统计（数量/覆盖率/最大区域）。
"""
import colorsys

from PIL import Image, ImageDraw, ImageFilter

from .. import config
from . import util


_PALETTE = [
    (244, 67, 54), (33, 150, 243), (255, 193, 7), (76, 175, 80),
    (156, 39, 176), (0, 188, 212), (255, 87, 34), (63, 81, 181),
    (255, 235, 59), (0, 150, 136), (233, 30, 99), (121, 85, 72),
]


def _otsu(gray):
    hist = gray.histogram()
    total = sum(hist)
    if total == 0:
        return 127
    sum_all = sum(i * c for i, c in enumerate(hist))
    w_b = 0.0
    sum_b = 0.0
    best_t, best_v = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        v = w_b * w_f * (m_b - m_f) ** 2
        if v > best_v:
            best_v, best_t = v, t
    return best_t


def _region_grow(gray, tolerance):
    """用灰度相似度做无种子区域生长。

    每个未标记像素作为新区域的种子，向 4-邻域扩展；邻域像素与种子灰度差
    不超过 tolerance 时归入同一区域。不能用“当前像素与邻域像素比较”的链式
    扩展，否则平滑渐变会把整张图串成一个区域。
    """
    w, h, rows = util.gray_matrix(gray)
    labels = [[0] * w for _ in range(h)]
    components = {}
    next_label = 0
    tolerance = max(0, int(tolerance))

    for y in range(h):
        for x in range(w):
            if labels[y][x] != 0:
                continue
            seed_value = rows[y][x]
            next_label += 1
            label = next_label
            labels[y][x] = label
            stack = [(x, y)]
            points = []
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if labels[ny][nx] != 0:
                        continue
                    if abs(rows[ny][nx] - seed_value) <= tolerance:
                        labels[ny][nx] = label
                        stack.append((nx, ny))
            components[label] = points
    return labels, components


def _binary_mask(gray, params):
    """根据 method 生成二值掩码（L 图像，255 为前景）。"""
    value = params.get("value", None)
    if value is None:
        value = _otsu(gray)
    return gray.point(lambda v: 255 if v >= int(value) else 0)


def _labels_to_image(labels, w, h, region_colors):
    """把标签矩阵渲染成彩色区域图（RGB）。region_colors: {label: (r,g,b)}。"""
    data = []
    for y in range(h):
        for x in range(w):
            lbl = labels[y][x]
            data.append(region_colors.get(lbl, (0, 0, 0)))
    img = Image.new("RGB", (w, h))
    img.putdata(data)
    return img


def _components_from_mask(mask):
    w, h, rows = util.gray_matrix(mask)
    return util.connected_components(rows, w, h, threshold=128)


def _region_stats(components, w, h, orig_work):
    regions = []
    for label, pts in components.items():
        x0, y0, x1, y1 = util.points_bbox(pts)
        area = len(pts)
        mean = (0, 0, 0)
        try:
            crop = orig_work.crop((x0, y0, x1 + 1, y1 + 1)).resize((1, 1), Image.Resampling.BILINEAR)
            mean = crop.getpixel((0, 0))
        except Exception:
            pass
        regions.append({
            "id": label, "area": area, "coverage": round(area / float(w * h), 4),
            "box": [x0, y0, x1 - x0, y1 - y0], "mean_color": list(mean),
        })
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def segment(image, params):
    """执行分割，返回 overlay + 区域统计。"""
    method = params.get("method", "threshold")
    orig = util.ensure_rgb(image)
    work = util.downscale_to_max(orig, config.FEATURE_WORK_DIM)
    ratio = util.scale_ratio(orig.size, work.size)
    w, h = work.size

    if method == "color":
        labels, components = _color_clustering(work, int(params.get("colors", 6)))
    else:
        gray = util.to_grayscale(work)
        if method == "region":
            labels, components = _region_grow(gray, params.get("block", 15))
        else:
            mask = _binary_mask(gray, params)
            labels, components = _components_from_mask(mask)

    region_colors = {}
    for i, label in enumerate(components.keys()):
        region_colors[label] = _PALETTE[i % len(_PALETTE)]

    color_map = _labels_to_image(labels, w, h, region_colors)
    # 边界：区域图边缘检测
    boundaries = color_map.filter(ImageFilter.FIND_EDGES).point(lambda v: 0 if v < 30 else v)
    color_map = Image.blend(color_map, boundaries.convert("RGB"), 0.35)

    # 半透明叠加回原图
    overlay = Image.blend(work, color_map, float(params.get("alpha", 0.45)))
    overlay = overlay.resize(orig.size, Image.Resampling.BILINEAR)

    regions = _region_stats(components, w, h, work)
    for r in regions:
        r["box"] = [int(round(v * ratio)) for v in r["box"]]
        r["area"] = int(round(r["area"] * ratio * ratio))

    foreground = sum(r["area"] for r in regions)
    return {
        "method": method,
        "region_count": len(regions),
        "coverage": round(foreground / float(orig.size[0] * orig.size[1]), 4),
        "regions": regions,
        "image": overlay,
    }


def _color_clustering(rgb, n_colors):
    """颜色量化 + 每主色连通域，返回合并的 label 矩阵与 components。"""
    quantized = rgb.quantize(colors=max(2, n_colors), method=Image.Quantize.MEDIANCUT).convert("RGB")
    w, h, qrows = util.rgb_matrix(quantized)
    # 统计出现频率最高的颜色（背景白色除外）
    from collections import Counter
    counter = Counter(qrows[y][x] for y in range(h) for x in range(w))
    target_colors = [c for c, _ in counter.most_common(n_colors + 2)]

    labels = [[0] * w for _ in range(h)]
    components = {}
    next_label = 0
    for color in target_colors:
        # 该颜色的二值掩码
        mask_rows = [[255 if qrows[y][x] == color else 0 for x in range(w)] for y in range(h)]
        _, comps = util.connected_components(mask_rows, w, h, threshold=128)
        for _lbl, pts in comps.items():
            if len(pts) < (w * h) * 0.002:  # 过滤过小区域
                continue
            next_label += 1
            for px, py in pts:
                labels[py][px] = next_label
            components[next_label] = pts
    return labels, components


def draw_region_outline(image, boxes, color=(255, 255, 255)):
    """在图上描出区域外接框（调试/展示用）。"""
    img = util.ensure_rgb(image).copy()
    draw = ImageDraw.Draw(img)
    for b in boxes:
        x, y, w, h = b
        draw.rectangle([x, y, x + w, y + h], outline=color, width=1)
    return img
