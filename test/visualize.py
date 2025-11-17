# visualize.py

#!/usr/bin/env python3
"""
Draw YOLO-format label files (.txt with: class x_center y_center width height, normalized)
onto images and save visualizations.

Usage examples:
  # visualize recursively, save outputs
  python visualize_yolo_labels.py --imgdir /path/to/images --labdir /path/to/labels --out /path/to/vis --recursive

  # provide names file (one class name per line), show instead of saving:
  python visualize_yolo_labels.py --imgdir imgs/ --labdir labels/ --names names.txt --show

Requirements:
  pip install opencv-python tqdm
"""

import argparse
from pathlib import Path
import cv2
from typing import Dict, List, Tuple
from tqdm import tqdm

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def find_images(imgdir: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in imgdir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    else:
        files = [p for p in imgdir.iterdir() if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of (class, x_center, y_center, width, height) normalized floats
    """
    if not label_path.exists():
        return []
    out = []
    with label_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            out.append((cls, x, y, w, h))
    return out


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Normalized center -> pixel xyxy"""
    cx = xc * img_w
    cy = yc * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))
    # clamp
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def color_for_class(cls: int) -> Tuple[int, int, int]:
    # deterministic color map: cycle through hues
    import colorsys
    h = (cls * 0.1234567) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
    return (int(255 * b), int(255 * g), int(255 * r))  # OpenCV BGR


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def draw_labels_on_image(img_path: Path, labels_dir: Path, out_path: Path, names: Dict[int, str],
                         thickness: int = 2, font_scale: float = 0.5, alpha: float = 1.0):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"WARNING: cannot read image {img_path}")
        return False
    h, w = img.shape[:2]

    # corresponding label file location: try same name .txt in labels_dir mirroring structure
    try:
        rel = img_path.relative_to(args.imgdir)
    except Exception:
        rel = img_path.name
    label_file = labels_dir / rel.with_suffix('.txt')

    labels = read_yolo_labels(label_file)

    overlay = img.copy()
    for (cls, xc, yc, bw, bh) in labels:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        color = color_for_class(cls)
        # rectangle
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        # label text
        label_txt = names.get(cls, str(cls))
        (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # background rectangle for text
        text_bg_y1 = max(0, y1 - int(th + 6))
        cv2.rectangle(overlay, (x1, text_bg_y1), (x1 + tw + 6, y1), color, -1)
        cv2.putText(overlay, label_txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # alpha blend if requested (alpha==1.0 is no blend)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        result = img
    else:
        result = overlay

    ensure_parent(out_path)
    cv2.imwrite(str(out_path), result)
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Visualize YOLO-format labels on images")
    p.add_argument('--imgdir', required=False, type=Path, help="Directory with images")
    p.add_argument('--labdir', required=False, type=Path, help="Directory with YOLO .txt labels (mirrors imgdir)")
    p.add_argument('--out', required=False, type=Path, help="Output directory for visualized images")
    p.add_argument('--names', required=False, type=Path, help="Optional class names file (one name per line)")
    p.add_argument('--recursive', action='store_true', help="Search image directory recursively")
    p.add_argument('--show', action='store_true', help="Show images (OpenCV window) instead of saving")
    p.add_argument('--thickness', type=int, default=2, help="Box thickness")
    p.add_argument('--alpha', type=float, default=1.0, help="Overlay alpha (1.0 = no blending, <1.0 blend boxes with image)")
    p.add_argument('--font-scale', type=float, default=0.5, help="Font scale for label text")
    p.add_argument('--max', type=int, default=0, help="Max images to process (0 = all)")
    args = p.parse_args()

    DEF_IMG_DIR = '../images'
    DEF_LABEL_DIR = '../output-yolo-labels'
    DEF_OUTPUT_DIR = '../visualize'
    DEF_LABEL_NAMES = ["car", "empty"]

    if args.imgdir is None:
        args.imgdir = Path(DEF_IMG_DIR)
    if args.labdir is None:
        args.labdir = Path(DEF_LABEL_DIR)
    if args.out is None:
        args.out = Path(DEF_OUTPUT_DIR)

    if not args.imgdir.exists():
        raise SystemExit(f"Image dir not found: {args.imgdir}")
    if not args.labdir.exists():
        print(f"WARNING: label dir does not exist: {args.labdir} (labels will be treated as missing)")

    names_map = DEF_LABEL_NAMES if args.names else {}
    images = find_images(args.imgdir, args.recursive)
    if args.max > 0:
        images = images[:args.max]
    if len(images) == 0:
        print("No images found.")
        raise SystemExit(0)

    if args.show:
        # simple show loop
        for img_path in images:
            out_tmp = Path('/tmp') / ('vis_' + img_path.name)
            draw_labels_on_image(img_path, args.labdir, out_tmp, names_map, thickness=args.thickness,
                                 font_scale=args.font_scale, alpha=args.alpha)
            img = cv2.imread(str(out_tmp))
            if img is None:
                continue
            cv2.imshow("vis", img)
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord('q')):  # ESC or q to quit
                break
        cv2.destroyAllWindows()
    else:
        processed = 0
        for img_path in tqdm(images, desc="visualizing"):
            try:
                rel = img_path.relative_to(args.imgdir)
            except Exception:
                rel = Path(img_path.name)
            out_path = args.out / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = draw_labels_on_image(img_path, args.labdir, out_path, names_map,
                                      thickness=args.thickness, font_scale=args.font_scale, alpha=args.alpha)
            if ok:
                processed += 1
        print(f"Saved {processed}/{len(images)} visualizations to {args.out}")
