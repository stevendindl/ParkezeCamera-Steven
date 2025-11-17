# prelabel.py

#!/usr/bin/env python3
"""
Generate YOLO-format pre-annotations (.txt) from a trained Ultralytics YOLO model.

Outputs:
  For each image.jpg (or png/etc), writes image.txt with lines:
    <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>

Example:
  python preannotate.py \
    --model best.pt \
    --src /path/to/images \
    --dst /path/to/labels \
    --conf 0.25 \
    --imgsz 640 \
    --save-visuals

Requires:
  pip install ultralytics tqdm opencv-python
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple
from ultralytics import YOLO
import cv2
from tqdm import tqdm


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

DEFAULT_INPUT_DIR = '../images'
DEFAULT_OUTPUT_DIR = '../output-yolo-labels'
MODEL_PATH = '../models/best-2025-11-17.pt'


def find_images(src: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        return [p for p in src.rglob('*') if p.suffix.lower() in IMAGE_EXTS]
    else:
        return [p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS]


def xyxy_to_yolo(xyxy: Tuple[float, float, float, float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    # xyxy = (x1, y1, x2, y2)
    x1, y1, x2, y2 = xyxy
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    # normalize
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)


def ensure_parent(outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)


def save_yolo_txt(labels: List[Tuple[int, float, float, float, float]], outpath: Path):
    """
    labels: list of (class_id, x_center, y_center, width, height) - normalized floats
    """
    ensure_parent(outpath)
    with outpath.open('w') as f:
        for cls, x, y, w, h in labels:
            # format with 6 decimal places
            f.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def main():
    p = argparse.ArgumentParser(description="Pre-annotate images with a YOLO model and save YOLO-format .txt files")
    p.add_argument('--model', required=False, help="path to .pt model (best.pt / last.pt / your custom)")
    p.add_argument('--src', required=False, help="source images directory")
    p.add_argument('--dst', required=False, help="destination labels directory (mirrors src structure if --mirror)")
    p.add_argument('--conf', type=float, default=0.25, help="confidence threshold for detections")
    p.add_argument('--imgsz', type=int, default=640, help="inference image size")
    p.add_argument('--save-visuals', action='store_true', help="save annotated images to dst/_visuals")
    p.add_argument('--recursive', action='store_true', default=True, help="search src recursively")
    p.add_argument('--skip-existing', action='store_true', help="skip files that already have .txt")
    p.add_argument('--max-images', type=int, default=0, help="stop after N images (0 = all)")
    p.add_argument('--device', type=str, default='', help="device for inference, e.g. 'cpu' or '0' (CUDA id). Leave empty to auto.")
    args = p.parse_args()

    if args.model is None:
        args.model = MODEL_PATH
    if args.src is None:
        args.src = DEFAULT_INPUT_DIR
    if args.dst is None:
        args.dst = DEFAULT_OUTPUT_DIR

    model = YOLO(args.model)
    if args.device:
        # ultralytics accepts device= in predict(); we will pass it when calling predict
        device = args.device
    else:
        device = None

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"Source dir {src} does not exist")

    images = find_images(src, recursive=args.recursive)
    images = sorted(images)
    if args.max_images > 0:
        images = images[:args.max_images]
    if len(images) == 0:
        print("No images found. Exiting.")
        return

    visuals_dir = dst / "_visuals"
    if args.save_visuals:
        visuals_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images. Running inference with model={args.model}, conf={args.conf}, imgsz={args.imgsz}")

    for img_path in tqdm(images):
        # determine output txt path, mirror structure relative to src
        try:
            rel = img_path.relative_to(src)
        except Exception:
            # fallback: just use name
            rel = Path(img_path.name)
        out_txt = dst / rel.with_suffix('.txt')
        if args.skip_existing and out_txt.exists():
            continue

        # run prediction
        # using model.predict which returns a list of Results (one per image)
        res = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, device=device, verbose=False, save=False)  # returns list
        # res could be a list; we used single image so take first
        if isinstance(res, list):
            out_res = res[0]
        else:
            out_res = res

        # fetch boxes; ultralytics Boxes object
        boxes = getattr(out_res, 'boxes', None)
        labels_to_write = []
        if boxes is not None and len(boxes) > 0:
            # boxes.xyxy, boxes.cls, boxes.conf exist
            # convert tensors to numpy if needed
            try:
                xyxys = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
            except Exception:
                # fallback for older/wrapper versions: boxes.xyxy, boxes.cls may already be lists
                xyxys = boxes.xyxy
                cls_ids = boxes.cls
                confs = boxes.conf

            # read image to know width/height (for safety if model inference changed shape)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"WARNING: couldn't read image {img_path}, skipping")
                continue
            h, w = img.shape[:2]

            for (x1, y1, x2, y2), clsid, conf in zip(xyxys, cls_ids, confs):
                # optional filter (we already used --conf at predict call, but keep guard)
                if conf < args.conf:
                    continue
                xc, yc, bw, bh = xyxy_to_yolo((float(x1), float(y1), float(x2), float(y2)), w, h)
                # clamp to [0,1]
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                labels_to_write.append((int(clsid), xc, yc, bw, bh))

            # write out .txt
            save_yolo_txt(labels_to_write, out_txt)

            # optionally save visualization image with boxes+labels drawn
            if args.save_visuals:
                # use ultralytics' visualization if available: out_res.plot() returns image BGR (cv2)
                try:
                    vis = out_res.plot()  # returns np.ndarray BGR
                    vis_out = visuals_dir / rel
                    ensure_parent(vis_out)
                    cv2.imwrite(str(vis_out), vis)
                except Exception:
                    # fallback: draw boxes manually
                    vis_img = img.copy()
                    for (x1, y1, x2, y2), clsid, conf in zip(xyxys, cls_ids, confs):
                        if conf < args.conf:
                            continue
                        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(vis_img, f"{clsid}:{conf:.2f}", (int(x1), int(max(0, y1 - 5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    vis_out = visuals_dir / rel
                    ensure_parent(vis_out)
                    cv2.imwrite(str(vis_out), vis_img)
        else:
            # no boxes -> ensure empty label file exists (useful for some training pipelines)
            save_yolo_txt([], out_txt)

    print("Done. Labels saved to:", dst)
    if args.save_visuals:
        print("Visualizations saved to:", visuals_dir)


if __name__ == '__main__':
    main()
