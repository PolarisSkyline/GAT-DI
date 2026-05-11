"""
generate_nir_external_raw_txts.py
────────────────────────────────────────────────────────────────────────────────
最小化一次性辅助脚本：基于 NIR 图像序列，生成外部对比所需的 NIR-source raw txt。

【生成目标】（均放置到 external_trackers/raw/）
  bytetrack_nir_raw.txt  ← Ultralytics 内置 bytetrack.yaml
  botsort_nir_raw.txt    ← Ultralytics 内置 botsort.yaml（with_reid=False）

【Table A 外部对比口径（精简版）】
  ByteTrack-NIR / BoT-SORT-NIR / GAT-DI (Scene-aware + AQG)

【使用方式】
  python generate_nir_external_raw_txts.py
  python generate_nir_external_raw_txts.py --trackers bytetrack
  python generate_nir_external_raw_txts.py --conf 0.3

【依赖】
  pip install ultralytics pyyaml
  （numpy 通常已随 ultralytics 安装）
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────────────────────
NIR_SEQ_DIR = r"E:\mokuai\track_val\peizhun\nir_degraded"

NIR_WEIGHTS_DIR = (
    r"E:\module\ultralytics-main\ultralytics-main\runs\detect\runs"
    r"\nir_experiments\yolov11n_nir_baseline3\weights"
)

OUTPUT_ROOT = r"E:\mokuai\video\peizhun\data\experiments"
EXT_RAW_DIR = os.path.join(OUTPUT_ROOT, "external_trackers", "raw")

ALL_TRACKERS = ["bytetrack", "botsort"]

_RAW_TXT_NAME = {
    "bytetrack": "bytetrack_nir_raw.txt",
    "botsort":   "botsort_nir_raw.txt",
}


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_weights(weights_dir: str) -> str:
    for name in ("best.pt", "last.pt"):
        p = os.path.join(weights_dir, name)
        if os.path.isfile(p):
            print(f"[weights] 找到 {name} → {p}")
            return p
    pts = sorted(glob.glob(os.path.join(weights_dir, "*.pt")),
                 key=os.path.getmtime, reverse=True)
    if pts:
        print(f"[weights] 使用最新 .pt → {pts[0]}")
        return pts[0]
    raise FileNotFoundError(f"[weights] ❌ {weights_dir} 下未找到任何 .pt 文件。")


def _collect_images(nir_seq_dir: str) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(nir_seq_dir, ext)))
        paths.extend(glob.glob(os.path.join(nir_seq_dir, "**", ext), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"[nir_seq] ❌ {nir_seq_dir} 下未找到图像文件。")
    print(f"[nir_seq] 发现 {len(paths)} 张图像")
    return paths


def _write_mot_txt(records: list[tuple], output_txt: str) -> int:
    """records: [(frame, tid, x, y, w, h, conf), ...]"""
    os.makedirs(os.path.dirname(os.path.abspath(output_txt)), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for frame, tid, x, y, w, h, conf in records:
            f.write(
                f"{frame},{tid},{x:.2f},{y:.2f},"
                f"{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
            )
    return len(records)


# ══════════════════════════════════════════════════════════════════════════════
# 内置 IoU-SORT 实现（用于 SORT-NIR 和 DeepSORT-NIR，无需外部包）
# ══════════════════════════════════════════════════════════════════════════════

def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU 矩阵 (M,N)，输入格式 [x1,y1,x2,y2]。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1 = boxes_a[:, 0:1]; ay1 = boxes_a[:, 1:2]
    ax2 = boxes_a[:, 2:3]; ay2 = boxes_a[:, 3:4]
    bx1 = boxes_b[:, 0];   by1 = boxes_b[:, 1]
    bx2 = boxes_b[:, 2];   by2 = boxes_b[:, 3]
    iw = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _greedy_match(iou_mat: np.ndarray, iou_thresh: float):
    matched = []
    if iou_mat.size == 0:
        return matched
    flat = np.argsort(-iou_mat.ravel())
    used_t, used_d = set(), set()
    for idx in flat:
        t, d = divmod(int(idx), iou_mat.shape[1])
        if iou_mat[t, d] < iou_thresh:
            break
        if t in used_t or d in used_d:
            continue
        matched.append((t, d))
        used_t.add(t); used_d.add(d)
    return matched


class _Track:
    _next_id = 1

    def __init__(self, box: np.ndarray, conf: float, max_age: int):
        self.tid     = _Track._next_id; _Track._next_id += 1
        self.box     = box.copy()   # [x1,y1,x2,y2]
        self.conf    = conf
        self.age     = 0
        self.hits    = 1
        self.max_age = max_age

    @property
    def alive(self):
        return self.age <= self.max_age

    def predict(self):
        self.age += 1   # 保持上一帧位置（无 Kalman）

    def update(self, box: np.ndarray, conf: float):
        self.box = box.copy(); self.conf = conf
        self.age = 0; self.hits += 1


def _run_iou_sort(
    all_dets: list[list[tuple]],
    iou_thresh: float = 0.3,
    min_hits:   int   = 1,
    max_age:    int   = 3,
) -> list[tuple]:
    """
    纯 IoU 贪心 SORT（无 Kalman，无 ReID）。
    返回 [(frame, tid, x, y, w, h, conf), ...]。
    """
    _Track._next_id = 1
    tracks: list[_Track] = []
    records: list[tuple] = []

    for frame_idx, dets in enumerate(all_dets, start=1):
        for trk in tracks:
            trk.predict()

        if dets:
            det_arr  = np.array([[d[0], d[1], d[2], d[3]] for d in dets])
            det_conf = np.array([d[4] for d in dets])

            if tracks:
                trk_arr = np.array([t.box for t in tracks])
                iou_mat = _iou_batch(trk_arr, det_arr)
                matched = _greedy_match(iou_mat, iou_thresh)
                matched_d = {d for _, d in matched}
                for t_idx, d_idx in matched:
                    tracks[t_idx].update(det_arr[d_idx], det_conf[d_idx])
                for d_idx in range(len(dets)):
                    if d_idx not in matched_d:
                        tracks.append(_Track(det_arr[d_idx], det_conf[d_idx], max_age))
            else:
                for i in range(len(dets)):
                    tracks.append(_Track(det_arr[i], det_conf[i], max_age))

        for trk in tracks:
            if trk.age == 0 and trk.hits >= min_hits:
                x1, y1, x2, y2 = trk.box
                records.append((frame_idx, trk.tid,
                                 x1, y1, x2 - x1, y2 - y1, trk.conf))

        tracks = [t for t in tracks if t.alive]

    return records


def _yolo_detect_all(model, image_paths, conf, iou, imgsz) -> list[list[tuple]]:
    """纯检测（无 tracker），返回 per-frame list of (x1,y1,x2,y2,conf)。"""
    all_dets: list[list[tuple]] = []
    for result in model.predict(source=image_paths, conf=conf, iou=iou,
                                imgsz=imgsz, stream=True, verbose=False):
        frame_dets = []
        if result.boxes is not None and len(result.boxes):
            xyxy  = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, confs):
                frame_dets.append((float(box[0]), float(box[1]),
                                   float(box[2]), float(box[3]), float(c)))
        all_dets.append(frame_dets)
    return all_dets


# ══════════════════════════════════════════════════════════════════════════════
# BoT-SORT：生成 with_reid=False 的临时 yaml
# ══════════════════════════════════════════════════════════════════════════════

def _make_botsort_no_reid_yaml(enable_gmc: bool = True) -> str:
    """
    生成临时 botsort yaml：with_reid=False，可选是否保留 GMC。
    enable_gmc=True  → 保留 sparseOptFlow（分批处理时内存可控）
    enable_gmc=False → gmc_method=none（全量处理时用于降低内存）
    """
    import yaml as _yaml

    # 找 Ultralytics 内置 botsort.yaml
    src_yaml = None
    try:
        import ultralytics
        for candidate in [
            Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml",
            Path(ultralytics.__file__).parent / "trackers" / "botsort.yaml",
        ]:
            if candidate.exists():
                src_yaml = candidate
                break
    except Exception:
        pass

    if src_yaml:
        with open(src_yaml, "r", encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}
    else:
        # 兜底最小配置
        cfg = {
            "tracker_type":      "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh":  0.1,
            "new_track_thresh":  0.25,
            "track_buffer":      30,
            "match_thresh":      0.8,
            "fuse_score":        True,
        }

    cfg["with_reid"] = False
    cfg.pop("reid_weights", None)
    if not enable_gmc:
        cfg["gmc_method"] = "none"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        prefix="botsort_no_reid_", dir=tempfile.gettempdir(),
        encoding="utf-8",
    )
    _yaml.dump(cfg, tmp)
    tmp.close()
    gmc_note = "gmc=none" if not enable_gmc else f"gmc={cfg.get('gmc_method','sparseOptFlow')}"
    print(f"[botsort] 临时 yaml（with_reid=False, {gmc_note}）: {tmp.name}")
    return tmp.name


# ══════════════════════════════════════════════════════════════════════════════
# 各 tracker 生成函数
# ══════════════════════════════════════════════════════════════════════════════

def _ul_results_to_records(results) -> list[tuple]:
    records = []
    for frame_idx, result in enumerate(results, start=1):
        if result.boxes is None:
            continue
        boxes = result.boxes
        if not hasattr(boxes, "id") or boxes.id is None:
            continue
        ids   = boxes.id.cpu().numpy().astype(int)
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for tid, box, c in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                records.append((frame_idx, int(tid), x1, y1, w, h, float(c)))
    return records


def gen_sort(model, image_paths, output_txt, conf, iou, imgsz) -> bool:
    print(f"\n[generate] ▶ SORT-NIR  (YOLO-NIR 检测 + IoU-SORT 关联，无需 sort.yaml)")
    print(f"           输出 → {output_txt}")
    try:
        all_dets = _yolo_detect_all(model, image_paths, conf, iou, imgsz)
        records  = _run_iou_sort(all_dets, iou_thresh=0.3, min_hits=1, max_age=3)
    except Exception as e:
        print(f"[generate] ❌ SORT 失败：{e}")
        return False
    n = _write_mot_txt(records, output_txt)
    if n == 0:
        print(f"[generate] ⚠️  SORT 写出行数为 0（尝试降低 --conf）")
        return False
    print(f"[generate] ✅ SORT-NIR → {n} 行写入 {output_txt}")
    return True


def gen_deepsort(model, image_paths, output_txt, conf, iou, imgsz) -> bool:
    """DeepSORT 无 ReID 退化版：IoU-SORT，max_age=5（略宽松，贴近 DeepSORT 默认行为）。"""
    print(f"\n[generate] ▶ DEEPSORT-NIR  (YOLO-NIR 检测 + IoU-SORT，无 ReID，无需 deepsort.yaml)")
    print(f"           输出 → {output_txt}")
    try:
        all_dets = _yolo_detect_all(model, image_paths, conf, iou, imgsz)
        records  = _run_iou_sort(all_dets, iou_thresh=0.25, min_hits=1, max_age=5)
    except Exception as e:
        print(f"[generate] ❌ DeepSORT 失败：{e}")
        return False
    n = _write_mot_txt(records, output_txt)
    if n == 0:
        print(f"[generate] ⚠️  DeepSORT 写出行数为 0")
        return False
    print(f"[generate] ✅ DEEPSORT-NIR → {n} 行写入 {output_txt}")
    return True


def gen_bytetrack(model, image_paths, output_txt, conf, iou, imgsz) -> bool:
    print(f"\n[generate] ▶ BYTETRACK-NIR  tracker=bytetrack.yaml")
    print(f"           输出 → {output_txt}")
    try:
        results = model.track(
            source=image_paths, tracker="bytetrack.yaml",
            conf=conf, iou=iou, imgsz=imgsz,
            stream=True, verbose=False, persist=False,
        )
        records = _ul_results_to_records(results)
    except Exception as e:
        import traceback
        print(f"[generate] ❌ ByteTrack 失败：{repr(e)}")
        traceback.print_exc()
        return False
    n = _write_mot_txt(records, output_txt)
    if n == 0:
        print(f"[generate] ⚠️  ByteTrack 写出行数为 0")
        return False
    print(f"[generate] ✅ BYTETRACK-NIR → {n} 行写入 {output_txt}")
    return True


def gen_botsort(model, image_paths, output_txt, conf, iou, imgsz,
                batch_size: int = 50) -> bool:
    """
    BoT-SORT 分批推理版本：
    - 每批 batch_size 帧，批次间 persist=True 保持追踪状态连续
    - 重新启用 GMC（sparseOptFlow），观察对结果的影响
    - 每批结束后主动释放内存，避免 OOM
    """
    print(f"\n[generate] ▶ BOTSORT-NIR  (botsort.yaml, with_reid=False, GMC=ON, batch={batch_size})")
    print(f"           输出 → {output_txt}")

    import gc

    # 释放前序 tracker 遗留内存
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    tmp_yaml = None
    try:
        tmp_yaml = _make_botsort_no_reid_yaml(enable_gmc=True)   # GMC 开启

        all_records: list[tuple] = []
        total = len(image_paths)
        n_batches = (total + batch_size - 1) // batch_size

        for b_idx in range(n_batches):
            b_start = b_idx * batch_size
            b_end   = min(b_start + batch_size, total)
            batch   = image_paths[b_start:b_end]
            print(f"  [batch {b_idx+1}/{n_batches}] 帧 {b_start+1}–{b_end} ...", end=" ")

            try:
                results = model.track(
                    source=batch,
                    tracker=tmp_yaml,
                    conf=conf, iou=iou, imgsz=imgsz,
                    stream=True, verbose=False,
                    persist=True,   # 批次间保持追踪器状态
                )
                for local_idx, result in enumerate(results):
                    frame_idx = b_start + local_idx + 1   # 全局帧号（从 1 开始）
                    if result.boxes is None:
                        continue
                    boxes = result.boxes
                    if not hasattr(boxes, "id") or boxes.id is None:
                        continue
                    ids   = boxes.id.cpu().numpy().astype(int)
                    xyxy  = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    for tid, box, c in zip(ids, xyxy, confs):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        if w > 0 and h > 0:
                            all_records.append(
                                (frame_idx, int(tid), x1, y1, w, h, float(c))
                            )
                print(f"累计 {len(all_records)} 行")
            except Exception as batch_exc:
                import traceback
                print(f"\n[generate] ❌ 第 {b_idx+1} 批失败：{repr(batch_exc)}")
                traceback.print_exc()
                return False

            # 批次间释放内存
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    except Exception as e:
        import traceback
        print(f"[generate] ❌ BoT-SORT 失败：{repr(e)}")
        print("[generate] 完整 traceback：")
        traceback.print_exc()
        return False
    finally:
        if tmp_yaml and os.path.exists(tmp_yaml):
            try:
                os.remove(tmp_yaml)
            except OSError:
                pass

    n = _write_mot_txt(all_records, output_txt)
    if n == 0:
        print(f"[generate] ⚠️  BoT-SORT 写出行数为 0")
        return False
    print(f"[generate] ✅ BOTSORT-NIR (GMC=ON) → {n} 行写入 {output_txt}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

_GEN_FN = {
    "bytetrack": gen_bytetrack,
    "botsort":   gen_botsort,
}


def run_generate(nir_seq_dir, weights_path, out_dir,
                 trackers, conf, iou, imgsz,
                 batch_size: int = 50) -> dict[str, bool]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ 未找到 ultralytics，请先安装：pip install ultralytics")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)
    image_paths = _collect_images(nir_seq_dir)
    print(f"\n[weights] 加载 NIR 权重: {weights_path}")
    model = YOLO(weights_path)

    status: dict[str, bool] = {}
    for tracker_name in trackers:
        out_txt = os.path.join(out_dir, _RAW_TXT_NAME[tracker_name])
        if tracker_name == "botsort":
            status[tracker_name] = gen_botsort(
                model, image_paths, out_txt, conf, iou, imgsz, batch_size
            )
        else:
            status[tracker_name] = _GEN_FN[tracker_name](
                model, image_paths, out_txt, conf, iou, imgsz
            )
    return status


def main():
    parser = argparse.ArgumentParser(
        description="基于 NIR 图像序列生成外部对比 NIR-source raw txt。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--nir_seq",     default=NIR_SEQ_DIR)
    parser.add_argument("--nir_weights", default=None,
                        help="权重文件或目录（默认自动查找 best.pt）")
    parser.add_argument("--out_dir",     default=EXT_RAW_DIR)
    parser.add_argument("--trackers", nargs="+",
                        choices=ALL_TRACKERS, default=ALL_TRACKERS)
    parser.add_argument("--conf",       type=float, default=0.5)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--batch_size", type=int,   default=50,
                        help="BoT-SORT 分批推理每批帧数（默认 50，减小可降低内存占用）")
    args = parser.parse_args()

    raw_w = args.nir_weights or NIR_WEIGHTS_DIR
    weights_path = _resolve_weights(raw_w) if os.path.isdir(raw_w) else raw_w
    if not os.path.isfile(weights_path):
        print(f"❌ 权重文件不存在: {weights_path}"); sys.exit(1)

    print("\n" + "=" * 70)
    print("  NIR-source External Tracker Raw TXT 生成工具（精简版）")
    print("  生成目标：bytetrack_nir_raw.txt / botsort_nir_raw.txt")
    print(f"  NIR 序列  : {args.nir_seq}")
    print(f"  权重      : {weights_path}")
    print(f"  输出目录  : {args.out_dir}")
    print(f"  Trackers  : {', '.join(args.trackers)}")
    print("=" * 70)

    status = run_generate(args.nir_seq, weights_path, args.out_dir,
                          args.trackers, args.conf, args.iou, args.imgsz,
                          args.batch_size)

    print("\n" + "=" * 70)
    print("  生成结果汇报：")
    failed = []
    for t in args.trackers:
        ok = status.get(t, False)
        out_txt = os.path.join(args.out_dir, _RAW_TXT_NAME[t])
        print(f"  {'✅' if ok else '❌'} {t.upper()}-NIR → {out_txt}")
        if not ok:
            failed.append(t)
    print("=" * 70)

    if not failed:
        print(
            "\n✅ 全部 NIR-source raw txt 生成完毕。\n"
            "  下一步：\n"
            "    python run_all_experiments.py --run_external_trackers"
        )
    else:
        print(
            f"\n⚠️  以下 tracker 生成失败：{failed}\n"
            f"  重试命令：\n"
            f"    python generate_nir_external_raw_txts.py"
            f" --trackers {' '.join(failed)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
