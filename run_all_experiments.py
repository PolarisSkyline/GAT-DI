"""
run_all_experiments.py
────────────────────────────────────────────────────────────────────────────────
一键生成论文 Table 1–5，与 gat_di_v3_final.py 终稿收敛版对齐。

使用方式：
    # 仅跑主实验 Table 1–5（默认）
    python run_all_experiments.py

    # 额外跑 Table A（adapter mode，需提前放置 raw txt）
    python run_all_experiments.py --run_external_trackers

    # 额外跑 Table 3B（Rescue α 敏感性，supplementary）
    python run_all_experiments.py --run_optional_rescue_sensitivity

    # 额外跑 Table A（runner mode，自动调用本地 ByteTrack 仓库）
    python run_all_experiments.py --run_external_trackers \\
        --external_tracker_mode runner \\
        --tracker_video_path /path/to/video_or_seq \\
        --sort_exp_file /path/to/sort.py \\
        --sort_ckpt /path/to/sort.pth \\
        --deepsort_exp_file /path/to/deepsort.py \\
        --deepsort_ckpt /path/to/deepsort.pth \\
        --deepsort_model_folder /path/to/reid_model/ \\
        --bytetrack_exp_file /path/to/bytetrack.py \\
        --bytetrack_ckpt /path/to/bytetrack.pth

依赖：
    gat_di_v3_final.py   （终稿收敛版）
    eval_utils.py

当前终版口径：
  方法名         : GAT-DI (Scene-aware + AQG)
  默认 AQG       : 0.45 / 0.75
  默认时间处理   : Fixed-FPS
  核心贡献       : Scene-aware + AQG
  RGB rescue     : optional auxiliary heuristic; not a core claimed contribution
  Table 1        : 主结果，ALL scene，含 Association_FPS
  Table 2        : 消融实验（B1: No Scene-Aware / B2: No AQG /
                   B3: No RGB Rescue (auxiliary) / Full Ours）
  Table 3A       : AQG 敏感性（* 标在 0.45/0.75）
  Table 4        : 场景分解（FOG / GLARE）
  Table 5        : Temporal Scaling Sensitivity
                   （Fixed-FPS Default vs Synthetic Variable-Scale）

注意：
  1. 所有结果以当前实际重跑为准。
  2. 终版资产（FINAL_CSVS）共 6 个 CSV（Table 3B 已降级为 supplementary）：
       - table1_main_results.csv
       - table2_ablation.csv
       - table3a_sensitivity_aqg.csv
       - table4_scenario_fog.csv
       - table4_scenario_glare.csv
       - table5_temporal_scaling.csv
  3. Table 3B（Rescue α 敏感性）为 supplementary，需 --run_optional_rescue_sensitivity 启用。
  4. Table A 为附加资产（tableA_external_tracker_comparison.csv），
     需 --run_external_trackers 启用，不影响 Table 1–5。
     【NIR-source 口径】外部对比缩小为：
       ByteTrack-NIR / BoT-SORT-NIR / GAT-DI (Scene-aware + AQG)
     所需 raw txt：
       bytetrack_nir_raw.txt / botsort_nir_raw.txt
  5. Table A 支持两种模式：
       adapter（默认）：从已有 raw txt 读取，无需安装外部 tracker repo。
       runner：直接调用本地 ByteTrack 仓库脚本运行，需提供完整配置。
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 路径配置（默认值）
# ──────────────────────────────────────────────────────────────────────────────
RGB_FILE = r"E:\module\ultralytics-main\ultralytics-main\rgb_norm.txt"
NIR_FILE = r"E:\module\ultralytics-main\ultralytics-main\nir_norm.txt"

GT_PATH = r"E:\module\ultralytics-main\ultralytics-main\gt.txt"
GT_FORMAT = "txt"
GT_VIDEO_ID = None
GT_MODALITY = None

OUTPUT_ROOT = r"E:\mokuai\video\peizhun\data\experiments"
TABLES_DIR = os.path.join(OUTPUT_ROOT, "tables")

# ──────────────────────────────────────────────────────────────────────────────
# External tracker 目录常量（与 eval_utils.py 保持一致）
# ──────────────────────────────────────────────────────────────────────────────
_EXT_ROOT    = os.path.join(OUTPUT_ROOT, "external_trackers")
EXT_RAW_DIR  = os.path.join(_EXT_ROOT, "raw")
EXT_NORM_DIR = os.path.join(_EXT_ROOT, "normalized")
EXT_EVAL_DIR = os.path.join(_EXT_ROOT, "eval")

# External tracker raw txt 默认路径（NIR-source 口径，Table A 只用 ByteTrack + BoT-SORT）
DEFAULT_BYTETRACK_RAW = os.path.join(EXT_RAW_DIR, "bytetrack_nir_raw.txt")
DEFAULT_BOTSORT_RAW   = os.path.join(EXT_RAW_DIR, "botsort_nir_raw.txt")
# 以下保留供旧接口向后兼容，Table A 不再调用
DEFAULT_SORT_RAW      = os.path.join(EXT_RAW_DIR, "sort_nir_raw.txt")
DEFAULT_DEEPSORT_RAW  = os.path.join(EXT_RAW_DIR, "deepsort_nir_raw.txt")

# ByteTrack 仓库默认路径（已验证）
DEFAULT_BYTETRACK_REPO_ROOT = (
    r"E:\module\ultralytics-main\ultralytics-main\files (1)"
    r"\ByteTrack-main\ByteTrack-main"
)

# ──────────────────────────────────────────────────────────────────────────────
# 导入
# ──────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gat_di_v3_final import lifecycle_fusion_final
from eval_utils import (
    load_gt,
    load_mot_txt,
    get_scene_frames,
    evaluate_mot,
    timed_run,
    print_metrics_table,
    save_results_csv,
    run_baseline_rgb_only,
    run_baseline_nir_only,
    run_single_threshold,
    run_dual_threshold,
    run_motion_delayed_conf,
    # ── adapter mode（已有 + 新增 BoT-SORT）──────────────────────────────
    run_baseline_sort,
    run_baseline_deepsort,
    run_baseline_bytetrack,
    run_baseline_botsort,
    # ── runner mode（新增）─────────────────────────────────────────────────
    run_external_sort_runner,
    run_external_deepsort_runner,
    run_external_bytetrack_runner,
    ensure_external_tracker_dirs,
)

# ──────────────────────────────────────────────────────────────────────────────
# 全局常量
# ──────────────────────────────────────────────────────────────────────────────
OURS_NAME = "GAT-DI (Scene-aware + AQG)"

# 终版主资产（不含 Table 3B）
FINAL_CSVS = [
    "table1_main_results.csv",
    "table2_ablation.csv",
    "table3a_sensitivity_aqg.csv",
    "table4_scenario_fog.csv",
    "table4_scenario_glare.csv",
    "table5_temporal_scaling.csv",
]

# supplementary 资产（Table 3B：Rescue α 敏感性，非终版主资产）
SUPPLEMENTARY_CSVS = [
    "table3b_sensitivity_rescue.csv",
]

# 附加资产（Table A，独立于终版资产）
EXTERNAL_CSVS = [
    "tableA_external_tracker_comparison.csv",
]

OBSOLETE_CSVS = [
    "table3_sensitivity.csv",
    "table3_sensitivity_aqg.csv",
    "table5_vfr.csv",
]


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all paper experiments and export final CSV assets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=OUTPUT_ROOT,
        help="实验输出根目录，默认使用脚本内 OUTPUT_ROOT。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="用于记录到 meta.json 的随机种子。",
    )
    parser.add_argument(
        "--clean_obsolete",
        action="store_true",
        help="若指定，则在运行前删除 tables 目录下 obsolete CSV。",
    )

    # ── 外部 tracker 对比（不影响 Table 1–5）──────────────────────────────
    parser.add_argument(
        "--run_external_trackers",
        action="store_true",
        help=(
            "启用外部 tracker 对比（Table A：SORT / DeepSORT / ByteTrack vs Ours）。\n"
            "不指定时不跑外部 tracker，Table 1–5 完全不受影响。"
        ),
    )
    parser.add_argument(
        "--external_tracker_mode",
        type=str,
        choices=["adapter", "runner"],
        default="adapter",
        help=(
            "外部 tracker 运行模式：\n"
            "  adapter（默认）：从 raw txt 读取并标准化，无需安装外部 repo。\n"
            "    所需文件：--bytetrack_raw_txt / --botsort_raw_txt\n"
            "  runner：直接调用本地 ByteTrack 仓库脚本运行追踪，自动产出 raw txt。\n"
            "    必须配置：--bytetrack_repo_root / --tracker_video_path\n"
            "               + 各 tracker 的 --*_exp_file / --*_ckpt"
        ),
    )

    # ── Table 3B supplementary（Rescue α 敏感性，默认不跑）──────────────
    parser.add_argument(
        "--run_optional_rescue_sensitivity",
        action="store_true",
        help=(
            "启用 Table 3B：Rescue α 敏感性分析（supplementary）。\n"
            "RGB rescue 为 optional auxiliary heuristic，此分析不属于终版主资产。\n"
            "不指定时不跑，主实验 Table 1–5 完全不受影响。"
        ),
    )

    # ── adapter mode 参数（Table A 只用 ByteTrack + BoT-SORT）────────────
    parser.add_argument(
        "--bytetrack_raw_txt",
        type=str,
        default=DEFAULT_BYTETRACK_RAW,
        help=(
            "ByteTrack NIR-source raw txt 路径（adapter mode 使用）。\n"
            f"默认: {DEFAULT_BYTETRACK_RAW}"
        ),
    )
    parser.add_argument(
        "--botsort_raw_txt",
        type=str,
        default=DEFAULT_BOTSORT_RAW,
        help=(
            "BoT-SORT NIR-source raw txt 路径（adapter mode 使用）。\n"
            f"默认: {DEFAULT_BOTSORT_RAW}\n"
            "使用 generate_nir_external_raw_txts.py 可基于 NIR 图像序列自动生成此文件。"
        ),
    )

    # ── runner mode 公共参数 ───────────────────────────────────────────────
    parser.add_argument(
        "--bytetrack_repo_root",
        type=str,
        default=DEFAULT_BYTETRACK_REPO_ROOT,
        help=(
            "本地 ByteTrack 仓库根目录（runner mode 使用）。\n"
            "该目录下应存在 tools/track.py / tools/track_sort.py / tools/track_deepsort.py。\n"
            f"默认: {DEFAULT_BYTETRACK_REPO_ROOT}"
        ),
    )
    parser.add_argument(
        "--tracker_video_path",
        type=str,
        default=None,
        help=(
            "输入视频或图像序列路径（runner mode 必须提供）。\n"
            "传给外部 tracker 脚本的 --path 参数。"
        ),
    )
    parser.add_argument(
        "--tracker_conf",
        type=float,
        default=0.6,
        help="外部 tracker 检测置信度阈值（runner mode，默认 0.6）。",
    )
    parser.add_argument(
        "--tracker_nms",
        type=float,
        default=0.7,
        help="外部 tracker NMS 阈值（runner mode，默认 0.7）。",
    )
    parser.add_argument(
        "--tracker_tsize",
        type=int,
        default=640,
        help="外部 tracker 输入尺寸（runner mode，默认 640）。",
    )

    # ── SORT runner 参数 ───────────────────────────────────────────────────
    parser.add_argument(
        "--sort_exp_file",
        type=str,
        default=None,
        help=(
            "SORT exp 配置文件路径（runner mode 必须提供）。\n"
            "对应 tools/track_sort.py 的 -f 参数。"
        ),
    )
    parser.add_argument(
        "--sort_ckpt",
        type=str,
        default=None,
        help=(
            "SORT 检测模型 checkpoint 路径（runner mode 必须提供）。\n"
            "对应 tools/track_sort.py 的 -c 参数。"
        ),
    )

    # ── DeepSORT runner 参数 ───────────────────────────────────────────────
    parser.add_argument(
        "--deepsort_exp_file",
        type=str,
        default=None,
        help=(
            "DeepSORT exp 配置文件路径（runner mode 必须提供）。\n"
            "对应 tools/track_deepsort.py 的 -f 参数。"
        ),
    )
    parser.add_argument(
        "--deepsort_ckpt",
        type=str,
        default=None,
        help=(
            "DeepSORT 检测模型 checkpoint 路径（runner mode 必须提供）。\n"
            "对应 tools/track_deepsort.py 的 -c 参数。"
        ),
    )
    parser.add_argument(
        "--deepsort_model_folder",
        type=str,
        default=None,
        help=(
            "DeepSORT ReID 特征模型目录（runner mode 必须提供）。\n"
            "对应 tools/track_deepsort.py 的 --model_folder 参数。"
        ),
    )

    # ── ByteTrack runner 参数 ──────────────────────────────────────────────
    parser.add_argument(
        "--bytetrack_exp_file",
        type=str,
        default=None,
        help=(
            "ByteTrack exp 配置文件路径（runner mode 必须提供）。\n"
            "对应 tools/track.py 的 -f 参数。"
        ),
    )
    parser.add_argument(
        "--bytetrack_ckpt",
        type=str,
        default=None,
        help=(
            "ByteTrack 检测模型 checkpoint 路径（runner mode 必须提供）。\n"
            "对应 tools/track.py 的 -c 参数。"
        ),
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# GT 统一加载
# ══════════════════════════════════════════════════════════════════════════════

def _load_gt_once() -> dict:
    return load_gt(
        gt_path=GT_PATH,
        gt_format=GT_FORMAT,
        video_id=GT_VIDEO_ID if GT_VIDEO_ID is not None else "",
        modality=GT_MODALITY if GT_MODALITY is not None else "nir",
        exclude_outside=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 统一 run + eval
# ══════════════════════════════════════════════════════════════════════════════

def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace(":", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(",", "")
        .replace("=", "")
        .replace("*", "")
        .replace(".", "p")
        .lower()
    )


def _run_eval(
    name: str,
    run_fn,
    gt_data: dict,
    output_root: str,
    scenes: tuple[str, ...] = ("ALL",),
) -> tuple[dict[str, dict], float]:
    out_file = os.path.join(output_root, f"{_safe_name(name)}.txt")
    print(f"\n▶ Running [{name}] ...")
    _, elapsed = timed_run(run_fn, RGB_FILE, NIR_FILE, out_file)

    pred_data = load_mot_txt(out_file)
    all_frames = sorted(set(gt_data.keys()) | set(pred_data.keys()))

    results = {}
    for scene in scenes:
        subset = get_scene_frames(all_frames, scene)
        results[scene] = evaluate_mot(gt_data, pred_data, subset)

    return results, elapsed


def _fps(n_frames: int, elapsed: float) -> float:
    return round(n_frames / elapsed, 1) if elapsed > 0 else 0.0


def _balanced_default_kwargs() -> dict:
    """
    当前终版默认：
      - balanced_mode=True
      - use_vfr_alignment=False  → Fixed-FPS default
    """
    return {
        "balanced_mode": True,
        "use_vfr_alignment": False,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: Main Results
# ══════════════════════════════════════════════════════════════════════════════

def run_table1(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 1: Main Results (ALL scenes)")
    print("=" * 72)

    t1_cols = [
        "HOTA", "AssA", "DetA", "MOTA", "IDF1",
        "FP", "FN", "IDSW", "Frag", "Association_FPS"
    ]

    methods = [
        ("RGB-Only", lambda r, n, o: run_baseline_rgb_only(r, n, o)),
        ("NIR-Only", lambda r, n, o: run_baseline_nir_only(r, n, o)),
        ("Single-threshold", lambda r, n, o: run_single_threshold(r, n, o)),
        ("Dual-threshold", lambda r, n, o: run_dual_threshold(r, n, o)),
        ("Motion-Delayed-Conf", lambda r, n, o: run_motion_delayed_conf(r, n, o)),
        (OURS_NAME, lambda r, n, o: lifecycle_fusion_final(r, n, o, **_balanced_default_kwargs())),
    ]

    rows = []
    for name, fn in methods:
        res, elapsed = _run_eval(name, fn, gt_data, output_root, scenes=("ALL",))
        metrics = dict(res["ALL"])
        n_frames = metrics.get("Frames", 1)
        metrics["Association_FPS"] = _fps(n_frames, elapsed)
        rows.append((name, metrics))

    print_metrics_table(rows, t1_cols, title="Table 1: Main Results", scene="ALL")
    save_results_csv(
        rows,
        t1_cols,
        os.path.join(tables_dir, "table1_main_results.csv"),
        scene="ALL",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: Ablation Study（终稿口径：Scene-aware + AQG 为核心贡献）
# ══════════════════════════════════════════════════════════════════════════════

def run_table2(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 2: Ablation Study — Core Components")
    print("  Core contributions: Scene-aware + AQG")
    print("  (TKCC / Delayed Birth excluded: no incremental gain in latest runs)")
    print("=" * 72)

    t2_cols = ["HOTA", "AssA", "DetA", "FP", "FN", "IDSW", "Frag"]

    ablations = [
        # B1: No Scene-Aware（贡献明显：40.85 vs 41.70）
        ("B1: No Scene-Aware",
         {"ablate_no_scene_aware": True, **_balanced_default_kwargs()}),
        # B2: No AQG（贡献明显：38.83 vs 41.70）
        ("B2: No AQG",
         {"ablate_no_aqg": True, **_balanced_default_kwargs()}),
        # B3: No RGB Rescue（auxiliary heuristic，仅 -0.05；保留供评审参考）
        ("B3: No RGB Rescue (auxiliary)",
         {"ablate_no_rgb_rescue": True, **_balanced_default_kwargs()}),
        # Full Ours
        (OURS_NAME, _balanced_default_kwargs()),
    ]

    rows = []
    for name, kwargs in ablations:
        def _fn(r, n, o, kw=kwargs):
            lifecycle_fusion_final(r, n, o, **kw)

        res, _ = _run_eval(name, _fn, gt_data, output_root, scenes=("ALL",))
        rows.append((name, res["ALL"]))

    print_metrics_table(rows, t2_cols, title="Table 2: Ablation Study — Core Components", scene="ALL")
    save_results_csv(
        rows,
        t2_cols,
        os.path.join(tables_dir, "table2_ablation.csv"),
        scene="ALL",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 3A: AQG Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def run_table3a(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 3A: AQG Sensitivity  (* = Paper Final Default)")
    print("=" * 72)

    t3a_cols = ["HOTA", "MOTA", "IDF1", "FP", "FN"]

    configs = [
        ("AQG Default* (lo=0.45, hi=0.75)",
         {"aqg_low_override": 0.45, "aqg_high_override": 0.75, **_balanced_default_kwargs()}),
        ("AQG Old (lo=0.35, hi=0.65)",
         {"aqg_low_override": 0.35, "aqg_high_override": 0.65, **_balanced_default_kwargs()}),
        ("AQG Strict (lo=0.55, hi=0.85)",
         {"aqg_low_override": 0.55, "aqg_high_override": 0.85, **_balanced_default_kwargs()}),
        ("AQG Loose (lo=0.30, hi=0.60)",
         {"aqg_low_override": 0.30, "aqg_high_override": 0.60, **_balanced_default_kwargs()}),
    ]

    rows = []
    for name, kwargs in configs:
        def _fn(r, n, o, kw=kwargs):
            lifecycle_fusion_final(r, n, o, **kw)

        res, _ = _run_eval(name, _fn, gt_data, output_root, scenes=("ALL",))
        rows.append((name, res["ALL"]))

    print_metrics_table(rows, t3a_cols, title="Table 3A: AQG Sensitivity", scene="ALL")
    save_results_csv(
        rows,
        t3a_cols,
        os.path.join(tables_dir, "table3a_sensitivity_aqg.csv"),
        scene="ALL",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 3B: Rescue α Sensitivity（supplementary，不属于终版主资产）
# 需 --run_optional_rescue_sensitivity 显式启用。
# RGB rescue 为 optional auxiliary heuristic，此分析用于 supplementary 参考。
# ══════════════════════════════════════════════════════════════════════════════

def run_table3b(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 3B: Rescue α Sensitivity  [SUPPLEMENTARY]")
    print("  (RGB rescue is an optional auxiliary heuristic;")
    print("   this table is supplementary, not a core contribution analysis)")
    print("=" * 72)

    t3b_cols = ["HOTA", "MOTA", "IDF1", "FP", "FN"]

    configs = [
        ("Rescue α=0.20", {"rescue_alpha_override": 0.20, **_balanced_default_kwargs()}),
        ("Rescue α=0.25", {"rescue_alpha_override": 0.25, **_balanced_default_kwargs()}),
        ("Rescue α=0.30", {"rescue_alpha_override": 0.30, **_balanced_default_kwargs()}),
        ("Rescue α=0.35", {"rescue_alpha_override": 0.35, **_balanced_default_kwargs()}),
    ]

    rows = []
    for name, kwargs in configs:
        def _fn(r, n, o, kw=kwargs):
            lifecycle_fusion_final(r, n, o, **kw)

        res, _ = _run_eval(name, _fn, gt_data, output_root, scenes=("ALL",))
        rows.append((name, res["ALL"]))

    print_metrics_table(rows, t3b_cols, title="Table 3B: Rescue α Sensitivity [Supplementary]", scene="ALL")
    save_results_csv(
        rows,
        t3b_cols,
        os.path.join(tables_dir, "table3b_sensitivity_rescue.csv"),
        scene="ALL",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: Scenario Breakdown
# ══════════════════════════════════════════════════════════════════════════════

def run_table4(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 4: Scenario Breakdown (FOG / GLARE)")
    print("=" * 72)

    t4_cols = ["HOTA", "AssA", "FP", "FN", "IDSW", "Frag"]

    methods = [
        ("Single-threshold", lambda r, n, o: run_single_threshold(r, n, o)),
        ("Dual-threshold", lambda r, n, o: run_dual_threshold(r, n, o)),
        (OURS_NAME, lambda r, n, o: lifecycle_fusion_final(r, n, o, **_balanced_default_kwargs())),
    ]

    for scene in ("FOG", "GLARE"):
        rows = []
        for name, fn in methods:
            res, _ = _run_eval(name, fn, gt_data, output_root, scenes=(scene,))
            rows.append((name, res[scene]))

        print_metrics_table(
            rows,
            t4_cols,
            title=f"Table 4: Scenario Breakdown [{scene}]",
            scene=scene,
        )
        save_results_csv(
            rows,
            t4_cols,
            os.path.join(tables_dir, f"table4_scenario_{scene.lower()}.csv"),
            scene=scene,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Table 5: Temporal Scaling Sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def run_table5(gt_data: dict, output_root: str, tables_dir: str) -> None:
    print("\n" + "=" * 72)
    print("TABLE 5: Temporal Scaling Sensitivity")
    print("  Fixed-FPS Default (Ours) vs Synthetic Variable-Scale")
    print("=" * 72)

    t5_cols = ["HOTA", "AssA", "MOTA", "IDSW", "Frag", "FP", "FN"]

    methods = [
        ("Fixed-FPS Default",
         lambda r, n, o: lifecycle_fusion_final(
             r, n, o,
             balanced_mode=True,
             use_vfr_alignment=False,
         )),
        ("Synthetic Variable-Scale",
         lambda r, n, o: lifecycle_fusion_final(
             r, n, o,
             balanced_mode=True,
             use_vfr_alignment=True,
         )),
    ]

    rows = []
    for name, fn in methods:
        res, _ = _run_eval(name, fn, gt_data, output_root, scenes=("ALL",))
        rows.append((name, res["ALL"]))

    print_metrics_table(
        rows,
        t5_cols,
        title="Table 5: Temporal Scaling Sensitivity",
        scene="ALL",
    )
    save_results_csv(
        rows,
        t5_cols,
        os.path.join(tables_dir, "table5_temporal_scaling.csv"),
        scene="ALL",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Table A: External Tracker Comparison（附加资产，独立于 Table 1–5）
# ══════════════════════════════════════════════════════════════════════════════

def run_tableA_external_trackers(
    gt_data:              dict,
    output_root:          str,
    tables_dir:           str,
    bytetrack_raw_txt:    str  = DEFAULT_BYTETRACK_RAW,
    botsort_raw_txt:      str  = DEFAULT_BOTSORT_RAW,
    mode:                 str  = "adapter",
    # ── runner mode 专用参数（adapter mode 忽略这些参数）─────────────────
    bytetrack_repo_root:  str  = DEFAULT_BYTETRACK_REPO_ROOT,
    tracker_video_path:   str  = None,
    sort_exp_file:        str  = None,
    sort_ckpt:            str  = None,
    deepsort_exp_file:    str  = None,
    deepsort_ckpt:        str  = None,
    deepsort_model_folder:str  = None,
    bytetrack_exp_file:   str  = None,
    bytetrack_ckpt:       str  = None,
    tracker_conf:         float = 0.6,
    tracker_nms:          float = 0.7,
    tracker_tsize:        int   = 640,
) -> None:
    """
    TABLE A: External Tracker Comparison（NIR-source 口径，精简版）

    对比方法（NIR-source）：
      ByteTrack-NIR / BoT-SORT-NIR / GAT-DI (Scene-aware + AQG) [Ours]

    ──────────────────────────────────────────────────────────────
    公平比较原则（Fair Comparison Protocol）：
      - 外部 tracker 对比均基于 NIR 图像序列产出 raw txt（NIR-source）。
      - 与 GAT-DI 的 NIR-anchor 设定保持一致，确保输入模态对齐。
      - Adapter mode is used to fairly evaluate external trackers
        under the same evaluation pipeline as the main method.
    ──────────────────────────────────────────────────────────────

    此函数不影响 Table 1–5 的任何结果或 CSV。
    """
    print("\n" + "=" * 72)
    print("TABLE A: External Tracker Comparison  [NIR-source]")
    print(f"  Mode     : {mode}")
    print(f"  Methods  : ByteTrack-NIR / BoT-SORT-NIR vs {OURS_NAME}")
    print(f"  NIR seq  : E:\\mokuai\\track_val\\peizhun\\nir_degraded  (NIR-source)")
    if mode == "runner":
        print(f"  Repo     : {bytetrack_repo_root}")
        print(f"  Video    : {tracker_video_path}")
    print("=" * 72)

    for d in (EXT_RAW_DIR, EXT_NORM_DIR, EXT_EVAL_DIR):
        os.makedirs(d, exist_ok=True)

    tA_cols = ["HOTA", "AssA", "DetA", "MOTA", "IDF1", "FP", "FN", "IDSW", "Frag"]

    # ── 根据 mode 构建 ext_methods ─────────────────────────────────────────
    if mode == "runner":
        ext_methods = [
            ("ByteTrack-NIR",
             lambda r, n, o: run_external_bytetrack_runner(
                 r, n, o,
                 bytetrack_repo_root = bytetrack_repo_root,
                 video_path          = tracker_video_path or "",
                 exp_file            = bytetrack_exp_file,
                 ckpt                = bytetrack_ckpt,
                 conf                = tracker_conf,
                 nms                 = tracker_nms,
                 tsize               = tracker_tsize,
                 raw_txt             = bytetrack_raw_txt,
                 norm_dir            = EXT_NORM_DIR,
                 eval_dir            = EXT_EVAL_DIR,
             )),
            # Ours：与 Table 1 完全一致，保证对比公平
            ("BoT-SORT-NIR",
             lambda r, n, o, _raw=botsort_raw_txt:
                 run_baseline_botsort(r, n, o, raw_txt=_raw,
                                      norm_dir=EXT_NORM_DIR, eval_dir=EXT_EVAL_DIR)),
            (OURS_NAME,
             lambda r, n, o: lifecycle_fusion_final(r, n, o, **_balanced_default_kwargs())),
        ]
    else:
        # adapter mode（默认）—— NIR-source 口径，精简为 ByteTrack + BoT-SORT
        ext_methods = [
            ("ByteTrack-NIR",
             lambda r, n, o, _raw=bytetrack_raw_txt:
                 run_baseline_bytetrack(r, n, o, raw_txt=_raw,
                                        norm_dir=EXT_NORM_DIR, eval_dir=EXT_EVAL_DIR)),
            ("BoT-SORT-NIR",
             lambda r, n, o, _raw=botsort_raw_txt:
                 run_baseline_botsort(r, n, o, raw_txt=_raw,
                                      norm_dir=EXT_NORM_DIR, eval_dir=EXT_EVAL_DIR)),
            # Ours：与 Table 1 完全一致，保证对比公平
            (OURS_NAME,
             lambda r, n, o: lifecycle_fusion_final(r, n, o, **_balanced_default_kwargs())),
        ]

    # ── 逐方法运行 + 评估 ──────────────────────────────────────────────────
    rows: list[tuple[str, dict]] = []
    missing_trackers: list[str]  = []

    for name, fn in ext_methods:
        try:
            res, _ = _run_eval(name, fn, gt_data, output_root, scenes=("ALL",))
            m = res["ALL"]
            if m.get("Frames", 0) == 0 and name != OURS_NAME:
                missing_trackers.append(name)
                print(
                    f"  ⚠️  [{name}] 结果为空（raw txt 可能不存在），"
                    f"以全 0 指标纳入表格。"
                )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            print(f"\n  ❌ [{name}] 运行失败：")
            for line in str(exc).splitlines():
                print(f"    {line}")
            missing_trackers.append(name)
            m = {c: 0 for c in tA_cols}
            print(f"  [{name}] 以全 0 指标占位纳入表格。")

        rows.append((name, m))

    print_metrics_table(rows, tA_cols,
                        title="Table A: External Tracker Comparison",
                        scene="ALL")

    csv_path = os.path.join(tables_dir, "tableA_external_tracker_comparison.csv")
    save_results_csv(rows, tA_cols, csv_path, scene="ALL")

    # ── metadata ──────────────────────────────────────────────────────────
    _write_external_meta(
        output_root         = output_root,
        bytetrack_raw       = bytetrack_raw_txt,
        botsort_raw         = botsort_raw_txt,
        mode                = mode,
        missing_trackers    = missing_trackers,
        csv_path            = csv_path,
        bytetrack_repo_root = bytetrack_repo_root,
        tracker_video_path  = tracker_video_path,
    )

    if missing_trackers:
        if mode == "adapter":
            print(f"\n[Table A] ⚠️  以下 tracker 缺失，请参考下方说明放置结果后重跑：")
            _print_raw_txt_guide(bytetrack_raw_txt, botsort_raw_txt)
        else:
            print(
                f"\n[Table A] ⚠️  以下 tracker 运行失败：{missing_trackers}\n"
                f"  请检查上方错误信息，补全缺失配置后重新运行。\n"
                f"  使用 --help 查看 runner mode 所需参数。"
            )


def _print_raw_txt_guide(bytetrack_raw, botsort_raw):
    print()
    print("  ─────────────────────────────────────────────────────────")
    print("  External Tracker NIR-source Raw TXT 放置说明（adapter mode）：")
    print()
    print("  【当前口径】外部对比精简为 ByteTrack-NIR + BoT-SORT-NIR，")
    print("  结果均须基于 NIR 图像序列 (nir_degraded) 生成。")
    print()
    print("  ── 方式一：使用辅助脚本自动生成（推荐）──────────────────")
    print("    python generate_nir_external_raw_txts.py")
    print("    （自动生成 bytetrack_nir_raw.txt / botsort_nir_raw.txt）")
    print()
    print("  ── 方式二：手动放置 raw txt ──────────────────────────────")
    print("  1. 基于 NIR 图像序列运行 tracker，保存为标准 MOT17 格式")
    print("     （frame,id,x,y,w,h,score,...）。")
    print()
    print("  2. 分别放置到以下路径：")
    print(f"       ByteTrack-NIR : {bytetrack_raw}")
    print(f"       BoT-SORT-NIR  : {botsort_raw}")
    print()
    print("  3. 重新运行:")
    print("       python run_all_experiments.py --run_external_trackers")
    print("  ─────────────────────────────────────────────────────────")


def _write_external_meta(
    output_root, bytetrack_raw, botsort_raw,
    mode, missing_trackers, csv_path,
    bytetrack_repo_root=None, tracker_video_path=None,
):
    meta = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "script":      "run_all_experiments.py",
        "table":       "Table A: External Tracker Comparison (NIR-source)",
        "methods":     ["ByteTrack-NIR", "BoT-SORT-NIR", "GAT-DI (Scene-aware + AQG)"],
        "nir_source_note": (
            "All external tracker raw txt files are generated from NIR image "
            "sequence (nir_degraded), consistent with GAT-DI NIR-anchor setting."
        ),
        "nir_image_seq": r"E:\mokuai\track_val\peizhun\nir_degraded",
        "nir_weights_dir": (
            r"E:\module\ultralytics-main\ultralytics-main\runs\detect\runs"
            r"\nir_experiments\yolov11n_nir_baseline3\weights"
        ),
        "mode":        mode,
        "ours_name":           OURS_NAME,
        "ours_default_kwargs": _balanced_default_kwargs(),
        "tracker_raw_paths":   {
            "bytetrack": bytetrack_raw,
            "botsort":   botsort_raw,
        },
        "tracker_norm_dir":    EXT_NORM_DIR,
        "tracker_eval_dir":    EXT_EVAL_DIR,
        "missing_trackers":    missing_trackers,
        "output_csv":          csv_path,
    }
    if mode == "runner":
        meta["runner_config"] = {
            "bytetrack_repo_root": bytetrack_repo_root,
            "tracker_video_path":  tracker_video_path,
        }
    meta_dir  = os.path.join(output_root, "external_trackers")
    meta_path = os.path.join(meta_dir, "meta_external_tracker_comparison.json")
    os.makedirs(meta_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  → 外部对比 metadata: {meta_path}")


# ══════════════════════════════════════════════════════════════════════════════
# metadata / housekeeping
# ══════════════════════════════════════════════════════════════════════════════

def write_meta_json(
    output_root: str,
    tables_dir: str,
    seed: int,
    ran_rescue_sensitivity: bool = False,
) -> str:
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": "run_all_experiments.py",
        "paper_version": "Paper Final Convergence",
        "ours_name": OURS_NAME,
        "core_contributions": ["Scene-aware", "AQG"],
        "rgb_rescue_note": (
            "RGB rescue is retained as an optional auxiliary heuristic "
            "for difficult glare cases; not a core claimed contribution."
        ),
        "default_aqg": {"low": 0.45, "high": 0.75},
        "default_temporal_handling": "Fixed-FPS",
        "default_kwargs": _balanced_default_kwargs(),
        "seed": seed,
        "rgb_file": RGB_FILE,
        "nir_file": NIR_FILE,
        "gt_path": GT_PATH,
        "gt_format": GT_FORMAT,
        "output_root": output_root,
        "tables_dir": tables_dir,
        "final_csv_assets": FINAL_CSVS,
        "supplementary_csv_assets": SUPPLEMENTARY_CSVS if ran_rescue_sensitivity else [],
        "external_csv_assets": EXTERNAL_CSVS,
        "obsolete_csv_assets": OBSOLETE_CSVS,
    }

    meta_path = os.path.join(output_root, "meta_run_all_experiments.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta_path


def clean_obsolete_csvs(tables_dir: str) -> None:
    removed = []
    for fname in OBSOLETE_CSVS:
        fpath = os.path.join(tables_dir, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            removed.append(fname)

    if removed:
        print("\n[clean] 已删除 obsolete CSV：")
        for fname in removed:
            print(f"  - {fname}")
    else:
        print("\n[clean] 未发现需要删除的 obsolete CSV。")


def print_final_asset_summary(
    tables_dir: str,
    ran_external: bool = False,
    ran_rescue_sensitivity: bool = False,
) -> None:
    print("\n✅ 全部表格生成完毕。")
    print(f"   CSV 保存目录: {tables_dir}")

    print("\n  终版资产 CSV（核心贡献：Scene-aware + AQG）：")
    for fname in FINAL_CSVS:
        tag = "✓" if os.path.exists(os.path.join(tables_dir, fname)) else "✗"
        print(f"    {tag} {fname}")

    if ran_rescue_sensitivity:
        print("\n  Supplementary CSV（Table 3B，RGB rescue auxiliary analysis）：")
        for fname in SUPPLEMENTARY_CSVS:
            tag = "✓" if os.path.exists(os.path.join(tables_dir, fname)) else "✗"
            print(f"    {tag} {fname}")

    if ran_external:
        print("\n  附加资产 CSV（Table A）：")
        for fname in EXTERNAL_CSVS:
            tag = "✓" if os.path.exists(os.path.join(tables_dir, fname)) else "✗"
            print(f"    {tag} {fname}")

    existing_obsolete = [
        fname for fname in OBSOLETE_CSVS
        if os.path.exists(os.path.join(tables_dir, fname))
    ]
    if existing_obsolete:
        print("\n  以下 CSV 属于旧版本遗留（obsolete），不纳入终版资产：")
        for fname in existing_obsolete:
            print(f"    - {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    output_root = args.out_dir
    tables_dir = os.path.join(output_root, "tables")
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if args.clean_obsolete:
        clean_obsolete_csvs(tables_dir)

    print("=" * 72)
    print("  GAT-DI v3-final  全量实验（Paper Final Convergence）")
    print(f"  Ours method    : {OURS_NAME}")
    print("  Default AQG    : 0.45 / 0.75")
    print("  Default time   : Fixed-FPS")
    print("  Core modules   : Scene-aware + AQG")
    print("  RGB rescue     : optional auxiliary heuristic (not core contribution)")
    print(f"  RGB            : {RGB_FILE}")
    print(f"  NIR            : {NIR_FILE}")
    print(f"  GT             : {GT_PATH}  [{GT_FORMAT}]")
    print(f"  Output root    : {output_root}")
    print(f"  Tables dir     : {tables_dir}")
    print("=" * 72)

    gt_data = _load_gt_once()
    if not gt_data:
        print("[error] GT 数据为空，请检查 GT_PATH / GT_FORMAT 配置。")
        return

    # ── 主实验 Table 1–5（不受附加参数影响）──────────────────────────────
    run_table1(gt_data, output_root, tables_dir)
    run_table2(gt_data, output_root, tables_dir)
    run_table3a(gt_data, output_root, tables_dir)
    run_table4(gt_data, output_root, tables_dir)
    run_table5(gt_data, output_root, tables_dir)

    # ── Table 3B（supplementary，需 --run_optional_rescue_sensitivity）────
    if args.run_optional_rescue_sensitivity:
        run_table3b(gt_data, output_root, tables_dir)

    # ── Table A（外部 tracker 对比，仅 --run_external_trackers 时运行）──
    if args.run_external_trackers:
        run_tableA_external_trackers(
            gt_data               = gt_data,
            output_root           = output_root,
            tables_dir            = tables_dir,
            bytetrack_raw_txt     = args.bytetrack_raw_txt,
            botsort_raw_txt       = args.botsort_raw_txt,
            mode                  = args.external_tracker_mode,
            bytetrack_repo_root   = args.bytetrack_repo_root,
            tracker_video_path    = args.tracker_video_path,
            sort_exp_file         = args.sort_exp_file,
            sort_ckpt             = args.sort_ckpt,
            deepsort_exp_file     = args.deepsort_exp_file,
            deepsort_ckpt         = args.deepsort_ckpt,
            deepsort_model_folder = args.deepsort_model_folder,
            bytetrack_exp_file    = args.bytetrack_exp_file,
            bytetrack_ckpt        = args.bytetrack_ckpt,
            tracker_conf          = args.tracker_conf,
            tracker_nms           = args.tracker_nms,
            tracker_tsize         = args.tracker_tsize,
        )

    meta_path = write_meta_json(
        output_root, tables_dir, args.seed,
        ran_rescue_sensitivity=args.run_optional_rescue_sensitivity,
    )
    print(f"\n  已写入 metadata: {meta_path}")

    print_final_asset_summary(
        tables_dir,
        ran_external=args.run_external_trackers,
        ran_rescue_sensitivity=args.run_optional_rescue_sensitivity,
    )


if __name__ == "__main__":
    main()