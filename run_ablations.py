"""
run_ablations.py
────────────────────────────────────────────────────────────────────────────────
独立消融实验脚本（终稿精简版）。

功能：
  - 单独运行 Table 2 的消融组合（聚焦核心贡献：Scene-aware + AQG）
  - 支持指定单个消融或全量运行
  - 结果打印到终端 + 保存 CSV

使用方式：
    # 全量运行（默认消融组合）
    python run_ablations.py

    # 只跑某一组
    python run_ablations.py --exp E2

消融开关通过 lifecycle_fusion_final(...) 的关键字参数控制，
不修改 gat_di_v3_final.py 本体逻辑。

消融方案说明（终稿口径）：
  E1: No Scene-Aware    → 核心贡献，贡献明显（40.85 vs 41.70）
  E2: No AQG            → 核心贡献，贡献明显（38.83 vs 41.70）
  E3: No RGB Rescue     → optional auxiliary heuristic（41.65 vs 41.70，差异 0.05）
  FULL: Ours            → 完整方法

已从默认组合中移除的项（不再作为核心消融）：
  ✗ No Delayed Birth  → B1 = 41.72 > Ours 41.70，无正向贡献，不再作为核心设计
  ✗ No TKCC           → B5 = 41.70 = Ours，无增量贡献，不再作为核心模块
  ✗ No TKCC + No RGB Rescue → 组合消融，贡献归因不清，已移除
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gat_di_v3_final import lifecycle_fusion_final
from eval_utils import (
    load_mot_txt, load_gt_csv,
    get_scene_frames, evaluate_mot,
    print_metrics_table, save_results_csv,
)

# ──────────────────────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────────────────────
RGB_FILE    = r"E:\mokuai\video\peizhun\data\pretracks\rgb\first_tracks.txt"
NIR_FILE    = r"E:\mokuai\video\peizhun\data\pretracks\nir\first_tracks.txt"
GT_CSV      = r"E:\mokuai\video\peizhun\pig_dataset_csv\gt_annotations_merged.csv"
VIDEO_ID    = "first"
OUTPUT_ROOT = r"E:\mokuai\video\peizhun\data\ablations"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

ABLATION_COLS = ["HOTA", "AssA", "DetA", "FP", "FN", "IDSW", "Frag"]

# ──────────────────────────────────────────────────────────────────────────────
# 消融实验定义（终稿精简版）
#
# 核心消融：Scene-aware（E1）+ AQG（E2）
# 可选消融：RGB rescue（E3，auxiliary heuristic，保留供评审参考）
# 完整方法：FULL
#
# 已移除：
#   No Delayed Birth → 无正向贡献（结果略高于 Ours），不再作为核心设计
#   No TKCC          → 无增量贡献（结果等于 Ours），不再作为核心模块
#   No TKCC + No RGB Rescue → 组合消融意义不大，已移除
# ──────────────────────────────────────────────────────────────────────────────
ABLATION_DEFS: dict[str, tuple[str, dict]] = {
    "E1": (
        "E1: No Scene-Aware",
        # ablate_no_scene_aware=True → 所有帧强制 CLEAR，无场景自适应
        # 核心贡献之一，贡献明显（HOTA 40.85 vs Full 41.70）
        {"ablate_no_scene_aware": True},
    ),
    "E2": (
        "E2: No AQG",
        # ablate_no_aqg=True → AQG 阈值固定为基础值，不随 avg_conf 动态变化
        # 核心贡献之一，贡献明显（HOTA 38.83 vs Full 41.70）
        {"ablate_no_aqg": True},
    ),
    "E3": (
        "E3: No RGB Rescue (auxiliary)",
        # ablate_no_rgb_rescue=True → GLARE 场景下不启用 RGB 补救
        # RGB rescue 为 optional auxiliary heuristic，
        # 贡献不显著（HOTA 41.65 vs Full 41.70，差异 0.05）
        {"ablate_no_rgb_rescue": True},
    ),
    "FULL": (
        "GAT-DI (Scene-aware + AQG) [Ours]",
        # 所有模块全开，对照组
        {},
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# 已移除的消融项（保留注释，便于追溯）
# ──────────────────────────────────────────────────────────────────────────────
# REMOVED_ABLATIONS = {
#     "NO_DELAYED_BIRTH": (
#         "No Delayed Birth",
#         {"ablate_no_delayed_birth": True},
#         # 移除原因：最新完整重跑 B1 = 41.72 > Ours 41.70
#         # Delayed Birth 无正向贡献，不再作为核心设计。
#     ),
#     "NO_TKCC": (
#         "No TKCC",
#         {"ablate_no_tkcc": True},
#         # 移除原因：最新完整重跑 B5 = 41.70 = Ours
#         # TKCC 无增量贡献，不再作为核心模块。
#     ),
#     "NO_TKCC_NO_RESCUE": (
#         "No TKCC + No RGB Rescue",
#         {"ablate_no_tkcc": True, "ablate_no_rgb_rescue": True},
#         # 移除原因：组合消融，无独立贡献归因意义，已移除。
#     ),
# }


def _balanced_default_kwargs() -> dict:
    return {"balanced_mode": True, "use_vfr_alignment": False}


def run_one_ablation(
    exp_id:    str,
    exp_name:  str,
    kwargs:    dict,
    scenes:    list[str] = ("ALL", "FOG", "GLARE"),
) -> dict[str, dict]:
    """运行单个消融实验，返回 {scene: metrics}。"""
    safe_name = exp_id.lower().replace(" ", "_").replace(":", "")
    out_file  = os.path.join(OUTPUT_ROOT, f"ablation_{safe_name}.txt")

    # 合并 balanced_mode 默认设置
    full_kwargs = {**_balanced_default_kwargs(), **kwargs}

    print(f"\n  ▶ [{exp_id}] {exp_name}")
    lifecycle_fusion_final(RGB_FILE, NIR_FILE, out_file, **full_kwargs)

    pred      = load_mot_txt(out_file)
    gt        = load_gt_csv(GT_CSV, VIDEO_ID)
    all_frames = sorted(set(gt.keys()) | set(pred.keys()))

    results: dict[str, dict] = {}
    for scene in scenes:
        subset = get_scene_frames(all_frames, scene)
        results[scene] = evaluate_mot(gt, pred, subset)
    return results


def run_all_ablations(target_exp: str | None = None):
    """全量或单个消融实验。"""
    if target_exp:
        if target_exp not in ABLATION_DEFS:
            print(f"[error] 未知实验 ID: {target_exp}")
            print(f"        可用: {list(ABLATION_DEFS.keys())}")
            sys.exit(1)
        ids_to_run = [target_exp]
    else:
        ids_to_run = list(ABLATION_DEFS.keys())

    all_results: dict[str, dict[str, dict]] = {}
    for exp_id in ids_to_run:
        name, kwargs = ABLATION_DEFS[exp_id]
        all_results[exp_id] = run_one_ablation(exp_id, name, kwargs)

    # ── 打印汇总表（ALL 场景） ────────────────────────────────────────────
    rows_all = [(ABLATION_DEFS[eid][0], res["ALL"])
                for eid, res in all_results.items()]
    print_metrics_table(rows_all, ABLATION_COLS,
                        title="Ablation Study — Core Components (Scene-aware + AQG)",
                        scene="ALL")
    save_results_csv(rows_all, ABLATION_COLS,
                     os.path.join(OUTPUT_ROOT, "ablation_all.csv"), scene="ALL")

    # ── 打印场景切片表 ────────────────────────────────────────────────────
    for scene in ("FOG", "GLARE"):
        rows_s = [(ABLATION_DEFS[eid][0], res.get(scene, {}))
                  for eid, res in all_results.items()]
        print_metrics_table(rows_s, ABLATION_COLS,
                            title=f"Ablation Study — {scene} scene",
                            scene=scene)
        save_results_csv(rows_s, ABLATION_COLS,
                         os.path.join(OUTPUT_ROOT, f"ablation_{scene.lower()}.csv"),
                         scene=scene)

    print(f"\n✅ 消融实验完成。CSV 保存目录: {OUTPUT_ROOT}")


def main():
    parser = argparse.ArgumentParser(description="GAT-DI 消融实验（终稿精简版）")
    parser.add_argument("--exp", type=str, default=None,
                        help="指定单个实验 ID（E1 / E2 / E3 / FULL）。不指定则全量运行。")
    args = parser.parse_args()
    run_all_ablations(args.exp)


if __name__ == "__main__":
    main()