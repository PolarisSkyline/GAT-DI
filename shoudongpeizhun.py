"""
手动标点配准工具 (并排显示版)
RGB和IR并排显示，同时标记对应点
修改：使用白边填充而不是复制像素
"""

import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


# 全局变量
points_rgb = []
points_ir = []
current_side = 'rgb'  # 'rgb' or 'ir'
click_enabled = True


def imread_chinese(filepath):
    try:
        img_array = np.fromfile(str(filepath), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except:
        return None


def imwrite_chinese(filepath, img):
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        is_success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if is_success:
            buffer.tofile(str(filepath))
            return True
        return False
    except:
        return False


def mouse_callback(event, x, y, flags, param):
    """鼠标回调 - 并排显示版"""
    global points_rgb, points_ir, current_side, click_enabled

    rgb_img, ir_img, rgb_display, ir_display, combined_display = param

    h, w = rgb_img.shape[:2]

    if not click_enabled:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # 判断点击在哪一侧
        if x < w:
            # 点击在RGB侧
            side = 'rgb'
            actual_x = x
            points_rgb.append((actual_x, y))
            color = (0, 255, 0)
            label = f"{len(points_rgb)}"

            # 在RGB侧绘制
            cv2.circle(rgb_display, (actual_x, y), 10, color, 2)
            cv2.circle(rgb_display, (actual_x, y), 3, color, -1)
            cv2.putText(rgb_display, label, (actual_x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"  ✓ RGB点 {len(points_rgb)}: ({actual_x}, {y})")

        else:
            # 点击在IR侧
            side = 'ir'
            actual_x = x - w
            points_ir.append((actual_x, y))
            color = (0, 0, 255)
            label = f"{len(points_ir)}"

            # 在IR侧绘制
            cv2.circle(ir_display, (actual_x, y), 10, color, 2)
            cv2.circle(ir_display, (actual_x, y), 3, color, -1)
            cv2.putText(ir_display, label, (actual_x + 15, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            print(f"  ✓ IR点 {len(points_ir)}: ({actual_x}, {y})")

        # 更新组合显示
        combined_display[:, :w] = rgb_display
        combined_display[:, w:] = ir_display

        # 添加状态信息
        status = f"RGB: {len(points_rgb)} points | IR: {len(points_ir)} points | Press 'u' to undo | 'q' to finish"
        cv2.putText(combined_display, status, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Manual Point Matching - RGB (Left) | IR (Right)", combined_display)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # 右键删除最后一个点
        if x < w and len(points_rgb) > 0:
            # 删除RGB点
            points_rgb.pop()
            print(f"  删除最后一个RGB点")
        elif x >= w and len(points_ir) > 0:
            # 删除IR点
            points_ir.pop()
            print(f"  删除最后一个IR点")

        # 重绘
        redraw_all(rgb_img, ir_img, rgb_display, ir_display, combined_display)


def redraw_all(rgb_img, ir_img, rgb_display, ir_display, combined_display):
    """重绘所有点"""
    h, w = rgb_img.shape[:2]

    # 重置
    rgb_display[:] = rgb_img.copy()
    ir_display[:] = ir_img.copy()

    # 重绘RGB点
    for i, (x, y) in enumerate(points_rgb):
        cv2.circle(rgb_display, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(rgb_display, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(rgb_display, str(i+1), (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 重绘IR点
    for i, (x, y) in enumerate(points_ir):
        cv2.circle(ir_display, (x, y), 10, (0, 0, 255), 2)
        cv2.circle(ir_display, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(ir_display, str(i+1), (x + 15, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 组合
    combined_display[:, :w] = rgb_display
    combined_display[:, w:] = ir_display

    # 状态信息
    status = f"RGB: {len(points_rgb)} points | IR: {len(points_ir)} points | Press 'u' to undo | 'q' to finish"
    cv2.putText(combined_display, status, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Manual Point Matching - RGB (Left) | IR (Right)", combined_display)


def collect_points_side_by_side(rgb_img, ir_img):
    """
    并排显示RGB和IR，同时标记对应点
    """

    global points_rgb, points_ir, click_enabled

    points_rgb = []
    points_ir = []
    click_enabled = True

    print("\n" + "=" * 80)
    print("交互式标点 - 并排显示模式")
    print("=" * 80)

    print("\n使用说明:")
    print("  1. RGB图像在左侧，IR图像在右侧")
    print("  2. 在RGB图像上点击标记特征点")
    print("  3. 在IR图像的对应位置标记相同特征点")
    print("  4. 点的序号必须对应（第1个RGB点对应第1个IR点）")
    print("  5. 至少需要4对对应点")
    print("\n操作:")
    print("  - 左键点击：在RGB或IR上添加点")
    print("  - 右键点击：删除对应侧的最后一个点")
    print("  - 按键'u'：撤销上一对点")
    print("  - 按键'r'：重新开始")
    print("  - 按键'q'：完成标记")

    print("\n标点建议:")
    print("  - 选择明显的特征点（耳朵尖、鼻子、眼角、关节等）")
    print("  - 尽量分布在整个图像区域")
    print("  - 建议标记 8-12 对点")

    # 调整图像大小
    h, w = rgb_img.shape[:2]
    max_width = 1920 // 2  # 每侧最大宽度
    max_height = 1080

    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_display_base = cv2.resize(rgb_img, (new_w, new_h))
        ir_display_base = cv2.resize(ir_img, (new_w, new_h))
        scale_factor = scale
        print(f"\n图像已缩放: {w}x{h} → {new_w}x{new_h} (缩放比例: {scale:.2f})")
    else:
        rgb_display_base = rgb_img.copy()
        ir_display_base = ir_img.copy()
        scale_factor = 1.0
        new_w, new_h = w, h

    # 确保RGB和IR尺寸一致
    if rgb_display_base.shape != ir_display_base.shape:
        ir_display_base = cv2.resize(ir_display_base,
                                     (rgb_display_base.shape[1], rgb_display_base.shape[0]))

    # 统一格式（都转为彩色）
    if len(ir_display_base.shape) == 2:
        ir_display_base = cv2.cvtColor(ir_display_base, cv2.COLOR_GRAY2BGR)

    h_display, w_display = rgb_display_base.shape[:2]

    # 创建工作副本
    rgb_display = rgb_display_base.copy()
    ir_display = ir_display_base.copy()

    # 创建组合显示（RGB | IR）
    combined_display = np.zeros((h_display, w_display * 2, 3), dtype=np.uint8)
    combined_display[:, :w_display] = rgb_display
    combined_display[:, w_display:] = ir_display

    # 添加分隔线
    cv2.line(combined_display, (w_display, 0), (w_display, h_display), (255, 255, 255), 3)

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_display, 'RGB (Click here)', (20, 40),
                font, 1.2, (0, 255, 0), 2)
    cv2.putText(combined_display, 'IR (Click here)', (w_display + 20, 40),
                font, 1.2, (0, 0, 255), 2)

    # 创建窗口
    window_name = "Manual Point Matching - RGB (Left) | IR (Right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback,
                        (rgb_display_base, ir_display_base,
                         rgb_display, ir_display, combined_display))
    cv2.imshow(window_name, combined_display)

    print("\n开始标记...")
    print("提示: 先在RGB左侧点击一个特征点，然后在IR右侧点击对应位置")

    # 主循环
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            # 完成
            if len(points_rgb) < 4 or len(points_ir) < 4:
                print(f"  ⚠ 至少需要4对点！当前: RGB={len(points_rgb)}, IR={len(points_ir)}")
                continue

            if len(points_rgb) != len(points_ir):
                print(f"  ⚠ RGB和IR点数必须相同！当前: RGB={len(points_rgb)}, IR={len(points_ir)}")
                continue

            print(f"\n✓ 标记完成: {len(points_rgb)} 对对应点")
            break

        elif key == ord('r'):
            # 重新开始
            points_rgb = []
            points_ir = []
            print("\n重新开始标记...")
            redraw_all(rgb_display_base, ir_display_base,
                      rgb_display, ir_display, combined_display)

        elif key == ord('u'):
            # 撤销最后一对
            if len(points_rgb) > 0 and len(points_ir) > 0:
                if len(points_rgb) == len(points_ir):
                    points_rgb.pop()
                    points_ir.pop()
                    print(f"  撤销最后一对点")
                elif len(points_rgb) > len(points_ir):
                    points_rgb.pop()
                    print(f"  撤销最后一个RGB点")
                else:
                    points_ir.pop()
                    print(f"  撤销最后一个IR点")

                redraw_all(rgb_display_base, ir_display_base,
                          rgb_display, ir_display, combined_display)

    cv2.destroyAllWindows()

    # 还原缩放
    if scale_factor != 1.0:
        points_rgb_scaled = [(x / scale_factor, y / scale_factor) for x, y in points_rgb]
        points_ir_scaled = [(x / scale_factor, y / scale_factor) for x, y in points_ir]
        return np.array(points_rgb_scaled, dtype=np.float32), np.array(points_ir_scaled, dtype=np.float32)
    else:
        return np.array(points_rgb, dtype=np.float32), np.array(points_ir, dtype=np.float32)


def compute_transformation_from_points(pts_src, pts_dst, method='affine'):
    """根据对应点计算变换矩阵"""

    print(f"\n计算变换矩阵 (方法: {method})...")

    if method == 'affine':
        M, inliers = cv2.estimateAffinePartial2D(
            pts_src, pts_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.99
        )

        if M is None:
            print("✗ 仿射变换计算失败")
            return None, None

        M_full = np.vstack([M, [0, 0, 1]])
        inlier_ratio = np.sum(inliers) / len(pts_src) if inliers is not None else 1.0

    elif method == 'homography':
        M_full, inliers = cv2.findHomography(
            pts_src, pts_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            confidence=0.99
        )

        if M_full is None:
            print("✗ 单应性变换计算失败")
            return None, None

        inlier_ratio = np.sum(inliers) / len(pts_src) if inliers is not None else 1.0

    else:
        print(f"✗ 未知方法: {method}")
        return None, None

    print(f"✓ 变换计算完成")
    print(f"  内点比例: {inlier_ratio:.2%}")
    print(f"  变换矩阵:\n{M_full}")

    # 计算重投影误差
    pts_dst_pred = cv2.perspectiveTransform(pts_src.reshape(-1, 1, 2), M_full)
    errors = np.linalg.norm(pts_dst.reshape(-1, 1, 2) - pts_dst_pred, axis=2).flatten()

    mean_error = np.mean(errors)
    max_error = np.max(errors)

    print(f"  平均误差: {mean_error:.2f} px")
    print(f"  最大误差: {max_error:.2f} px")

    if mean_error > 10:
        print(f"  ⚠ 误差较大，建议重新标记更精确的点")
    elif mean_error > 5:
        print(f"  ✓ 误差中等，配准质量一般")
    else:
        print(f"  ✓ 误差很小，配准质量优秀")

    return M_full, inlier_ratio


def apply_transformation(rgb_img, transform_matrix):
    """应用变换 - 使用白边填充"""
    h, w = rgb_img.shape[:2]
    aligned = cv2.warpPerspective(rgb_img, transform_matrix, (w, h),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
    return aligned


def visualize_result(rgb_aligned, ir_img, pts_rgb_transformed, pts_ir, output_path):
    """创建最终对比可视化"""

    h, w = ir_img.shape[:2]

    if len(ir_img.shape) == 2:
        ir_color = cv2.cvtColor(ir_img, cv2.COLOR_GRAY2BGR)
    else:
        ir_color = ir_img.copy()

    # 在对齐后的图像上绘制点
    rgb_vis = rgb_aligned.copy()
    ir_vis = ir_color.copy()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)]

    for i, ((x_rgb, y_rgb), (x_ir, y_ir)) in enumerate(zip(pts_rgb_transformed, pts_ir)):
        color = colors[i % len(colors)]

        cv2.circle(rgb_vis, (int(x_rgb), int(y_rgb)), 12, color, 2)
        cv2.putText(rgb_vis, str(i+1), (int(x_rgb)+15, int(y_rgb)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.circle(ir_vis, (int(x_ir), int(y_ir)), 12, color, 2)
        cv2.putText(ir_vis, str(i+1), (int(x_ir)+15, int(y_ir)-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 计算误差
        error = np.linalg.norm([x_rgb - x_ir, y_rgb - y_ir])
        if error > 10:
            cv2.line(rgb_vis, (int(x_rgb), int(y_rgb)), (int(x_ir), int(y_ir)), (0, 0, 255), 2)

    # 并排显示
    vis = np.hstack([rgb_vis, ir_vis])

    # 分隔线
    cv2.line(vis, (w, 0), (w, h), (255, 255, 255), 3)

    # 标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(vis, 'Aligned RGB', (20, 40), font, 1.2, (0, 255, 0), 2)
    cv2.putText(vis, 'IR Reference', (w + 20, 40), font, 1.2, (0, 255, 0), 2)
    cv2.putText(vis, 'Check: Colored markers should align',
                (20, h - 20), font, 0.8, (255, 255, 0), 2)

    cv2.imwrite(str(output_path), vis)


def main():
    print("\n" + "=" * 80)
    print("手动标点配准工具 (并排显示版) - 白边填充模式")
    print("=" * 80)

    # 固定路径
    RGB_DIR = r"E:\mokuai\video\extracted_frames\rgb\fifth"
    IR_DIR = r"E:\mokuai\video\extracted_frames\nir\fifth"
    OUTPUT_DIR = r"E:\mokuai\video\peizhun\fifth"

    print(f"\n路径配置:")
    print(f"  RGB: {RGB_DIR}")
    print(f"  IR:  {IR_DIR}")
    print(f"  输出: {OUTPUT_DIR}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载图像
    rgb_files = sorted(Path(RGB_DIR).glob("*.jpg")) + sorted(Path(RGB_DIR).glob("*.png"))
    ir_files = sorted(Path(IR_DIR).glob("*.jpg")) + sorted(Path(IR_DIR).glob("*.png"))

    if len(rgb_files) == 0 or len(ir_files) == 0:
        print("✗ 未找到图像")
        return

    print(f"\n找到 {len(rgb_files)} 对图像")

    # 选择标定图像
    use_first = input("使用第1对图像进行标定? (y/n): ").strip().lower()

    if use_first == 'y':
        idx = 0
    else:
        idx = int(input(f"请输入图像编号 (1-{len(rgb_files)}): ")) - 1

    rgb_calib_img = imread_chinese(rgb_files[idx])
    ir_calib_img = imread_chinese(ir_files[idx])

    if rgb_calib_img is None or ir_calib_img is None:
        print("✗ 无法读取图像")
        return

    # 并排标点
    pts_rgb, pts_ir = collect_points_side_by_side(rgb_calib_img, ir_calib_img)

    # 保存标点
    points_data = {
        'points_rgb': pts_rgb.tolist(),
        'points_ir': pts_ir.tolist(),
        'num_points': len(pts_rgb),
        'image_used': str(rgb_files[idx].name)
    }

    with open(output_path / "manual_points.json", 'w') as f:
        json.dump(points_data, f, indent=2)

    print(f"\n✓ 标点已保存: {output_path / 'manual_points.json'}")

    # 计算变换
    method = input("\n选择变换类型 (affine/homography) [默认: affine]: ").strip().lower()
    if not method:
        method = 'affine'

    transform, inlier_ratio = compute_transformation_from_points(pts_rgb, pts_ir, method)

    if transform is None:
        return

    # 保存变换
    transform_data = {
        'transform_matrix': transform.tolist(),
        'method': method,
        'inlier_ratio': float(inlier_ratio),
        'num_points': len(pts_rgb)
    }

    with open(output_path / "transform_matrix.json", 'w') as f:
        json.dump(transform_data, f, indent=2)

    print(f"✓ 变换矩阵已保存")

    # 测试对齐
    print("\n生成对齐可视化...")
    rgb_aligned = apply_transformation(rgb_calib_img, transform)

    # 计算变换后的点位置
    pts_rgb_transformed = cv2.perspectiveTransform(
        pts_rgb.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)

    vis_path = output_path / "alignment_result.jpg"
    visualize_result(rgb_aligned, ir_calib_img, pts_rgb_transformed, pts_ir, vis_path)

    print(f"✓ 可视化已保存: {vis_path}")

    # 批量处理
    print("\n" + "=" * 80)
    confirm = input("对齐效果满意? 继续批量处理? (y/n): ").strip().lower()

    if confirm != 'y':
        print("\n已取消")
        return

    print(f"\n批量处理 {len(rgb_files)} 对图像...")

    output_rgb_dir = output_path / "RGB_aligned"
    output_ir_dir = output_path / "IR_aligned"
    output_rgb_dir.mkdir(exist_ok=True)
    output_ir_dir.mkdir(exist_ok=True)

    success = 0

    for rgb_file, ir_file in tqdm(zip(rgb_files, ir_files), total=len(rgb_files), desc="配准中"):

        rgb_img = imread_chinese(rgb_file)
        ir_img = imread_chinese(ir_file)

        if rgb_img is None or ir_img is None:
            continue

        try:
            # 对齐
            rgb_aligned = apply_transformation(rgb_img, transform)

            # 简单裁剪中心区域
            h, w = ir_img.shape[:2]
            margin = int(min(w, h) * 0.05)

            rgb_crop = rgb_aligned[margin:h-margin, margin:w-margin]
            ir_crop = ir_img[margin:h-margin, margin:w-margin]

            # 缩放
            rgb_final = cv2.resize(rgb_crop, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
            ir_final = cv2.resize(ir_crop, (1280, 720), interpolation=cv2.INTER_LANCZOS4)

            # 保存
            if imwrite_chinese(output_rgb_dir / rgb_file.name, rgb_final):
                if imwrite_chinese(output_ir_dir / ir_file.name, ir_final):
                    success += 1

        except Exception as e:
            print(f"\n✗ 处理失败 {rgb_file.name}: {e}")

    print(f"\n✓ 批量处理完成: {success}/{len(rgb_files)}")
    print(f"\n输出位置:")
    print(f"  RGB: {output_rgb_dir}")
    print(f"  IR:  {output_ir_dir}")


if __name__ == "__main__":
    main()