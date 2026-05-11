import cv2
import os
import glob
import numpy as np
import re


def natural_sort_key(s):
    """确保图片严格按时间顺序读取"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def realistic_rgb_glare(image, severity):
    """引入 severity (0.0~1.0) 控制眩光强度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_mask = cv2.dilate(bright_mask, None, iterations=3)
    bloom = cv2.GaussianBlur(bright_mask, (151, 151), 40)
    bloom = cv2.cvtColor(bloom, cv2.COLOR_GRAY2BGR)

    # 根据 severity 动态调整过曝和光晕强度
    alpha = 1.0 + (0.2 * severity)
    beta = 40 * severity
    overexposed = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    bloom_weight = 0.8 * severity
    glare_img = cv2.addWeighted(overexposed, 1.0, bloom, bloom_weight, 0)
    return np.clip(glare_img, 0, 255).astype(np.uint8)


def realistic_rgb_fog(image, severity):
    """引入 severity (0.0~1.0) 控制浓雾强度"""
    # 雾越浓，对比度越低 (alpha越小)，亮度底线越高 (beta越大)
    alpha = 1.0 - (0.65 * severity)
    beta = 130 * severity
    foggy = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 雾越浓，画面越模糊
    blur_ksize = int(5 * severity) * 2 + 1  # 保证是奇数 1, 3, 5, 7, 9...
    if blur_ksize > 1:
        foggy = cv2.GaussianBlur(foggy, (blur_ksize, blur_ksize), 0)
    return foggy


def realistic_nir_degradation(image):
    """NIR 保持稳定，仅做轻微的物理环境噪点模拟"""
    attenuated = cv2.convertScaleAbs(image, alpha=0.85, beta=15)
    noise = np.random.normal(0, 12, attenuated.shape).astype(np.float32)
    noisy_nir = cv2.add(attenuated.astype(np.float32), noise)
    blurred_nir = cv2.GaussianBlur(noisy_nir, (3, 3), 0)
    return np.clip(blurred_nir, 0, 255).astype(np.uint8)


def get_disaster_profile(frame_idx, total_frames):
    """
    🌟 核心剧本：根据视频进度，动态计算当前属于什么灾难，以及灾难的强度。
    00%~10%: 晴空万里 (Baseline)
    10%~30%: 浓雾逐渐降临
    30%~50%: 浓雾逐渐消散
    50%~60%: 晴空万里
    60%~80%: 极昼眩光逐渐爆发
    80%~100%: 眩光逐渐消散
    """
    ratio = frame_idx / total_frames

    if ratio < 0.1:
        return "CLEAR", 0.0
    elif ratio < 0.3:
        return "FOG", (ratio - 0.1) / 0.2  # 0.0 -> 1.0
    elif ratio < 0.5:
        return "FOG", 1.0 - (ratio - 0.3) / 0.2  # 1.0 -> 0.0
    elif ratio < 0.6:
        return "CLEAR", 0.0
    elif ratio < 0.8:
        return "GLARE", (ratio - 0.6) / 0.2  # 0.0 -> 1.0
    else:
        return "GLARE", 1.0 - (ratio - 0.8) / 0.2  # 1.0 -> 0.0


def process_dataset(input_dir, output_dir, mode="RGB"):
    os.makedirs(output_dir, exist_ok=True)
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        img_files.extend(glob.glob(os.path.join(input_dir, ext)))

    # ⚠️ 必须自然排序，保证时间轴绝对连续！
    img_files = sorted(img_files, key=natural_sort_key)
    total_frames = len(img_files)

    print(f"🎬 找到 {total_frames} 张连续帧，开始执行 {mode} 模态【时序渐进式】物理级退化...")

    for idx, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)

        img = cv2.imread(img_path)
        if img is None: continue

        if mode == "RGB":
            # 获取当前帧的灾难类型和严重程度 (0.0 ~ 1.0)
            disaster_type, severity = get_disaster_profile(idx, total_frames)

            if disaster_type == "CLEAR" or severity <= 0.05:
                result_img = img  # 几乎无退化
            elif disaster_type == "FOG":
                result_img = realistic_rgb_fog(img, severity)
            elif disaster_type == "GLARE":
                result_img = realistic_rgb_glare(img, severity)

        elif mode == "NIR":
            # NIR 模态不受天气剧变影响，维持稳定红外底噪
            result_img = realistic_nir_degradation(img)

        cv2.imwrite(out_path, result_img)

        if (idx + 1) % 50 == 0:
            print(f"   已处理 {idx + 1}/{total_frames} 帧...")

    print(f"✅ {mode} 验证集生成完成！保存在: {output_dir}")


if __name__ == '__main__':
    # ==========================================
    # ⚠️ 请确保这里的输入文件夹是你刚才用脚本“对齐抽帧”出来的高清原图！
    # ==========================================

    # 1. 制造 RGB 的灾难时刻
    INPUT_RGB = r"E:\mokuai\track_val\peizhun\RGB_aligned"
    OUTPUT_RGB = r"E:\mokuai\track_val\peizhun\rgb_degraded"
    process_dataset(INPUT_RGB, OUTPUT_RGB, mode="RGB")

    # 2. 维持 NIR 的物理特性
    INPUT_NIR = r"E:\mokuai\track_val\peizhun\IR_aligned"
    OUTPUT_NIR = r"E:\mokuai\track_val\peizhun\nir_degraded"
    process_dataset(INPUT_NIR, OUTPUT_NIR, mode="NIR")