import os
import numpy as np
import pydicom
import cv2
import argparse
import concurrent.futures
from numba import jit, prange
import multiprocessing as mp

# 启用AVX2优化
os.environ['NUMBA_CPU_NAME'] = 'skylake-avx512'
os.environ['NUMBA_LLVM_AVX512'] = '1'


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def add_gaussian_noise(image, mean=0.0, sigma=25.0):
    """使用Numba加速的高斯噪声添加函数"""
    row, col = image.shape
    noisy_image = np.empty_like(image, dtype=np.uint8)

    for i in prange(row):
        for j in prange(col):
            noise = np.random.normal(mean, sigma)
            pixel_value = image[i, j] + noise
            # 限制像素值在0-255范围内
            if pixel_value < 0:
                pixel_value = 0
            elif pixel_value > 255:
                pixel_value = 255
            noisy_image[i, j] = pixel_value

    return noisy_image


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def add_poisson_noise(image):
    """使用Numba加速的泊松噪声添加函数"""
    row, col = image.shape
    noisy_image = np.empty_like(image, dtype=np.uint8)

    for i in prange(row):
        for j in prange(col):
            # 泊松噪声：像素值越高，噪声影响越小
            pixel_value = image[i, j]
            noisy_value = np.random.poisson(max(pixel_value, 1))
            # 限制像素值在0-255范围内
            if noisy_value < 0:
                noisy_value = 0
            elif noisy_value > 255:
                noisy_value = 255
            noisy_image[i, j] = noisy_value

    return noisy_image


@jit(nopython=True, fastmath=True, cache=True)
def rotate_image(image, angle):
    """使用Numba加速的图像旋转函数"""
    if angle == 90:
        return np.rot90(image, 1)
    elif angle == -90:
        return np.rot90(image, 3)
    elif angle == 180:
        return np.rot90(image, 2)
    else:
        return image


def process_single_dcm_file(dcm_path, output_dir, gaussian_sigma=25.0):
    """处理单个DCM文件"""
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(dcm_path)

        # 提取像素数据
        pixel_array = ds.pixel_array

        # 转换为8位灰度图像，保留更多原始信息
        if pixel_array.dtype != np.uint8:
            # 使用窗宽窗位信息（如果存在）来优化显示
            if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
                window_width = ds.WindowWidth
                window_center = ds.WindowCenter
                if hasattr(window_width, '__iter__'):
                    window_width = window_width[0]
                if hasattr(window_center, '__iter__'):
                    window_center = window_center[0]

                # 应用窗宽窗位
                pixel_min = window_center - window_width // 2
                pixel_max = window_center + window_width // 2
                pixel_array = np.clip(pixel_array, pixel_min, pixel_max)
                pixel_array = ((pixel_array - pixel_min) /
                               (pixel_max - pixel_min) * 255).astype(np.uint8)
            else:
                # 如果没有窗宽窗位信息，使用2%和98%百分位来减少异常值的影响
                p2, p98 = np.percentile(pixel_array, (2, 98))
                if p98 > p2:
                    pixel_array = np.clip(pixel_array, p2, p98)
                    pixel_array = ((pixel_array - p2) /
                                   (p98 - p2) * 255).astype(np.uint8)
                else:
                    pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        # 创建输出目录
        base_name = os.path.splitext(os.path.basename(dcm_path))[0]
        output_subdir = os.path.join(output_dir, base_name)
        os.makedirs(output_subdir, exist_ok=True)

        # 生成基础变换的图像
        transformations = [
            ("1_0", pixel_array),  # 原图
            ("2_0", rotate_image(pixel_array, 90)),  # 向左旋转90度
            ("3_0", rotate_image(pixel_array, -90)),  # 向右旋转90度
            ("4_0", rotate_image(pixel_array, 180)),  # 翻转180度
        ]

        # 添加带高斯噪声的版本
        gaussian_transformations = []
        for name, img in transformations:
            noisy_img = add_gaussian_noise(img, sigma=gaussian_sigma)
            gaussian_transformations.append((name.replace("_0", "_1"), noisy_img))

        # 添加带泊松噪声的版本
        poisson_transformations = []
        for name, img in transformations:
            noisy_img = add_poisson_noise(img)
            poisson_transformations.append((name.replace("_0", "_2"), noisy_img))

        # 合并所有图像
        all_images = transformations + gaussian_transformations + poisson_transformations

        # 保存所有图像
        for name, img in all_images:
            output_path = os.path.join(output_subdir, f"{name}.png")
            # 使用无损压缩的PNG格式保存
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        return f"处理完成: {base_name} -> {output_subdir}"
    except Exception as e:
        return f"处理文件 {dcm_path} 时出错: {str(e)}"


def process_dcm_folder(input_folder, output_folder, gaussian_sigma=25.0, max_workers=None):
    """使用多进程处理文件夹中的所有DCM文件"""
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有DCM文件
    dcm_files = []
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.dcm', '.dicom')):
                dcm_files.append(os.path.join(root, filename))

    # 设置工作进程数（使用所有可用的CPU核心）
    if max_workers is None:
        max_workers = mp.cpu_count()

    # 使用进程池并行处理
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_dcm_file, dcm_file, output_folder, gaussian_sigma): dcm_file
            for dcm_file in dcm_files
        }

        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_file):
            dcm_file = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"处理文件 {dcm_file} 时发生异常: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理DCM文件并生成多种变换的PNG图像')
    parser.add_argument('input_folder', help='输入文件夹路径，包含DCM文件')
    parser.add_argument('output_folder', help='输出文件夹路径')
    parser.add_argument('--workers', type=int, default=None,
                        help='工作进程数，默认为CPU核心数')
    parser.add_argument('--sigma', type=float, default=25.0,
                        help='高斯噪声的强度（标准差），默认为25.0')

    args = parser.parse_args()

    # 预热JIT编译器
    print("预热JIT编译器...")
    test_array = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    add_gaussian_noise(test_array)
    add_poisson_noise(test_array)
    rotate_image(test_array, 90)

    print(f"开始处理，使用 {args.workers if args.workers else mp.cpu_count()} 个工作进程...")
    process_dcm_folder(args.input_folder, args.output_folder, args.sigma, args.workers)
