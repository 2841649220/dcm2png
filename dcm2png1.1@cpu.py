import os
import sys
import warnings
import multiprocessing
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

import numpy as np
import pydicom
from PIL import Image
import cv2

# 忽略不必要的警告
warnings.filterwarnings("ignore")


@contextmanager
def timing_context(description: str):
    """计时上下文管理器"""
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.2f} 秒")


def check_cpu_features() -> Dict[str, Any]:
    """检查CPU支持的指令集和特性"""
    cpu_info = {
        'avx2': False,
        'avx512': False,
        'fma3': False,
        'cores': multiprocessing.cpu_count(),
        'memory': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3) if hasattr(os,
                                                                                                     'sysconf') else None
    }

    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        flags = info.get('flags', [])
        cpu_info['avx2'] = 'avx2' in flags
        cpu_info['avx512'] = any(f in flags for f in ['avx512f', 'avx512'])
        cpu_info['fma3'] = 'fma3' in flags
    except ImportError:
        # 如果cpuinfo不可用，尝试其他方法检测
        try:
            # 在Linux上检查/proc/cpuinfo
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                    cpu_info['avx2'] = 'avx2' in content
                    cpu_info['avx512'] = any(f in content for f in ['avx512f', 'avx512'])
                    cpu_info['fma3'] = 'fma3' in content
        except:
            pass

    return cpu_info


def optimize_numpy_for_cpu(cpu_features: Dict[str, Any]) -> None:
    """根据CPU特性优化NumPy"""
    # 设置线程数
    if cpu_features['cores'] > 1:
        os.environ['OMP_NUM_THREADS'] = str(cpu_features['cores'])
        os.environ['MKL_NUM_THREADS'] = str(cpu_features['cores'])

    # 如果支持AVX512，尝试使用更优化的库
    if cpu_features['avx512']:
        try:
            import mkl
            mkl.set_num_threads(cpu_features['cores'])
        except ImportError:
            pass


def normalize_dicom_image(ds: pydicom.Dataset) -> np.ndarray:
    """标准化DICOM图像数据"""
    img = ds.pixel_array.astype(np.float32)

    # 应用窗宽窗位
    if hasattr(ds, 'WindowWidth') and hasattr(ds, 'WindowCenter'):
        window_width = ds.WindowWidth
        window_center = ds.WindowCenter
        if isinstance(window_width, pydicom.multival.MultiValue):
            window_width = window_width[0]
        if isinstance(window_center, pydicom.multival.MultiValue):
            window_center = window_center[0]

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

    # 归一化到0-255
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img.astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
    """应用高斯噪声"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255)


def apply_poisson_noise(image: np.ndarray) -> np.ndarray:
    """应用泊松噪声"""
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(image * vals) / float(vals)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def process_single_dicom(dicom_path: str, output_dir: str) -> None:
    """处理单个DICOM文件"""
    try:
        # 读取DICOM文件
        with timing_context(f"读取 {Path(dicom_path).name}"):
            ds = pydicom.dcmread(dicom_path)
            image = normalize_dicom_image(ds)

        # 创建输出目录
        base_name = Path(dicom_path).stem
        output_folder = Path(output_dir) / base_name
        output_folder.mkdir(exist_ok=True)

        # 生成不同方向的图像
        orientations = {
            '1_0': image,  # 原图
            '2_0': cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),  # 向右旋转90度
            '3_0': cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),  # 向左旋转90度
            '4_0': cv2.rotate(image, cv2.ROTATE_180),  # 翻转180度
        }

        # 为每个方向添加噪声
        for orient_key, orient_img in orientations.items():
            # 保存无噪声版本
            Image.fromarray(orient_img).save(output_folder / f"{orient_key}.png")

            # 生成并保存高斯噪声版本
            gaussian_noisy = apply_gaussian_noise(orient_img)
            Image.fromarray(gaussian_noisy).save(output_folder / f"{orient_key.replace('_0', '_1')}.png")

            # 生成并保存泊松噪声版本
            poisson_noisy = apply_poisson_noise(orient_img)
            Image.fromarray(poisson_noisy).save(output_folder / f"{orient_key.replace('_0', '_2')}.png")

        print(f"处理完成: {base_name}")

    except Exception as e:
        print(f"处理文件 {dicom_path} 时出错: {str(e)}")


def process_dicom_files(dicom_files: List[str], output_dir: str, max_workers: Optional[int] = None) -> None:
    """处理多个DICOM文件"""
    cpu_features = check_cpu_features()
    print(f"CPU特性: {cpu_features}")

    # 根据CPU核心数设置工作线程数
    if max_workers is None:
        max_workers = min(cpu_features['cores'], len(dicom_files))

    # 优化NumPy设置
    optimize_numpy_for_cpu(cpu_features)

    # 使用进程池并行处理
    with timing_context("总处理时间"):
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for dicom_file in dicom_files:
                futures.append(
                    executor.submit(process_single_dicom, dicom_file, output_dir)
                )

            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理过程中出错: {str(e)}")


def find_dicom_files(input_dir: str) -> List[str]:
    """查找输入目录中的所有DICOM文件"""
    dicom_extensions = ['.dcm', '.dicom', '.DCM', '.DICOM']
    dicom_files = []

    for ext in dicom_extensions:
        dicom_files.extend(Path(input_dir).rglob(f"*{ext}"))

    return [str(f) for f in dicom_files]


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python dcm_to_png_cpu.py <输入目录> <输出目录>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # 查找DICOM文件
    with timing_context("查找DICOM文件"):
        dicom_files = find_dicom_files(input_dir)

    if not dicom_files:
        print("未找到DICOM文件")
        sys.exit(1)

    print(f"找到 {len(dicom_files)} 个DICOM文件")

    # 处理文件
    process_dicom_files(dicom_files, output_dir)

    print("所有处理完成")


if __name__ == "__main__":
    main()
