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


def check_gpu_availability() -> Dict[str, Any]:
    """检查GPU可用性"""
    gpu_info = {
        'available': False,
        'cuda_available': False,
        'num_gpus': 0,
        'gpu_names': []
    }

    # 检查CUDA
    try:
        import torch
        gpu_info['cuda_available'] = torch.cuda.is_available()
        if gpu_info['cuda_available']:
            gpu_info['available'] = True
            gpu_info['num_gpus'] = torch.cuda.device_count()
            gpu_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_info['num_gpus'])]
    except ImportError:
        pass

    # 检查OpenCL
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            gpu_info['available'] = True
            gpu_info['opencl_available'] = True
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.GPU)
                if devices:
                    gpu_info['num_gpus'] += len(devices)
                    gpu_info['gpu_names'].extend([d.name for d in devices])
    except ImportError:
        pass

    return gpu_info


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


def apply_gaussian_noise_cuda(image: np.ndarray, device, mean: float = 0, sigma: float = 0.1) -> np.ndarray:
    """使用CUDA应用高斯噪声"""
    import torch
    img_tensor = torch.from_numpy(image).float().to(device) / 255.0
    gaussian_noise = torch.randn_like(img_tensor) * sigma + mean
    noisy_image = torch.clamp(img_tensor + gaussian_noise, 0, 1)
    return (noisy_image.cpu().numpy() * 255).astype(np.uint8)


def apply_poisson_noise_cuda(image: np.ndarray, device) -> np.ndarray:
    """使用CUDA应用泊松噪声"""
    import torch
    img_tensor = torch.from_numpy(image).float().to(device) / 255.0
    poisson_noisy = torch.poisson(img_tensor * 255) / 255.0
    poisson_noisy = torch.clamp(poisson_noisy, 0, 1)
    return (poisson_noisy.cpu().numpy() * 255).astype(np.uint8)


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


def process_single_dicom_gpu(dicom_path: str, output_dir: str, gpu_info: Dict[str, Any], device_id: int = 0) -> None:
    """处理单个DICOM文件（GPU版本）"""
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

        # 使用GPU加速噪声添加（如果可用）
        if gpu_info.get('cuda_available', False):
            import torch
            # 设置设备
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

            for orient_key, orient_img in orientations.items():
                # 保存无噪声版本
                Image.fromarray(orient_img).save(output_folder / f"{orient_key}.png")

                # 生成并保存高斯噪声版本
                gaussian_noisy = apply_gaussian_noise_cuda(orient_img, device)
                Image.fromarray(gaussian_noisy).save(output_folder / f"{orient_key.replace('_0', '_1')}.png")

                # 生成并保存泊松噪声版本
                poisson_noisy = apply_poisson_noise_cuda(orient_img, device)
                Image.fromarray(poisson_noisy).save(output_folder / f"{orient_key.replace('_0', '_2')}.png")
        else:
            # 回退到CPU处理
            for orient_key, orient_img in orientations.items():
                # 保存无噪声版本
                Image.fromarray(orient_img).save(output_folder / f"{orient_key}.png")

                # 生成并保存高斯噪声版本
                gaussian_noisy = apply_gaussian_noise(orient_img)
                Image.fromarray(gaussian_noisy).save(output_folder / f"{orient_key.replace('_0', '_1')}.png")

                # 生成并保存泊松噪声版本
                poisson_noisy = apply_poisson_noise(orient_img)
                Image.fromarray(poisson_noisy).save(output_folder / f"{orient_key.replace('_0', '_2')}.png")

        print(f"处理完成: {base_name} (GPU: {device_id})")

    except Exception as e:
        print(f"处理文件 {dicom_path} 时出错: {str(e)}")


def process_dicom_files_gpu(dicom_files: List[str], output_dir: str) -> None:
    """处理多个DICOM文件（GPU版本）"""
    gpu_info = check_gpu_availability()
    print(f"GPU信息: {gpu_info}")

    if not gpu_info['available']:
        print("未检测到可用GPU，回退到CPU模式")
        from dcm_to_png_cpu import process_dicom_files
        process_dicom_files(dicom_files, output_dir)
        return

    # 如果有多个GPU，分配任务
    num_gpus = gpu_info.get('num_gpus', 1)
    files_per_gpu = (len(dicom_files) + num_gpus - 1) // num_gpus  # 向上取整

    # 使用多进程处理，每个GPU一个进程
    with timing_context("总处理时间"):
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i in range(num_gpus):
                start_idx = i * files_per_gpu
                end_idx = min(start_idx + files_per_gpu, len(dicom_files))
                gpu_files = dicom_files[start_idx:end_idx]

                if gpu_files:
                    futures.append(
                        executor.submit(process_gpu_batch, gpu_files, output_dir, gpu_info, i)
                    )

            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"处理过程中出错: {str(e)}")


def process_gpu_batch(dicom_files: List[str], output_dir: str, gpu_info: Dict[str, Any], device_id: int) -> None:
    """处理一批DICOM文件（GPU版本）"""
    for dicom_file in dicom_files:
        process_single_dicom_gpu(dicom_file, output_dir, gpu_info, device_id)


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
        print("用法: python dcm_to_png_gpu.py <输入目录> <输出目录>")
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
    process_dicom_files_gpu(dicom_files, output_dir)

    print("所有处理完成")


if __name__ == "__main__":
    main()