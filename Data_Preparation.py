# Split the Sentinel data from GEE into model input size (64*64)

import rasterio
import numpy as np
import os

# Function to split image into patches, pad if necessary, and skip empty patches
def split_image_to_patches(image_path, patch_size=64, output_dir='patches', prefix_to_remove=""):
    # 获取原始文件名（不带扩展名）
    original_name = os.path.splitext(os.path.basename(image_path))[0]

    # 移除指定的前缀 (比如 "S1_" 或 "S2_")
    if prefix_to_remove and original_name.startswith(prefix_to_remove):
        original_name = original_name[len(prefix_to_remove):]

    # 打开影像
    with rasterio.open(image_path) as src:
        # 读取整个图像
        image = src.read()
        height, width = image.shape[1], image.shape[2]

        # 获取无数据值的定义
        nodata_value = src.nodata
        if nodata_value is None:
            nodata_value = 0  # 可以根据你的需求更改为其他默认值

        # 计算行列需要的patch数
        num_patches_height = int(np.ceil(height / patch_size))
        num_patches_width = int(np.ceil(width / patch_size))

        # 创建输出目录（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 遍历每个patch
        for i in range(num_patches_height):
            for j in range(num_patches_width):
                # 计算当前patch的行列起始和结束索引
                start_row = i * patch_size
                start_col = j * patch_size
                end_row = min(start_row + patch_size, height)
                end_col = min(start_col + patch_size, width)

                # 提取当前patch
                patch = image[:, start_row:end_row, start_col:end_col]

                # 如果patch小于64x64，则进行0填充
                pad_height = patch_size - (end_row - start_row)
                pad_width = patch_size - (end_col - start_col)
                if pad_height > 0 or pad_width > 0:
                    patch = np.pad(patch, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=nodata_value)

                # 检查是否全为无效值（无数据）
                if nodata_value is not None and np.all(patch == nodata_value):
                    print(f"Skipped patch at {i}_{j} due to no valid data.")
                    continue

                # 保存当前patch，命名为 原始文件名_i_j.tif
                patch_filename = f'{original_name}_{i}_{j}.tif'
                patch_path = os.path.join(output_dir, patch_filename)

                if os.path.exists(patch_path):
                    continue

                # 写入patch到磁盘
                with rasterio.open(
                        patch_path, 'w',
                        driver='GTiff',
                        height=patch_size,
                        width=patch_size,
                        count=image.shape[0],  # 通道数量
                        dtype=image.dtype,
                        crs=src.crs,
                        transform=src.window_transform(((start_row, end_row), (start_col, end_col)))
                ) as dst:
                    dst.write(patch)

                print(f'Saved patch {patch_filename}')

# Prepare Sentinel-1 and Sentinel-2 images
sentinel1_image_path = r'./data/s1/s1_Austin_test.tif'
sentinel2_image_path = r'./data/s2/s2_Austin_test.tif'

# Split and pad Sentinel-1 image
split_image_to_patches(sentinel1_image_path, patch_size=64, output_dir='./data/s1_test', prefix_to_remove="s1_")

# Split and pad Sentinel-2 image
split_image_to_patches(sentinel2_image_path, patch_size=64, output_dir='./data/s2_test', prefix_to_remove="s2_")
