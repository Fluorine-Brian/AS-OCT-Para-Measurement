import os
import numpy as np
import pandas as pd
from collections import defaultdict
import itertools
import sys
from PIL import Image  # 用于加载PNG掩码图像

# --- 配置参数 ---
# 请将此路径修改为你的数据集根目录，即包含三个同学文件夹的 'mask' 文件夹的路径
# 例如：如果你的 mask.zip 解压到 C:\data\mask，那么这里就是 r"C:\data\mask"
DATA_ROOT_DIR = r"C:\srp_OCT\mask_compare\mask"  # <-- 请务必修改为你的实际路径

# 三位同学的文件夹名称列表
# 请根据你的实际文件夹名称修改，确保这些文件夹直接位于 DATA_ROOT_DIR 下
# 根据你提供的zip文件，第一个同学的文件夹名是 '严星茹mask'
# 请确保以下列表中的名称与你实际的三个同学文件夹名称完全一致
STUDENT_FOLDERS = [
    'yxy_mask',  # 假设这是第一个同学的文件夹名称
    'djr_mask',  # 请替换为实际的同学B文件夹名称
    'yjh_mask',  # 请替换为实际的同学C文件夹名称
]

# AS-OCT图像类型文件夹名称列表
# 请务必根据你的实际文件夹名称修改，确保这些文件夹直接位于每个同学文件夹下
# 根据你提供的zip文件，疾病类型文件夹名是 'normal'，而不是 'normalX5'
IMAGE_TYPES = ['normal', 'PACG+cataract', 'PACG', 'cataract']  # <-- 请务必核对这些名称与你的实际文件夹名称一致

# 需要对比的解剖部位列表
# 请务必根据你的实际文件夹名称修改，确保这些文件夹直接位于每个疾病类型文件夹下
# 根据你提供的zip文件，部位文件夹名是 '前房', '巩膜突', '晶状体'
# 请确保以下列表中的名称与你实际的五个部位文件夹名称完全一致
ANATOMICAL_PARTS = ['gongmotu', 'qianfang', '虹膜', 'jingzhuangti', '核']  # <-- 请务必核对这些名称与你的实际文件夹名称一致
EXCLUDE_PART = 'gongmotu'  # 不进行统计的部位

# --- 特定图像排除配置 ---
# 定义需要排除的图像列表。每个字典包含 'student_name', 'image_type', 'image_id'。
# 如果某个图像在某个同学的特定疾病类型下有问题，可以在这里添加。
# 例如：{'student_name': '余静荷-修改后20张掩码', 'image_type': 'normal', 'image_id': '93310857_20230516_092627_L_CASIA2_001_000'}
EXCLUDE_SPECIFIC_IMAGES = [
    {'student_name': '余静荷-修改后20张掩码', 'image_type': 'normal',
     'image_id': '93310857_20230516_092627_L_CASIA2_001_000'},
    # 如果有其他需要排除的图像，请在此处添加
    # {'student_name': '同学B的文件夹名称', 'image_type': 'disease_type_X', 'image_id': 'image_Y_id'},
]


# --- 辅助函数 ---

def load_png_mask(mask_path):
    """
    加载PNG掩码图像并转换为布尔NumPy数组。
    假设掩码图像是二值的，前景像素为非零，背景像素为零。
    Args:
        mask_path (str): PNG掩码图像文件路径。
    Returns:
        numpy.ndarray: 布尔类型的二进制掩码。如果加载失败，返回一个空的NumPy数组。
    """
    try:
        img = Image.open(mask_path).convert('L')  # 转换为灰度图，确保单通道
        mask_array = np.array(img)
        return mask_array > 0  # 将所有非零像素转换为True，零像素转换为False
    except FileNotFoundError:
        # print(f"警告: 掩码文件未找到: {mask_path}. 返回空数组。") # 避免过多打印
        return np.array([], dtype=bool)
    except Exception as e:
        print(f"错误加载掩码 {mask_path}: {e}. 返回空数组。")
        return np.array([], dtype=bool)


def calculate_iou(mask1, mask2):
    """
    计算两个二进制掩码的Intersection over Union (IoU)。
    Args:
        mask1 (numpy.ndarray): 第一个布尔类型或0/1的二进制掩码。
        mask2 (numpy.ndarray): 第二个布尔类型或0/1的二进制掩码。
    Returns:
        float: IoU值。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # 如果两个掩码都为空，IoU为1；如果一个为空另一个不为空，IoU为0
    return intersection / union


def calculate_dice(mask1, mask2):
    """
    计算两个二进制掩码的Dice系数。
    Args:
        mask1 (numpy.ndarray): 第一个布尔类型或0/1的二进制掩码。
        mask2 (numpy.ndarray): 第二个布尔类型或0/1的二进制掩码。
    Returns:
        float: Dice系数。
    """
    intersection = np.logical_and(mask1, mask2).sum()
    sum_of_areas = mask1.sum() + mask2.sum()

    if sum_of_areas == 0:
        return 1.0 if intersection == 0 else 0.0  # 如果两个掩码都为空，Dice为1；如果一个为空另一个不为空，Dice为0
    return (2.0 * intersection) / sum_of_areas


def is_image_excluded(student_name, image_type, image_id, exclude_list):
    """
    检查一个特定的图像是否在排除列表中。
    """
    for item in exclude_list:
        if item['student_name'] == student_name and \
                item['image_type'] == image_type and \
                item['image_id'] == image_id:
            return True
    return False


# --- 主程序 ---

def main():
    if len(STUDENT_FOLDERS) < 2:
        print("错误：至少需要两位同学的标注数据才能进行对比。请检查 STUDENT_FOLDERS 配置。")
        return

    # 存储所有学生、所有图像类型、所有部位、所有图像ID的掩码
    # 结构: student_name -> image_type -> part_name -> image_id -> mask_array (boolean numpy array)
    all_loaded_masks = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    print("--- 步骤1: 加载所有PNG掩码图像 ---")
    for student_folder in STUDENT_FOLDERS:
        student_name = student_folder.split(os.sep)[-1]
        print(f"处理同学: {student_name}")

        for image_type in IMAGE_TYPES:
            for part_name in ANATOMICAL_PARTS:
                part_folder_path = os.path.join(DATA_ROOT_DIR, student_folder, image_type, part_name)

                if not os.path.isdir(part_folder_path):
                    # print(f"警告: 路径不存在或不是目录: {part_folder_path}，跳过该部位。") # 避免过多打印
                    continue

                png_files = [f for f in os.listdir(part_folder_path) if f.endswith('.png')]

                if not png_files:
                    # print(f"警告: 在 {part_folder_path} 中未找到PNG掩码文件，跳过该部位。") # 避免过多打印
                    continue

                for png_file in png_files:
                    image_id = os.path.splitext(png_file)[0]  # 使用文件名（不含扩展名）作为图像ID

                    # 检查当前图像是否需要被排除
                    if is_image_excluded(student_name, image_type, image_id, EXCLUDE_SPECIFIC_IMAGES):
                        print(
                            f"排除图像: 同学 '{student_name}', 疾病 '{image_type}', 图像 '{image_id}' (根据配置跳过)。")
                        continue  # 跳过加载和存储此图像

                    mask_path = os.path.join(part_folder_path, png_file)
                    mask = load_png_mask(mask_path)

                    # 只有当mask不是空的NumPy数组时才存储
                    if mask.size > 0:
                        all_loaded_masks[student_name][image_type][part_name][image_id] = mask
                    else:
                        print(f"警告: 掩码文件 {mask_path} 加载失败或为空数组，跳过存储。")

    print("\n--- 步骤2: 进行掩码对比和指标计算 ---")

    individual_results_list = []

    # 生成所有独特的同学对 (例如 A-B, A-C, B-C)
    student_pairs = list(itertools.combinations(STUDENT_FOLDERS, 2))

    if not student_pairs:
        print("错误：没有足够的学生对进行对比。请检查 STUDENT_FOLDERS 配置。")
        return

    for student_folder1, student_folder2 in student_pairs:
        student_name1 = student_folder1.split(os.sep)[-1]
        student_name2 = student_folder2.split(os.sep)[-1]

        print(f"\n正在对比: {student_name1} vs {student_name2}")

        for image_type in IMAGE_TYPES:
            for part_name in ANATOMICAL_PARTS:
                if part_name == EXCLUDE_PART:
                    continue  # 排除巩膜突的统计

                # 检查两位学生是否都有该图像类型和部位的数据
                if image_type not in all_loaded_masks[student_name1] or \
                        part_name not in all_loaded_masks[student_name1][image_type] or \
                        image_type not in all_loaded_masks[student_name2] or \
                        part_name not in all_loaded_masks[student_name2][image_type]:
                    continue

                student1_part_masks = all_loaded_masks[student_name1][image_type][part_name]
                student2_part_masks = all_loaded_masks[student_name2][image_type][part_name]

                # 找出这对学生在该图像类型和部位下共同标注的图像ID
                common_image_ids_for_part = set(student1_part_masks.keys()).intersection(
                    set(student2_part_masks.keys()))

                if not common_image_ids_for_part:
                    continue

                for image_id in sorted(list(common_image_ids_for_part)):
                    # 再次检查，确保即使在 common_image_ids_for_part 中，
                    # 如果某个图像被排除，也不进行对比。
                    # 注意：由于我们在加载时已经排除了，这里理论上不会再遇到被排除的图像。
                    # 但作为双重保险，或者如果排除逻辑更复杂，可以保留。
                    # 目前的加载排除已经足够。

                    mask1 = student1_part_masks[image_id]
                    mask2 = student2_part_masks[image_id]

                    # 再次检查掩码是否有效（非空数组）
                    if mask1.size == 0 or mask2.size == 0:
                        print(
                            f"警告: 图像 {image_id}, 部位 {part_name} 在 {student_name1} 或 {student_name2} 中掩码无效 (空数组)，跳过此对比。")
                        continue

                    # 检查掩码形状是否兼容
                    if mask1.shape != mask2.shape:
                        print(
                            f"警告: 图像 {image_id}, 部位 {part_name} 在 {student_name1} 和 {student_name2} 中掩码形状不匹配 ({mask1.shape} vs {mask2.shape})，跳过此对比。")
                        continue

                    iou = calculate_iou(mask1, mask2)
                    dice = calculate_dice(mask1, mask2)

                    individual_results_list.append({
                        'Comparison_Pair': f"{student_name1} vs {student_name2}",
                        'Student1': student_name1,
                        'Student2': student_name2,
                        'Image_Type': image_type,
                        'Part': part_name,
                        'Image_ID': image_id,
                        'IoU': iou,
                        'Dice': dice
                    })

    print("\n--- 步骤3: 汇总结果并输出 ---")

    df_individual_results = pd.DataFrame(individual_results_list)

    if not df_individual_results.empty:
        detailed_output_excel_path = 'C:\srp_OCT\mask_compare\detailed_mask_comparison_results.xlsx'
        df_individual_results.to_excel(detailed_output_excel_path, index=False)
        print(f"\n详细的单个图像对比结果已保存到: {detailed_output_excel_path}")

        summary_df = df_individual_results.groupby(['Comparison_Pair', 'Image_Type', 'Part']).agg(
            IoU_Mean=('IoU', 'mean'),
            IoU_Std=('IoU', 'std'),
            Dice_Mean=('Dice', 'mean'),
            Dice_Std=('Dice', 'std')
        ).reset_index()

        summary_df['IoU_Std'] = summary_df['IoU_Std'].fillna(0)
        summary_df['Dice_Std'] = summary_df['Dice_Std'].fillna(0)

        summary_output_excel_path = 'C:\srp_OCT\mask_compare\summary_mask_comparison_results.xlsx'
        summary_df.to_excel(summary_output_excel_path, index=False)
        print(f"汇总的平均值和标准差结果已保存到: {summary_output_excel_path}")

        print("\n--- 汇总对比结果 (打印输出) ---")
        print(summary_df.to_string(index=False, float_format="%.4f"))

    else:
        print("没有生成任何对比结果。请检查数据路径、文件夹名称和标注文件。")


if __name__ == "__main__":
    main()
