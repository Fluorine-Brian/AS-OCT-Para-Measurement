"""
批量批量批量批量测量ACArea
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv


def find_area(mask_path):
    # 读取分割掩码图
    mask = cv2.imread(mask_path, 0)  # 以灰度模式读取图像
    # 找到最大白色部分的边界
    _, thresholded = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # 移除额外的维度
    points_reshaped = [point[0] for point in largest_contour]
    polygon = Polygon(points_reshaped)
    #计算面积
    area = polygon.area
    print(f"多边形面积: {area}")
    return area


def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open(out_csv_path, 'w', newline='') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)  # 创建 CSV 字典读取器
        fieldnames = csv_reader.fieldnames + ['Area']  # 添加新的字段名 'Area'
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)  # 创建 CSV 字典写入器

        # 写入 CSV 文件的标题行
        csv_writer.writeheader()

        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    image_path = image_path.replace("\\", "/")
                    # 从文件名中提取图像名称
                    image_name = os.path.splitext(file)[0]
                    # 查找匹配图像名称的行
                    for row in csv_reader:
                        if image_name in row['文件名']:
                            print('文件名--', row['文件名'], '   image_name--', image_name)
                            # 计算acarea
                            area = find_area(image_path)
                            # 更新行数据
                            row['Area'] = area
                            # 写入更新后的行到新的 CSV 文件
                            csv_writer.writerow(row)
                            break  # 停止在 CSV 文件中继续查找匹配的行
                    # 将原始 CSV 读取器的位置重置为文件开头，以便下一次迭代时重新开始读取
                    csvfile.seek(0)
        print("New CSV file created: OS-前房-ACArea.csv")


root_directory = r'C://srp_OCT/para_measure/output_img_1_qf'  # 根文件夹路径
csv_path = r'C://srp_OCT/para_measure/output_1011_all.csv'  # CSV 文件路径
out_csv_path = 'C://srp_OCT/para_measure/MyCSV/ACArea.csv'
process_images(root_directory, csv_path)
