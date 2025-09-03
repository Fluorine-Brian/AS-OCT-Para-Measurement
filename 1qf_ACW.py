#测量ACW ACW ACW ACW ACW ACW ACW ACW ACW ACW ACW ACW ACW
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv
import math

def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open(out_csv_path, 'w', newline='') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)
        fieldnames = csv_reader.fieldnames + ['Intersection Length']
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)

        # 写入 CSV 文件的标题行
        csv_writer.writeheader()

        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    # 从文件名中提取图像名称
                    image_name = os.path.splitext(file)[0]

                    # 查找匹配图像名称的行
                    for row in csv_reader:
                        if image_name in row['文件名']:
                            print('文件名--',row['文件名'],'   image_name--',image_name)
                            # 提取坐标点

                            img = cv2.imread(image_path)
                            height, width, channels = img.shape
                            point1 = (float(row['左眼X'])*width, float(row['左眼Y'])*height)
                            point2 = (float(row['右眼X'])*width, float(row['右眼Y'])*height)

                            # 计算交点长度
                            length = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

                            # 可视化线段、白色部分的边界和交点
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img, cmap='gray')
                            plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')  # 线段
                            plt.xlim(0, width)
                            plt.ylim(height, 0)
                            plt.show()
                            # 更新行数据
                            row['Intersection Length'] = length

                            # 写入更新后的行到新的 CSV 文件
                            csv_writer.writerow(row)

                            break  # 停止在 CSV 文件中继续查找匹配的行

                    # 将原始 CSV 读取器的位置重置为文件开头，以便下一次迭代时重新开始读取
                    csvfile.seek(0)

        print("New CSV file created: new_output.csv")

# 调用函数处理图像文件并更新 CSV 文件
# root_directory = '/home/liujia/project/measurement/baineizhang/晶状体/jingzhuangti'  # 根文件夹路径
# csv_path = '/home/liujia/project/measurement/cataract.csv'  # CSV 文件路径
root_directory = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_img_1_qf'  # 根文件夹路径
csv_path = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_1011_all.csv'  # CSV 文件路径
out_csv_path = 'MyCSV/ACW.csv'
process_images(root_directory, csv_path)