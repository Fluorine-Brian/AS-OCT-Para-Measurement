# 批量批量批量批量测量 ACD ACD ACD ACD ACD ACD ACD ACD ACD
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
import os
import csv


def find_intersection(mask_path, point1, point2):
    # 计算垂直平分线的两个点
    def perpendicular_points(mid, p1, p2):
        vector = np.array(p2) - np.array(p1)
        perp_vector = np.array([-vector[1], vector[0]])
        midpoint_vector = np.array(mid)
        pt1 = midpoint_vector + perp_vector
        pt2 = midpoint_vector - perp_vector
        return pt1, pt2

    # 读取分割掩码图
    mask = cv2.imread(mask_path, 0)  # 以灰度模式读取图像
    height, width = mask.shape

    # 计算线段的中点
    midpoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)

    # 找到最大白色部分的边界
    _, thresholded = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    # 计算垂直平分线的两个端点
    pt1, pt2 = perpendicular_points(midpoint, point1, point2)

    points_reshaped = [point[0] for point in largest_contour]  # 移除额外的维度
    polygon = Polygon(points_reshaped)

    line = LineString([pt1, pt2])

    # # 可视化线段、白色部分的边界和交点
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='gray')
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')  # 线段
    plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'b-')  # 垂直平分线
    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.show()

    if line.intersects(polygon):
        intersection = line.intersection(polygon)
        if intersection.geom_type == 'LineString':
            intersection_length = intersection.length
            print("Length of intersection segment:", intersection_length, ' -px')
        else:
            print("The intersection is not a line segment.")
            intersection_length = 0
    else:
        print("The line does not intersect with the polygon.")
        intersection_length = 0
    return intersection_length


def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open(out_csv_path, 'w', newline='') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)  # 创建 CSV 字典读取器
        fieldnames = csv_reader.fieldnames + ['Intersection Length']  # 添加新的字段名 'Intersection Length'
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)  # 创建 CSV 字典写入器

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
                            print('文件名--', row['文件名'], '   image_name--', image_name)
                            # 提取坐标点
                            # point1 = (float(row['SS1 X']), float(row['SS1 Y']))
                            # point2 = (float(row['SS2 X']), float(row['SS2 Y']))
                            img = cv2.imread(image_path)
                            print(image_path)
                            height, width, channels = img.shape
                            point1 = (float(row['左眼X']) * width, float(row['左眼Y']) * height)
                            point2 = (float(row['右眼X']) * width, float(row['右眼Y']) * height)
                            print('point1', point1, 'point2', point2)

                            # 计算交点长度
                            intersection_length = find_intersection(image_path, point1, point2)

                            # 更新行数据
                            row['Intersection Length'] = intersection_length

                            # 写入更新后的行到新的 CSV 文件
                            csv_writer.writerow(row)

                            break  # 停止在 CSV 文件中继续查找匹配的行

                    # 将原始 CSV 读取器的位置重置为文件开头，以便下一次迭代时重新开始读取
                    csvfile.seek(0)

        print("New CSV file created: new_output.csv")


# 调用函数处理图像文件并更新 CSV 文件
# root_directory = '/home/liujia/project/measurement/zhengchang/晶状体/jingzhuangti'  # 根文件夹路径
# csv_path = '/home/liujia/project/measurement/nomal.csv'  # CSV 文件路径
# root_directory = '/home/wanglei/AutoMeasure-ASOCT/measurement/data_test'  # 根文件夹路径
# csv_path = '/home/wanglei/AutoMeasure-ASOCT/measurement/output-0816.csv'  # CSV 文件路径

root_directory = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_img_1_qf'  # 根文件夹路径
csv_path = r'C:\Users\PS\Desktop\ASOCT\4_198fin\output_1011_all.csv'  # CSV 文件路径
out_csv_path = 'MyCSV/ACD.csv'
process_images(root_directory, csv_path)