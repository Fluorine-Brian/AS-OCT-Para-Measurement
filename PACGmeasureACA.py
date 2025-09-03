"""
批量测量aca750
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import sqrt, isclose
import cv2


# 读取灰度图并转换为numpy数组
def load_image(image_path):
    img = Image.open(image_path).convert('L')  # 确保是灰度图
    return np.array(img)


# 判断一个点是否在圆周上（允许一定的误差范围）
def is_on_circle(x, y, center, radius, epsilon):
    distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return abs(distance_squared - radius ** 2) <= epsilon ** 2


# 找到圆和白色区域的交点（只找圆周上的点）
def find_intersection_points(image, center, radius, epsilon=1):
    radius = int(radius)
    height, width = image.shape
    intersection_points = []
    for y in range(max(0, center[1] - radius), min(height, center[1] + radius + 1)):
        for x in range(max(0, center[0] - radius), min(width, center[0] + radius + 1)):
            if is_on_circle(x, y, center, radius, epsilon) and image[y, x] == 255:
                intersection_points.append((x, y))
                break  # 找到交点 退出循环
    return intersection_points


# 计算两点间的斜率
def calculate_slope(p1, p2):
    if p1[0] == p2[0]:  # 避免除以零的情况
        return float('inf')
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


# 求过点 B 的垂线方程
def perpendicular_line_equation(center, b_point):
    slope_center_b = calculate_slope(center, b_point)
    # 如果中心到 B 的斜率为无穷大（垂直线），则垂线为水平线
    if isclose(slope_center_b, float('inf'), abs_tol=1e-9):
        return 0, b_point[1]
    # 垂线斜率
    perp_slope = -1 / slope_center_b if slope_center_b != 0 else float('inf')
    # 使用点斜式方程 y - y1 = m(x - x1) 来计算垂线方程
    if isclose(perp_slope, float('inf'), abs_tol=1e-9):  # 如果垂线是垂直的
        return float('inf'), b_point[0]
    else:
        intercept = b_point[1] - perp_slope * b_point[0]
        return perp_slope, intercept


# 判断一个点是否在直线上（允许一定的误差范围）
def is_on_line(x, y, line, epsilon=1):
    slope, intercept = line
    if isclose(slope, float('inf'), abs_tol=1e-9):  # 如果直线是垂直的
        return abs(x - intercept) <= epsilon
    else:
        return abs(y - (slope * x + intercept)) <= epsilon


# 找到垂线和白色区域的交点
def find_perpendicular_intersection_points(image, line, epsilon=1):
    height, width = image.shape
    intersection_points = []
    for x in range(width):
        for y in range(height):
            if is_on_line(x, y, line, epsilon) and image[y, x] == 255:
                intersection_points.append((x, y))

    return intersection_points


def UseLine(a, b, image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 定义点 a 和 b 的坐标
    # 提取 x 和 y 坐标
    x_coords = [a[0], b[0]]
    y_coords = [a[1], b[1]]
    # 计算 ab 线段的方向向量
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    # 计算垂线的方向向量 (dy, -dx)
    perpendicular_vector = (dy, -dx)
    # 垂线的长度（可以根据需要调整）
    line_length = 200  # 示例长度，你可以根据图像大小调整
    # 计算垂线的两个端点
    b_perpendicular_1 = (
        int(b[0] + perpendicular_vector[0] * line_length / 2), int(b[1] + perpendicular_vector[1] * line_length / 2))
    b_perpendicular_2 = (
        int(b[0] - perpendicular_vector[0] * line_length / 2), int(b[1] - perpendicular_vector[1] * line_length / 2))
    # 创建一个与原图大小相同的空白图像，用于绘制垂线
    line_image = np.zeros_like(image)
    # 绘制原始线段和垂线
    cv2.line(line_image, a, b, (255, 255, 0), 20)  # 原始线段
    cv2.line(line_image, b_perpendicular_1, b_perpendicular_2, (255, 255, 0), 20)  # 垂线
    # 将原始图像和线条图像合并，以便在同一个图像上显示
    combined_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.8,
                                     cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR), 0.5, 0)
    # 创建一个布尔掩码，标记出白色区域（像素值为 255）
    white_mask = image == 255

    bottom_most_point = (0, 0)
    if dy != 0:
        slope_perpendicular = -dx / dy
        intercept_perpendicular = b[1] - slope_perpendicular * b[0]
        # 使用 np.where 根据垂线方程直接标记出 region_1 和 region_2
        y_coords, x_coords = np.indices(image.shape)
        region_1 = (y_coords < slope_perpendicular * x_coords + intercept_perpendicular) & white_mask
        region_2 = (y_coords >= slope_perpendicular * x_coords + intercept_perpendicular) & white_mask
        #####################################
        # 计算垂线与白色区域的所有交点
        vertical_line_mask = np.abs(
            y_coords - (slope_perpendicular * x_coords + intercept_perpendicular)) < 2  # 找到接近垂线的像素点
        valid_points = vertical_line_mask & white_mask  # 只保留白色区域内的点

        if np.any(valid_points):
            # 找到最靠近底部的点
            bottom_most_y = np.max(y_coords[valid_points])
            corresponding_x = x_coords[valid_points][np.argmax(y_coords[valid_points])]

            bottom_most_point = (int(corresponding_x), int(bottom_most_y))
            print(f"Bottom-most point on the perpendicular line within the white mask: {bottom_most_point}")
        else:
            print("No intersection found between the perpendicular line and the white region.")
    else:
        # 如果 dy == 0，垂线是垂直的
        return (0, 0), 0, 0
        # region_1 = (x_coords < b[0]) & white_mask
        # region_2 = (x_coords >= b[0]) & white_mask
    if bottom_most_point == (0, 0):
        return (0, 0), 0, 0
    # 计算两个区域的面积
    area_region_1 = np.sum(region_1)
    area_region_2 = np.sum(region_2)
    # 创建一个新的彩色图像，用于填充两个区域
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # 用不同的颜色填充两个区域
    colored_image[white_mask & region_1] = [255, 0, 0]  # 色 (Region 1)
    colored_image[white_mask & region_2] = [0, 255, 0]  # 色 (Region 2)
    # 在彩色图像上绘制原始线段和垂线
    cv2.circle(colored_image, bottom_most_point, 1, (255, 255, 22), 15)
    cv2.circle(colored_image, a, 1, (25, 255, 225), 15)
    cv2.circle(colored_image, b, 1, (255, 25, 225), 15)

    cv2.line(colored_image, a, b, (255, 0, 255), 2)  # 色线段
    cv2.line(colored_image, b_perpendicular_1, b_perpendicular_2, (255, 0, 255), 2)  # 色垂线
    # 显示结果图像

    # plt.figure(figsize=(10, 10))
    # plt.title(image_path)
    # plt.imshow(colored_image)
    # plt.show()
    # 打印两个区域的面积
    print(f"Area of Region 1: {area_region_1} pixels")
    print(f"Area of Region 2: {area_region_2} pixels")
    return bottom_most_point, area_region_1, area_region_2


# 主函数
def cacul_leftandright(image_path, center, radius):
    # 读取图像
    image = load_image(image_path)
    # 找到圆周上的交点
    intersection_points = find_intersection_points(image, center, radius, epsilon=8)
    b_point = (0, 0)
    if len(intersection_points) > 0:
        b_point = intersection_points[0]
        print('b_point: ', b_point)
    # 计算过点 B 的垂线方程
    perp_line = perpendicular_line_equation(center, b_point)
    # 找到垂线与白色区域的交点
    perp_intersection_points = find_perpendicular_intersection_points(image, perp_line)
    # 绘制结果
    if b_point == (0, 0):
        return (0, 0), ((0, 0), 0, 0)
    else:
        return b_point, UseLine(center, b_point, image_path)


def process_images(root_directory, csv_path):
    # 打开原始 CSV 文件和新的 CSV 文件
    with open(csv_path, 'r', newline='') as csvfile, \
            open('PACG数据中的aca750.csv', 'w', newline='', encoding='utf-8') as new_csvfile:
        csv_reader = csv.DictReader(csvfile)
        fieldnames = csv_reader.fieldnames + ['imgname'] + ['leftPoint250'] + ['leftLen250'] + ['leftAca250'] + [
            'rightPoint250'] + ['rightLen250'] + ['rightAca250'] \
                     + ['leftPoint500'] + ['leftLen500'] + ['leftAca500'] + ['rightPoint500'] + ['rightLen500'] + [
                         'rightAca500'] + \
                     ['leftPoint750'] + ['leftLen750'] + ['leftAca750'] + ['rightPoint750'] + ['rightLen750'] + [
                         'rightAca750']
        csv_writer = csv.DictWriter(new_csvfile, fieldnames=fieldnames)

        # 写入 CSV 文件的标题行
        csv_writer.writeheader()

        for root, dirs, files in os.walk(root_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_path = os.path.join(root, file)
                    # 从文件名中提取图像名称
                    # image_name = os.path.splitext(file)[0]
                    image_name = file[:-4]
                    # 查找匹配图像名称的行
                    for row in csv_reader:
                        if (image_name in row['filename']) or (row['filename'] in image_name):
                            #############################################################################################需要修改
                            if float((row['SS1 X'])) > 0.5 or float((row['SS2 X'])) < 0.5:
                                continue
                            # 提取坐标点
                            img = cv2.imread(image_path)
                            # print(image_path)
                            height, width, channels = img.shape
                            point1 = (int(float(row['SS1 X']) * width), int(float(row['SS1 Y']) * height))
                            point2 = (int(float(row['SS2 X']) * width), int(float(row['SS2 Y']) * height))
                            ##############################################################################################需要修改

                            print(point1)
                            # 画图
                            # if showpoint_num<10:
                            #     showpoint_num+=1
                            #     img = cv2.imread(image_path)
                            #     height, width, channels = img.shape
                            #     plt.figure(figsize=(8, 6))
                            #     plt.imshow(img, cmap='gray')
                            #     plt.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-')  # 线段
                            #
                            #     # plt.plot([point0[0], 1 + point0[0]], [point0[1], 1 + point0[1]], 'y*-',
                            #     #          linewidth=1.5)  # 点
                            #
                            #     plt.xlim(0, width)
                            #     plt.ylim(height, 0)
                            #     plt.show()

                            # 更新行数据
                            # 图像宽度（像素）image_height = 1868  # 图像高度（像素）;;; real_width = 16  # 实际宽度（毫米） real_height = 13  # 实际高度（毫米）

                            ########################################################## 0.25mm
                            center = point1  # 圆心坐标 (x, y)
                            radius = 35  # 圆的半径
                            result1 = cacul_leftandright(image_path, center, radius)
                            # print(result1)
                            print("left  pointb:", result1[0], "  pointc:", result1[1][0], " area1:", result1[1][1],
                                  " area2:", result1[1][2])
                            row['leftAca250'] = result1[1][2]
                            row['leftPoint250'] = result1[0]
                            row['leftLen250'] = sqrt(
                                (result1[0][0] - result1[1][0][0]) ** 2 + (result1[0][1] - result1[1][0][1]) ** 2)
                            # image_path = file  # 替换为你的图像路径
                            center = point2  # 圆心坐标 (x, y)
                            radius = 35  # 圆的半径
                            result2 = cacul_leftandright(image_path, center, radius)
                            # print(result2)
                            print("right  pointb:", result2[0], "  pointc:", result2[1][0], " area1:", result2[1][1],
                                  " area2:", result2[1][2])
                            row['rightAca250'] = result2[1][2]
                            row['rightPoint250'] = result2[0]
                            row['rightLen250'] = sqrt(
                                (result2[0][0] - result2[1][0][0]) ** 2 + (result2[0][1] - result2[1][0][1]) ** 2)
                            row['imgname'] = image_name
                            ########################################################## 0.5mm
                            center = point1  # 圆心坐标 (x, y)
                            radius = 69  # 圆的半径
                            result1 = cacul_leftandright(image_path, center, radius)
                            # print(result1)
                            print("left  pointb:", result1[0], "  pointc:", result1[1][0], " area1:", result1[1][1],
                                  " area2:", result1[1][2])
                            row['leftAca500'] = result1[1][2]
                            row['leftPoint500'] = result1[0]
                            row['leftLen500'] = sqrt(
                                (result1[0][0] - result1[1][0][0]) ** 2 + (result1[0][1] - result1[1][0][1]) ** 2)
                            # image_path = file  # 替换为你的图像路径
                            center = point2  # 圆心坐标 (x, y)
                            radius = 69  # 圆的半径
                            result2 = cacul_leftandright(image_path, center, radius)
                            # print(result2)
                            print("right  pointb:", result2[0], "  pointc:", result2[1][0], " area1:", result2[1][1],
                                  " area2:", result2[1][2])
                            row['rightAca500'] = result2[1][2]
                            row['rightPoint500'] = result2[0]
                            row['rightLen500'] = sqrt(
                                (result2[0][0] - result2[1][0][0]) ** 2 + (result2[0][1] - result2[1][0][1]) ** 2)
                            # row['imgname']= image_name

                            ########################################################## 0.75mm
                            center = point1  # 圆心坐标 (x, y)
                            radius = 105  # 圆的半径
                            result1 = cacul_leftandright(image_path, center, radius)
                            # print(result1)
                            print("left  pointb:", result1[0], "  pointc:", result1[1][0], " area1:", result1[1][1],
                                  " area2:", result1[1][2])
                            row['leftAca750'] = result1[1][2]
                            row['leftPoint750'] = result1[0]
                            row['leftLen750'] = sqrt(
                                (result1[0][0] - result1[1][0][0]) ** 2 + (result1[0][1] - result1[1][0][1]) ** 2)
                            # image_path = file  # 替换为你的图像路径
                            center = point2  # 圆心坐标 (x, y)
                            radius = 105  # 圆的半径
                            result2 = cacul_leftandright(image_path, center, radius)
                            # print(result2)
                            print("right  pointb:", result2[0], "  pointc:", result2[1][0], " area1:", result2[1][1],
                                  " area2:", result2[1][2])
                            row['rightAca750'] = result2[1][2]
                            row['rightPoint750'] = result2[0]
                            row['rightLen750'] = sqrt(
                                (result2[0][0] - result2[1][0][0]) ** 2 + (result2[0][1] - result2[1][0][1]) ** 2)
                            # row['imgname'] = image_name

                            # 写入更新后的行到新的 CSV 文件
                            csv_writer.writerow(row)
                            break  # 停止在 CSV 文件中继续查找匹配的行
                    # 将原始 CSV 读取器的位置重置为文件开头，以便下一次迭代时重新开始读取
                    csvfile.seek(0)
        print("New CSV file created: new_output.csv")


##############################################################################################需要修改

root_directory = r'C://srp_OCT/para_measure/aca/1pacg_qf'  # 根文件夹路径
csv_path = r'C://srp_OCT/para_measure/aca/1PACGall.csv'  # CSV 文件路径

process_images(root_directory, csv_path)
